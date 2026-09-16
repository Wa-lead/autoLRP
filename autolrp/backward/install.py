r"""Backward-hook installers: ``install_X(node, config)`` reads the saved
tensors of one node, resolves the rule, and registers a hook that
replaces the node's gradient inputs by relevance shares. It returns
the hook handle, or ``None`` when the native gradient is the right
routing (shape ops, selections) or the node saved nothing.
"""

import torch

from . import lrp_utils
from .lrp_utils import (
    stabilize,
    conv_ops,
    conv_transposed_ops,
    mm_ops,
    reduce_to_shape,
    apply_bias_split,
    topk_filter,
    cache_pair,
)
from .rules import passthrough as _rules_passthrough
from ..nodes import saved_tensors
from .graph import node_facts, parents, reaches_input, find_bias
from .resolve import resolve, _trace


_MISSING_LINEAR_STATE = (
    'its saved input/weight are unavailable -- the usual cause is parameters with requires_grad=False; call p.requires_grad_(True) on the model parameters')

# ---- helpers ---------------------------------------------------------

def _match_dtype(returned: tuple, original_gi: tuple) -> tuple:
    r"""Cast each returned tensor to the dtype of its position in
    ``original_gi``; autograd rejects a hook that changes a position's dtype,
    which happens when the rule math promotes bf16/fp16.
    """
    out = []
    for r, o in zip(returned, original_gi):
        if (r is not None and o is not None
                and isinstance(r, torch.Tensor)
                and isinstance(o, torch.Tensor)
                and r.dtype != o.dtype):
            out.append(r.to(o.dtype))
        else:
            out.append(r)
    # Preserve any trailing positions beyond the shorter sequence.
    if len(returned) > len(original_gi):
        out.extend(returned[len(original_gi):])
    return tuple(out)


_MISSING_STATE_WARNED: set = set()    # node names already warned about


def _warn_missing_state(node, detail: str) -> None:
    r"""Warn once per node name when an installer matched but the node did
    not save what the rule needs, so the native gradient runs there.
    """
    if node_facts(node).get('expected_native'):
        return                      # caller marked this fallback intentional
    name = node.name()
    if name in _MISSING_STATE_WARNED:
        return
    _MISSING_STATE_WARNED.add(name)
    import warnings
    warnings.warn(
        f"autoLRP: {name} matched an installer but {detail}; relevance "
        f"at this op degrades to plain gradient.", UserWarning,
        # 4 skips warn -> here -> installer -> dispatcher, so the warning
        # points at the call that triggered the dispatch.
        stacklevel=4)


# ---------------------------------------------------------------------------
# Delivery: from shares to the tuple autograd accepts
# ---------------------------------------------------------------------------

def as_grad_tuple(gi, shares):
    r"""``{position: R_share}`` to the tuple autograd accepts. A share is
    sum-reduced to the operand's shape and cast to its dtype; ``None``
    means the operand is not attributed and gets zeros, not its native
    gradient. Positions autograd gave as ``None`` stay ``None``;
    positions not in ``shares`` are untouched.
    """
    out = list(gi)
    for position, share in shares.items():
        if position < len(gi) and gi[position] is not None:
            out[position] = torch.zeros_like(gi[position]) if share is None else reduce_to_shape(share, gi[position].shape)
    return _match_dtype(tuple(out), gi)


def _reduce_gqa(r, n_rep):
    r"""Backward of ``repeat_interleave(n_rep, dim=1)``: sum relevance over the
    ``n_rep`` repeated key/value head groups,
    ``(B, Hkv*n_rep, S, D) -> (B, Hkv, S, D)``."""
    B, H, S, D = r.shape
    return r.reshape(B, H // n_rep, n_rep, S, D).sum(2)


class _Named:
    r"""A node presented to ``resolve`` under another name, with its own
    facts kept and ``facts`` added. For the fused attention node, which
    is one node standing for two products and a softmax."""
    __slots__ = ('metadata', '_name')

    def __init__(self, real, name, facts=None):
        self.metadata = {'lrp': {**node_facts(real), **(facts or {})}}
        self._name = name

    def name(self):
        return self._name


# ---- two-operand nodes ------------------------------------------------
#
# A rule of a two-operand table returns (R_a, R_b), None for an operand
# it does not attribute; which operands, ``attribute``, comes out of
# resolve with the rule: from the entry, else from the node's facts.

def install_matmul(node, config):
    r"""``z = a @ b (+ bias)``: addmm, mm and bmm."""
    a, b = saved_tensors(node)
    if a.tensor is None or b.tensor is None:
        _warn_missing_state(node, _MISSING_LINEAR_STATE)
        return
    bias = find_bias(node, position=0) if a.position == 1 else None    # addmm keeps its bias at 0
    fn, kw = resolve(config.rule, node)
    fwd0, bwd_a, bwd_b = mm_ops()
    eps, rf = config.eps, config.relevance_filter

    def _hook(gi, go, _a=a.tensor, _b=b.tensor):
        with torch.no_grad():
            fwd = cache_pair(fwd0)                       # z = fwd(a, b) once, shared with the rule
            R_out = apply_bias_split(go[0], fwd(_a, _b), bias, eps) if bias is not None else go[0]
            R_a, R_b = fn(_a, _b, R_out, eps, fwd, bwd_a, bwd_b, **kw)
            if rf < 1.0:
                R_a, R_b = (None if R is None else topk_filter(R, rf) for R in (R_a, R_b))
        return as_grad_tuple(gi, {a.position: R_a, b.position: R_b})
    return node.register_hook(_hook)


def install_conv(node, config):
    r"""``ConvolutionBackward``: ``Conv1d``/``2d``/``3d`` and their transposed
    forms. The image is the first operand, the kernel its weight."""
    x, w = saved_tensors(node)
    if x.tensor is None or w.tensor is None:
        _warn_missing_state(node, _MISSING_LINEAR_STATE)
        return
    xt, wt = x.tensor, w.tensor
    ndim = xt.ndim - 2
    stride = getattr(node, '_saved_stride', (1,) * ndim)
    padding = getattr(node, '_saved_padding', (0,) * ndim)
    dilation = getattr(node, '_saved_dilation', (1,) * ndim)
    groups = getattr(node, '_saved_groups', 1)
    bias = find_bias(node, position=2)
    if bias is not None:
        bias = bias.view([1, -1] + [1] * ndim)
    if getattr(node, '_saved_transposed', False):
        output_padding = getattr(node, '_saved_output_padding', (0,) * ndim)
        fwd0, bwd_x, bwd_w = conv_transposed_ops(ndim, stride, padding, output_padding, dilation, groups, xt, wt)
    else:
        fwd0, bwd_x, bwd_w = conv_ops(ndim, stride, padding, dilation, groups, xt.shape)
    fn, kw = resolve(config.rule, node)
    eps, rf = config.eps, config.relevance_filter

    def _hook(gi, go):
        with torch.no_grad():
            fwd = cache_pair(fwd0)
            R_out = apply_bias_split(go[0], fwd(xt, wt), bias, eps) if bias is not None else go[0]
            R_x, R_w = fn(xt, wt, R_out, eps, fwd, bwd_x, bwd_w, **kw)
            if rf < 1.0 and R_x is not None:
                R_x = topk_filter(R_x, rf)
        return as_grad_tuple(gi, {x.position: R_x, w.position: R_w})
    return node.register_hook(_hook)


# ---- distributions (mul / div / add) ---------------------------------

def install_mul(node, config):
    r"""``MulBackward``: the entry that addresses the node picks from
    ``MUL_RULES``; with one operand from the input (``weight_operand``)
    the rule attributes that side alone, all of the relevance. A native
    mul saves an operand only if the other one needs a gradient, so with
    a constant on one side the input is not saved; a one-sided rule does
    not read it."""
    a, b = saved_tensors(node)
    fn, kw = resolve(config.rule, node)
    eps = config.eps

    def _hook(gi, go, _a=a.tensor, _b=b.tensor):
        with torch.no_grad():
            R_a, R_b = fn(_a, _b, go[0], eps, **kw)
        return as_grad_tuple(gi, {a.position: R_a, b.position: R_b})
    return node.register_hook(_hook)


def install_div(node, config):
    r"""``DivBackward``, as :func:`install_mul`; the second operand enters
    as its stabilized reciprocal so the split matches ``a * (1/b)``."""
    a, b = saved_tensors(node)
    b_inv = None if b.tensor is None else 1.0 / stabilize(b.tensor, config.eps)
    fn, kw = resolve(config.rule, node)
    eps = config.eps

    def _hook(gi, go, _a=a.tensor, _b=b_inv):
        with torch.no_grad():
            R_a, R_b = fn(_a, _b, go[0], eps, **kw)
        return as_grad_tuple(gi, {a.position: R_a, b.position: R_b})
    return node.register_hook(_hook)


def install_add(node, config):
    r"""Our ``AddBackward``/``SubBackward`` (saved operands): the entry that
    addresses the node picks from ``ADD_RULES``. A native add or sub saved
    nothing and its gradient runs: a constant summand is a bias."""
    a, b = saved_tensors(node)
    if a.tensor is None or b.tensor is None:
        return
    fn, kw = resolve(config.rule, node)
    eps = config.eps

    def _hook(gi, go, _a=a.tensor, _b=b.tensor):
        with torch.no_grad():
            R_a, R_b = fn(_a, _b, go[0], eps, **kw)
        return as_grad_tuple(gi, {a.position: R_a, b.position: R_b})
    return node.register_hook(_hook)


# ---- reductions (norm / cumsum / mean / sum) -------------------------

def install_norm(node, config):
    r"""``x.norm()`` / ``linalg.vector_norm``: a reduction whose native
    gradient is ``x / ||x||``, a scaling. Share the output relevance
    over the reduced elements in proportion to ``|x|``, the same policy
    as our mean and sum."""
    x, = (t.tensor for t in saved_tensors(node))
    if x is None:
        _warn_missing_state(
            node, 'its saved input is unavailable -- the usual cause is '
            'parameters with requires_grad=False; call p.requires_grad_(True) '
            'on the model parameters')
        return
    dim = getattr(node, '_saved_dim', None)
    keepdim = bool(getattr(node, '_saved_keepdim', False))
    eps = config.eps
    rule, kw = resolve(config.rule, node)

    def _hook(gi, go, _x=x, _dim=dim, _keep=keepdim, _eps=eps, _rule=rule, _kw=kw):
        R_in = _rule(_x, None, go[0], _eps, dim=_dim, keepdim=_keep, **_kw)
        return as_grad_tuple(gi, {0: R_in})
    return node.register_hook(_hook)


def install_cumsum(node, config):
    r"""Our ``CumsumBackward``: the rule in ``config.rule`` (``epsilon``,
    :func:`~autolrp.backward.rules.cumsum_epsilon`). A native node saved
    nothing and keeps its gradient."""
    x, = (t.tensor for t in saved_tensors(node))
    if x is None or not hasattr(node, 'dim'):
        return                                         # native: nothing to read
    dim = node.dim
    eps = config.eps
    rule, kw = resolve(config.rule, node)

    def _hook(gi, go, _x=x, _dim=dim, _eps=eps, _rule=rule, _kw=kw):
        with torch.no_grad():
            R_in = _rule(_x, None, go[0], _eps, dim=_dim, **_kw)
        return as_grad_tuple(gi, {0: R_in})
    return node.register_hook(_hook)


def install_mean_or_sum(node, config):
    r"""Our ``MeanBackward``/``SumBackward``: share ``R_out`` over the
    reduced elements in proportion to ``|x|``. A native node saved
    nothing and keeps its gradient.
    """
    activation, = (t.tensor for t in saved_tensors(node))
    if activation is None:
        return

    if not hasattr(node, 'dim'):
        return  # not our wrapper -- can't recover dim/keepdim
    dim, keepdim = node.dim, getattr(node, 'keepdim', False)
    rule, kw = resolve(config.rule, node)

    def _hook(gi, go, _x=activation, _dim=dim, _keepdim=keepdim,
              _eps=config.eps, _rule=rule, _kw=kw):
        R_in = _rule(_x, None, go[0], _eps, dim=_dim, keepdim=_keepdim, **_kw)
        return as_grad_tuple(gi, {0: R_in})

    return node.register_hook(_hook)


# ---------------------------------------------------------------------------
# One-operand installers: elementwise, softmax, layer norm
# ---------------------------------------------------------------------------

def _install_unary(node, config, x, y, **saved):
    r"""Register the hook of a one-operand node: ``rule(x, y, R_out, eps,
    **saved, **kw)`` into position 0, the rule resolved from ``config.rule``.
    A passthrough entry installs the plain passthrough hook."""
    fn, kw = resolve(config.rule, node)
    if fn is _rules_passthrough:
        return _passthrough_hook(node)
    eps = config.eps

    def _hook(gi, go, _fn=fn, _kw=kw, _x=x, _y=y, _saved=saved, _eps=eps):
        with torch.no_grad():
            R_in = _fn(_x, _y, go[0], _eps, **_saved, **_kw)
        return as_grad_tuple(gi, {0: R_in})
    return node.register_hook(_hook)


def _nothing_saved(node, config):
    """A rule other than passthrough asked for state the node did not keep:
    say so once, report it, pass relevance through."""
    fn, _ = resolve(config.rule, node)
    if fn is not _rules_passthrough:
        _warn_missing_state(node, f"{fn.__name__} needs the node's input and "
                                  f"output, which it did not save")
        _trace(None, f'{fn.__name__}: nothing saved, passthrough')
    return _passthrough_hook(node)


def install_elementwise(node, config):
    r"""Our elementwise wrappers (``forward/ops.py``) save ``(x, y)``; a
    native node is a calling form the rewrite declined."""
    x, y = (t.tensor for t in saved_tensors(node))
    if x is None or y is None:
        return _nothing_saved(node, config)
    return _install_unary(node, config, x, y)


def install_softmax(node, config):
    r"""Our ``Softmax`` saves ``(x, y)`` and its ``dim``; a native softmax
    saved only ``y``."""
    x, y = (t.tensor for t in saved_tensors(node))
    if x is None or y is None:
        return _nothing_saved(node, config)
    return _install_unary(node, config, x, y, dim=getattr(node, 'dim', -1))


def install_layernorm(node, config):
    r"""Native ``NativeLayerNormBackward``: input, affine weight and bias,
    and the saved mean and rstd; the normalized shape is a parameter."""
    x, weight, bias, mean, rstd = (t.tensor for t in saved_tensors(node))
    ns = getattr(node, '_saved_normalized_shape', None)
    if x is None or ns is None:
        return _nothing_saved(node, config)
    return _install_unary(node, config, x, None, normalized_shape=ns, weight=weight, bias=bias,
                          mean=mean, rstd=rstd)


# ---- passthrough -----------------------------------------------------

def _passthrough_hook(node):
    def _hook(gi, go):
        return as_grad_tuple(gi, {0: go[0]})
    return node.register_hook(_hook)


def install_passthrough(node, config):
    r"""``R_in = R_out`` into position 0: norms without a rule table,
    sign-preserving unaries, and any rule-bearing node that saved nothing
    its rule needs. ``None`` positions stay ``None``.
    """
    _trace(None, 'passthrough')
    return _passthrough_hook(node)


def install_noop(node, config):
    r"""No-op installer: autograd-native VJP correctly routes
    :math:`R` for shape-routing ops (Reshape, View, Permute, ...).
    """
    pass


# ---- fused attention (consults the BmmBackward rule entry) ----

def _fused_shares(config, node, live_left, live_right, weights_on_left):
    r"""The relevance function for one product ``L @ Rt`` inside a fused
    attention node: ``(L, Rt, R_out) -> (R_L, R_Rt)``, or ``None`` when
    neither operand comes from the input. Resolved as the decomposed
    graph's bmm would be: both live, the ``bilinear`` fact (and
    ``attention_weights: 0`` for ``A @ V``, whose weights are on its left
    by construction); one live, ``weight_operand`` on the other."""
    eps = config.eps
    if not (live_left or live_right):
        return None
    if live_left and live_right:
        facts = {'bilinear': True, 'attention_weights': 0} if weights_on_left else {'bilinear': True}
    else:
        facts = {'weight_operand': 1 if live_left else 0}
    fn, kw = resolve(config.rule, _Named(node, 'BmmBackward0', facts))
    fwd, bwd_a, bwd_b = mm_ops()

    def run(L, Rt, R):
        R_L, R_Rt = fn(L, Rt, R, eps, fwd, bwd_a, bwd_b, **kw)
        return (torch.zeros_like(L) if R_L is None else R_L,       # not attributed: zeros
                torch.zeros_like(Rt) if R_Rt is None else R_Rt)
    return run


def install_sdpa(node, config):
    r"""Fused scaled-dot-product attention node, used when
    :func:`autolrp.set_decompose_attention` is off. Reconstructs the
    attention matrix from the saved query, key, value and logsumexp, then
    runs the same rules the decomposed graph would: each of the two
    products resolved by which operands come from the input
    (:func:`_fused_shares`), the softmax addressed as ``SoftmaxBackward``
    through :class:`_Named`.
    Writes the query, key and value positions; GQA expands K and V and sums
    the relevance back over the repeated heads.
    """
    q, k, v = (t.tensor for t in saved_tensors(node))
    lse = getattr(node, '_saved_logsumexp', None)
    if lse is None:
        lse = getattr(node, '_saved_log_sumexp', None)
    if q is None or k is None or v is None or lse is None:
        _warn_missing_state(node, 'this SDPA backend did not save query/key/value/logsumexp; use autolrp.set_decompose_attention(True) for full rule coverage')
        return
    mask = getattr(node, '_saved_attn_mask', None)
    if mask is None:
        mask = getattr(node, '_saved_attn_bias', None)
    is_causal = bool(getattr(node, '_saved_is_causal', False))
    scale = getattr(node, '_saved_scale', None)
    eps = config.eps
    # One node stands for softmax and two products; each product is
    # resolved on its own, as the decomposed graph would.
    ps = parents(node, skip_aliases=False)
    live_q, live_k, live_v = (i < len(ps) and ps[i] is not None and reaches_input(ps[i]) for i in range(3))
    live_a = live_q or live_k
    _av = _fused_shares(config, node, live_a, live_v, weights_on_left=True)
    _qk = _fused_shares(config, node, live_q, live_k, weights_on_left=False)
    if _av is None and _qk is None:
        return
    softmax_fn, _ = resolve(config.rule, _Named(node, 'SoftmaxBackward0'))

    def _hook(gi, go, _q=q, _k=k, _v=v, _lse=lse, _mask=mask,
              _is_causal=is_causal, _scale=scale, _eps=eps,
              _av=_av, _qk=_qk, _softmax=softmax_fn):
        with torch.no_grad():
            R_out = go[0]
            B, Hq, T, D = _q.shape
            S = _k.shape[2]
            # Rebuild A = softmax(scores) and the GQA-expanded K/V from
            # the saved state.
            A, scores, k_e, v_e, sc, n_rep = lrp_utils.reconstruct_sdpa(
                _q, _k, _v, _lse, _mask, _is_causal, _scale)

            n = B * Hq
            fl = lambda t: t.reshape(n, *t.shape[2:])
            # attn @ V: the fused signature names the operands, A = weights
            Af, Vf = fl(A), fl(v_e)
            R_A, R_V = _av(Af, Vf, fl(R_out)) if _av else (torch.zeros_like(Af), torch.zeros_like(Vf))
            R_A = torch.zeros_like(Af) if R_A is None else R_A
            R_V = torch.zeros_like(Vf) if R_V is None else R_V
            # softmax: R_A -> R_scores (relevance at the scaled-masked scores)
            R_scores = _softmax(fl(scores), fl(A), R_A, _eps, dim=-1)
            # The `*scale` is a constant multiplication: relevance passes
            # through it unchanged, as in the decomposed path.
            Qf, Ktf = fl(_q), fl(k_e.transpose(-2, -1))
            R_Q, R_Kt = _qk(Qf, Ktf, R_scores) if _qk else (torch.zeros_like(Qf), torch.zeros_like(Ktf))
            R_Q = torch.zeros_like(Qf) if R_Q is None else R_Q
            R_Kt = torch.zeros_like(Ktf) if R_Kt is None else R_Kt
            R_q = R_Q.reshape(B, Hq, T, D)
            R_v = R_V.reshape(B, Hq, S, D)
            R_k = R_Kt.transpose(-2, -1).reshape(B, Hq, S, D)
            if n_rep > 1:                                          # GQA: fold heads back
                R_k = _reduce_gqa(R_k, n_rep)
                R_v = _reduce_gqa(R_v, n_rep)
        out = [None] * len(gi)
        if len(gi) > 0 and gi[0] is not None:
            out[0] = R_q
        if len(gi) > 1 and gi[1] is not None:
            out[1] = R_k
        if len(gi) > 2 and gi[2] is not None:
            out[2] = R_v
        return _match_dtype(tuple(out), gi)

    return node.register_hook(_hook)
