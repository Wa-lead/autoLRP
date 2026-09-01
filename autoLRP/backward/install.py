r"""Backward-hook installers: ``install_X(node, config)`` reads the saved
tensors of one node, resolves the rule, and registers a hook that
replaces the node's gradient inputs by relevance shares. It returns
the hook handle, or ``None`` when the native gradient is the right
routing (shape ops, selections) or the node saved nothing.
"""
import torch
import torch.nn.functional as F

from . import lrp_utils
from .lrp_utils import (
    run_linear_rule,
    stabilize,
    conv_ops,
    conv_transposed_ops,
    reduction_share,
    apply_bias_split,
)
from .rules import LINEAR_RULES, MUL_RULES, BMM_RULES, ADD_RULES
from .analysis import node_facts, parents, reaches_input
from .resolve import resolve, match, _normalize
from ..utils import find_bias


_MISSING_LINEAR_STATE = (
    'its saved input/weight are unavailable -- the usual cause is parameters with requires_grad=False; call p.requires_grad_(True) on the model parameters')

# ---- helpers ---------------------------------------------------------

def _match_dtype(returned: tuple, original_gi: tuple) -> tuple:
    r"""Cast each returned tensor to the dtype of its slot in
    ``original_gi``; autograd rejects a hook that changes a slot's dtype,
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
    # Preserve any trailing slots beyond the shorter sequence.
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
# Shared machinery: relevance plumbing, rule resolution, tracing
# ---------------------------------------------------------------------------

def _reduce_to(t: torch.Tensor, target) -> torch.Tensor:
    r"""Sum-reduce ``t`` to match ``target``'s shape.

    Returns ``t`` unchanged when ``target`` is ``None`` or not a
    tensor, or when the shapes already match.
    """
    if target is None or not isinstance(target, torch.Tensor) \
            or t.shape == target.shape:
        return t
    return lrp_utils.reduce_to_shape(t, target.shape)

def as_grad_tuple(gi, slot_shares):
    r"""``{slot: R_share}`` to the tuple autograd accepts: slots autograd
    gave as ``None`` stay ``None``, shares are sum-reduced to the operand's
    shape, dtypes match ``gi``. Slots not in ``slot_shares`` are untouched.
    """
    out = list(gi)
    for sl, share in slot_shares.items():
        if sl < len(gi) and gi[sl] is not None and share is not None:
            out[sl] = _reduce_to(share, gi[sl])
    return _match_dtype(tuple(out), gi)

_TRACE = None                     # list of (key, what) while explain() runs


def _trace(key, what):
    if _TRACE is not None:
        _TRACE.append((key, what))


def resolve_rule(mapping, node, registry):
    r"""The rule function and keyword arguments for ``node`` from the
    config's ``rule`` dict, looked up in ``registry`` (the family's table).
    The entry is picked by :func:`~autoLRP.backward.resolve.match`; a
    ``'detach'`` entry becomes ``detach_lhs``/``detach_rhs`` from the slot
    fact named by its ``by=``, or the family's fallback when the node has
    no such fact. A name the family cannot run is an error.
    """
    key = match(mapping, node)
    spec = mapping[key]
    name, kw = _normalize(spec)
    if name == 'detach':
        name, kw = _detach_side(kw, node, registry)
    if callable(name):
        fn = name
    elif name not in registry:
        raise ValueError(
            f"entry {key!r}={spec!r} addresses {node.name()}, whose family "
            f"cannot run {name!r}. Valid here: {sorted(registry)} and "
            f"'detach'")
    else:
        fn = registry[name]
    _trace(key, getattr(fn, '__name__', repr(fn)))
    return fn, kw


def _detach_side(kw, node, registry):
    kw = dict(kw)                     # never mutate the config's entry
    by = kw.pop('by', None)
    if not isinstance(by, str) or not by:
        raise ValueError(
            "'detach' needs by=<fact name>, e.g. "
            "('detach', {'by': 'weights_operand'}); 'detach_lhs' / "
            "'detach_rhs' name a side directly")
    facts = node_facts(node)
    if by not in facts:
        return registry.default, {}   # no referent: the family fallback
    slot = facts[by]
    if isinstance(slot, bool) or slot not in (0, 1):
        raise ValueError(
            f"'detach' by={by!r}: on {node.name()} that fact is {slot!r}, "
            f"not a side (0 for the left operand, 1 for the right)")
    return ('detach_lhs' if slot == 0 else 'detach_rhs'), kw


def _live_slots(node):
    r"""Operand slots of a two-operand node that relevance can flow to.
    A constant operand (a Python number, a tensor without grad) has no
    parent node, so ``next_functions`` holds ``None`` there."""
    ps = parents(node, skip_aliases=False)
    return [i for i in (0, 1) if i < len(ps) and ps[i] is not None]


def _install_single_operand(node, slot):
    r"""One live operand: the op is a scaling or a sign change of that
    operand by a constant, and all relevance passes to it unchanged."""
    _trace(None, 'passthrough (constant operand)')

    def _hook(gi, go, _slot=slot):
        return as_grad_tuple(gi, {_slot: go[0]})
    return node.register_hook(_hook)


def _reduce_gqa(r, n_rep):
    r"""Backward of ``repeat_interleave(n_rep, dim=1)``: sum relevance over the
    ``n_rep`` repeated key/value head groups,
    ``(B, Hkv*n_rep, S, D) -> (B, Hkv, S, D)``."""
    B, H, S, D = r.shape
    return r.reshape(B, H // n_rep, n_rep, S, D).sum(2)


class _Named:
    r"""A node presented under another name for config addressing, with
    its own facts kept."""
    __slots__ = ('metadata', '_name')

    def __init__(self, real, name):
        self.metadata = {'lrp': dict(node_facts(real))}
        self._name = name

    def name(self):
        return self._name


# ---- linear family (addmm / mm / conv / bmm) -------------------------

def _input_slots(node, slots):
    r"""Which of ``slots`` hold an operand that reaches the wrapped input."""
    ps = parents(node, skip_aliases=False)
    return [i for i in slots if i < len(ps) and reaches_input(ps[i])]


def _install_product(node, config, a, b, slot_a, slot_b, bias=None):
    r"""Common installer for ``z = a @ b (+ bias)``: addmm, mm, bmm. Which
    operands reach the wrapped input decides the family. One: a linear
    layer with the other operand as weight, ``LINEAR_RULES``, all relevance
    to that operand; an ``Addmm``/``Mm`` node is addressed by its own name,
    a ``Bmm`` node as ``MmBackward``. Both: bilinear, addressed as
    ``BmmBackward``, ``BMM_RULES``. Neither: nothing installs.
    """
    live = _input_slots(node, (slot_a, slot_b))
    if not live:
        return
    eps = config.eps
    rf = config.relevance_filter
    T = lambda t: t.transpose(-2, -1)

    if len(live) == 2:
        addr = node if 'BmmBackward' in node.name() else _Named(node, 'BmmBackward0')
        fn, kw = resolve_rule(config.rule, addr, BMM_RULES)
        fwd = lambda x, y: torch.matmul(x, y)
        bwd_a = lambda o, s: torch.matmul(s, T(o))
        bwd_b = lambda o, s: torch.matmul(T(o), s)

        def _hook(gi, go, _fn=fn, _a=a, _b=b, _kw=kw, _bias=bias):
            with torch.no_grad():
                R_out = go[0]
                if _bias is not None:
                    R_out = apply_bias_split(R_out, fwd(_a, _b), _bias, eps)
                R_a, R_b = _fn(_a, _b, R_out, eps, fwd, bwd_a, bwd_b, **_kw)
            return as_grad_tuple(gi, {slot_a: R_a, slot_b: R_b})
        return node.register_hook(_hook)

    addr = node if 'BmmBackward' not in node.name() else _Named(node, 'MmBackward0')
    rule_fn, rule_kwargs = resolve_rule(config.rule, addr, LINEAR_RULES)
    if live[0] == slot_a:                       # x @ w
        x, w, x_slot = a, b, slot_a
        fwd = lambda x_, w_: torch.matmul(x_, w_)
        bwd_a = lambda w_, s: torch.matmul(s, T(w_))
        bwd_b = lambda x_, s: torch.matmul(T(x_), s)
    else:                                       # w @ x
        x, w, x_slot = b, a, slot_b
        fwd = lambda x_, w_: torch.matmul(w_, x_)
        bwd_a = lambda w_, s: torch.matmul(T(w_), s)
        bwd_b = lambda x_, s: torch.matmul(s, T(x_))

    def _hook(gi, go, _x=x, _w=w, _b=bias, _slot=x_slot,
              _rule_fn=rule_fn, _rule_kw=rule_kwargs,
              _fwd=fwd, _ba=bwd_a, _bb=bwd_b):
        R_in, _ = run_linear_rule(
            _x, _w, _b, go[0], _rule_fn, _rule_kw,
            _fwd, _ba, _bb, eps, relevance_filter=rf)
        return as_grad_tuple(gi, {_slot: R_in})
    return node.register_hook(_hook)


def install_addmm(node, config):
    r"""``addmm(bias, mat1, mat2)``: slots 0 (bias), 1, 2."""
    a = getattr(node, '_saved_mat1', None)
    b = getattr(node, '_saved_mat2', None)
    if a is None or b is None:
        _warn_missing_state(node, _MISSING_LINEAR_STATE)
        return
    return _install_product(node, config, a, b, 1, 2,
                            bias=find_bias(node, parent_idx=0))


def install_mm(node, config):
    r"""``mm(self, mat2)``: slots 0, 1."""
    a = getattr(node, '_saved_self', None)
    b = getattr(node, '_saved_mat2', None)
    if a is None or b is None:
        _warn_missing_state(node, _MISSING_LINEAR_STATE)
        return
    return _install_product(node, config, a, b, 0, 1)


def install_conv(node, config):
    r"""Install the LRP hook for a ``ConvolutionBackward`` node.

    Covers ``Conv1d`` / ``Conv2d`` / ``Conv3d`` and their transposed
    forms (``_saved_transposed``). The input-gradient slot is ``gi[0]``.
    """
    x = getattr(node, '_saved_input', None)
    w = getattr(node, '_saved_weight', None)
    if x is None or w is None:
        _warn_missing_state(node, _MISSING_LINEAR_STATE)
        return

    stride = getattr(node, '_saved_stride', (1,) * (x.ndim - 2))
    padding = getattr(node, '_saved_padding', (0,) * (x.ndim - 2))
    dilation = getattr(node, '_saved_dilation', (1,) * (x.ndim - 2))
    groups = getattr(node, '_saved_groups', 1)
    ndim = x.ndim - 2

    bias = find_bias(node, parent_idx=2)
    if bias is not None:
        bias = bias.view([1, -1] + [1] * ndim)

    rule_fn, rule_kwargs = resolve_rule(config.rule, node, LINEAR_RULES)
    if getattr(node, '_saved_transposed', False):
        output_padding = getattr(node, '_saved_output_padding', (0,) * ndim)
        fwd, bwd_a, bwd_b = conv_transposed_ops(
            ndim, stride, padding, output_padding, dilation, groups, x, w)
    else:
        fwd, bwd_a, bwd_b = conv_ops(ndim, stride, padding, dilation, groups, x.shape)
    eps = config.eps
    rf = config.relevance_filter

    def _hook(gi, go, _x=x, _w=w, _b=bias,
              _rule_fn=rule_fn, _rule_kw=rule_kwargs,
              _eps=eps, _fwd=fwd, _ba=bwd_a, _bb=bwd_b, _rf=rf):
        R_out = go[0]
        R_in, R_w = run_linear_rule(
            _x, _w, _b, R_out, _rule_fn, _rule_kw,
            _fwd, _ba, _bb, _eps, relevance_filter=_rf)
        return as_grad_tuple(gi, {0: R_in, 1: R_w})

    return node.register_hook(_hook)


def install_bmm(node, config):
    r"""``bmm(self, mat2)``: slots 0, 1. Two operands from the input is
    the attention case (``BMM_RULES``); one is a batched linear layer
    with a constant or parameter weight, addressed as ``MmBackward``."""
    a = getattr(node, '_saved_self', None)
    b = getattr(node, '_saved_mat2', None)
    if a is None or b is None:
        _warn_missing_state(node, _MISSING_LINEAR_STATE)
        return
    return _install_product(node, config, a, b, 0, 1)


# ---- distributions (mul / div / add) ---------------------------------

def install_mul(node, config):
    r"""``MulBackward``: one operand from the input, relevance passes to it
    unchanged; two, the entry that addresses the node (a fact, else
    ``'MulBackward'``) picks from ``MUL_RULES``. The statistic transparency
    of norms comes from the ``statistic_operand`` entry of ``BASE``.
    """
    live = _live_slots(node)
    if len(live) == 1:
        return _install_single_operand(node, live[0])
    a = getattr(node, '_saved_self', None)
    b = getattr(node, '_saved_other', None)
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return

    eps = config.eps
    fn, kw = resolve_rule(config.rule, node, MUL_RULES)
    fwd = lambda x, y: x * y
    bwd_a = lambda o, s: s * o
    bwd_b = lambda o, s: s * o

    def _hook(gi, go, _fn=fn, _a=a, _b=b, _kw=kw, _eps=eps,
              _fwd=fwd, _ba=bwd_a, _bb=bwd_b):
        with torch.no_grad():
            R_a, R_b = _fn(_a, _b, go[0], _eps, _fwd, _ba, _bb, **_kw)
        return as_grad_tuple(gi, {0: R_a, 1: R_b})

    return node.register_hook(_hook)

def install_div(node, config):
    r"""``DivBackward``, as :func:`install_mul`; for ``'proportional'`` the
    second operand enters as its stabilized reciprocal so the split
    matches ``a * (1/b)``.
    """
    live = _live_slots(node)
    if len(live) == 1:
        return _install_single_operand(node, live[0])
    a = getattr(node, '_saved_self', None)
    b = getattr(node, '_saved_other', None)
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return

    eps = config.eps
    fn, kw = resolve_rule(config.rule, node, MUL_RULES)

    def _hook(gi, go, _fn=fn, _a=a, _b=b, _kw=kw, _eps=eps):
        with torch.no_grad():
            R_out = go[0]
            b_inv = 1.0 / stabilize(_b, _eps)
            aa = _a if _a.shape == R_out.shape else _a.expand_as(R_out)
            bb = (b_inv if b_inv.shape == R_out.shape
                  else b_inv.expand_as(R_out))
            R_a, R_b = _fn(aa, bb, R_out, _eps, None, None, None, **_kw)
        return as_grad_tuple(gi, {0: R_a, 1: R_b})

    return node.register_hook(_hook)


def install_add(node, config):
    r"""Our ``AddBackward``/``SubBackward`` (saved operands): the entry that
    addresses the node picks from ``ADD_RULES``. A native add or sub saved
    nothing, and its gradient runs.
    """
    saved = getattr(node, 'saved_tensors', None)
    if not saved or len(saved) < 2:
        return
    sv, bv = saved[0], saved[1]
    ps = node.next_functions
    if len(ps) < 2:
        return

    fn, kw = resolve_rule(config.rule, node, ADD_RULES)
    eps = config.eps
    fwd = lambda x, y: x + y
    bwd_a = lambda o, s: s
    bwd_b = lambda o, s: s

    def _hook(gi, go, _fn=fn, _a=sv, _b=bv, _kw=kw, _eps=eps,
              _fwd=fwd, _ba=bwd_a, _bb=bwd_b):
        with torch.no_grad():
            R_a, R_b = _fn(_a, _b, go[0], _eps, _fwd, _ba, _bb, **_kw)
        return as_grad_tuple(gi, {0: R_a, 1: R_b})

    return node.register_hook(_hook)


# ---- reductions (norm / cumsum / mean / sum) -------------------------

def install_norm(node, config):
    r"""``x.norm()`` / ``linalg.vector_norm``: a reduction whose native
    gradient is ``x / ||x||``, a scaling. Share the output relevance
    over the reduced elements in proportion to ``|x|``, the same policy
    as our mean and sum. Warns and keeps the native gradient when the
    node did not save its input."""
    x = getattr(node, '_saved_self', None)
    if x is None:
        _warn_missing_state(
            node, 'its saved input is unavailable -- the usual cause is '
            'parameters with requires_grad=False; call p.requires_grad_(True) '
            'on the model parameters')
        return
    dim = getattr(node, '_saved_dim', None)
    keepdim = bool(getattr(node, '_saved_keepdim', False))
    if isinstance(dim, int):
        dim = (dim,)
    elif dim is not None:
        dim = tuple(dim)
        if len(dim) == 0:
            dim = None
    eps = config.eps
    _trace(None, 'reduction_share')

    def _hook(gi, go, _x=x, _dim=dim, _keep=keepdim, _eps=eps):
        R_in = reduction_share(_x, go[0], _dim, _keep, _eps)
        return as_grad_tuple(gi, {0: R_in})
    return node.register_hook(_hook)


def install_cumsum(node, config):
    r"""``cumsum``: a linear map with 0/1 weights, ``y_j = sum_{i<=j} x_i``.
    The epsilon rule gives ``R_i = x_i * sum_{j>=i} R_j / y_j``, a reversed
    cumulative sum of ``R / y``; the native gradient would hand every
    ``x_i`` the full ``R_j`` of each later output."""
    saved = getattr(node, 'saved_tensors', None)      # our Cumsum wrapper
    if not saved or not hasattr(node, 'dim'):
        return                                         # native: nothing to read
    x, dim = saved[0], node.dim
    eps = config.eps
    _trace(None, 'epsilon (cumsum)')

    def _hook(gi, go, _x=x, _dim=dim, _eps=eps):
        with torch.no_grad():
            y = torch.cumsum(_x, _dim)
            s = go[0] / stabilize(y, _eps)
            R_in = _x * torch.flip(torch.cumsum(torch.flip(s, (_dim,)), _dim), (_dim,))
        return as_grad_tuple(gi, {0: R_in})
    return node.register_hook(_hook)


def install_mean_or_sum(node, config):
    r"""Our ``MeanBackward``/``SumBackward``: share ``R_out`` over the
    reduced elements in proportion to ``|x|``. A native node saved
    nothing and keeps its gradient.
    """
    saved = getattr(node, 'saved_tensors', None)
    if not saved or len(saved) == 0:
        return
    activation = saved[0]

    if not hasattr(node, 'dim'):
        return  # not our wrapper -- can't recover dim/keepdim
    raw_dim = node.dim
    keepdim = getattr(node, 'keepdim', False)

    # reduction_share wants None or a tuple of ints.
    if raw_dim is None:
        dim = None
    elif isinstance(raw_dim, int):
        dim = (raw_dim,)
    else:
        dim = tuple(raw_dim)
    _trace(None, 'reduction_share')

    def _hook(gi, go, _act=activation, _dim=dim, _keepdim=keepdim,
              _eps=config.eps):
        R_out = go[0]
        with torch.no_grad():
            R_in = lrp_utils.reduction_share(_act, R_out, _dim, _keepdim, _eps)
        return as_grad_tuple(gi, {0: R_in})

    return node.register_hook(_hook)


# ---------------------------------------------------------------------------
# Softmax variants (chosen by config.softmax)
# ---------------------------------------------------------------------------

def _install_softmax_saved(node, config, share_fn):
    r"""Common installer for the softmax rules that read our ``Softmax``
    node's saved ``(input, output)``: registers ``share_fn(x, s, dim, R_out)``
    as the relevance map, or passes through when the forward did not go
    through our node."""
    saved = getattr(node, 'saved_tensors', None)
    if not saved or len(saved) < 2:
        return install_passthrough(node, config)
    x, s = saved[0], saved[1]
    dim = getattr(node, 'dim', -1)

    def _hook(gi, go, _x=x, _s=s, _dim=dim, _fn=share_fn):
        R_in = _fn(_x, _s, _dim, go[0])
        return as_grad_tuple(gi, {0: R_in})

    return node.register_hook(_hook)


def install_softmax_jacobian(node, config):
    r"""Softmax Jacobian rule (Achtibat et al. 2024, Prop. 3.1) from our
    ``Softmax`` node's saved input and output; passthrough when the
    forward did not go through it.
    """
    return _install_softmax_saved(
        node, config,
        lambda x, s, dim, R: lrp_utils.softmax_jacobian(x, s, R, dim))

def install_softmax_detach(node, config):
    r"""Softmax output treated as a constant gate: ``R_in = s * R_out``.
    Passthrough when the forward did not go through our ``Softmax``.
    """
    return _install_softmax_saved(
        node, config,
        lambda x, s, dim, R: lrp_utils.softmax_detach(s, R))


# ---------------------------------------------------------------------------
# LayerNorm variants (chosen by config.layernorm)
# ---------------------------------------------------------------------------

def _install_layernorm_saved(node, config, ln_fn):
    r"""Common installer for the LayerNorm rules that need only the node's
    saved input, normalized shape, weight and bias: registers
    ``ln_fn(x, ns, w, b, R_out, eps)`` as the relevance map, or passes
    through when the saved state is missing."""
    x = getattr(node, '_saved_input', None)
    ns = getattr(node, '_saved_normalized_shape', None)
    w = getattr(node, '_saved_weight', None)
    b = getattr(node, '_saved_bias', None)
    if x is None or ns is None:
        return install_passthrough(node, config)

    eps = config.eps

    def _hook(gi, go, _x=x, _ns=ns, _w=w, _b=b, _eps=eps, _fn=ln_fn):
        R_in = _fn(_x, _ns, _w, _b, go[0], _eps)
        return as_grad_tuple(gi, {0: R_in})

    return node.register_hook(_hook)


def install_layernorm_yx(node, config):
    r"""LayerNorm ``y/x`` rule (Ali et al. 2022) from the node's saved input,
    weight and bias; passthrough when they are missing.
    """
    return _install_layernorm_saved(node, config, lrp_utils.layernorm_yx)

def install_layernorm_identity(node, config):
    r"""Fused LayerNorm, the default: the bias takes its share, everything
    else passes relevance through; equals the decomposed LayerNorm.
    """
    x = getattr(node, '_saved_input', None)
    ns = getattr(node, '_saved_normalized_shape', None)
    w = getattr(node, '_saved_weight', None)
    b = getattr(node, '_saved_bias', None)
    if x is None or ns is None:
        return install_passthrough(node, config)
    mean = getattr(node, '_saved_result1', None)
    rstd = getattr(node, '_saved_result2', None)
    eps = config.eps

    def _hook(gi, go, _x=x, _ns=ns, _w=w, _b=b, _eps=eps,
              _m=mean, _r=rstd):
        R_in = lrp_utils.layernorm_identity(_x, _ns, _w, _b, go[0], _eps,
                                            mean=_m, rstd=_r)
        return as_grad_tuple(gi, {0: R_in})

    return node.register_hook(_hook)


def install_layernorm_detach_std(node, config):
    r"""LayerNorm with the standard deviation held constant (Achtibat et al.
    2024, Eq. 9, as in LXT); passthrough when the saved state is missing.
    """
    return _install_layernorm_saved(node, config, lrp_utils.layernorm_detach_std)


# ---------------------------------------------------------------------------
# Activation y/x (chosen by config.activation='yx')
# ---------------------------------------------------------------------------

# Forward activation function for a grad_fn, keyed by grad_fn name; the
# lookup is an exact match on the name or on the name with trailing digits
# stripped. Used when ``_saved_result`` is unavailable and ``y = f(x)``
# must be recomputed.
_ACTIVATION_FORWARD = {
    'ReluBackward':        F.relu,
    'LeakyReluBackward':   F.leaky_relu,
    'GeluBackward':        F.gelu,
    'SiluBackward':        F.silu,
    'TanhBackward':        torch.tanh,
    'SigmoidBackward':     torch.sigmoid,
    'HardtanhBackward':    F.hardtanh,
    'HardswishBackward':   F.hardswish,
    'HardsigmoidBackward': F.hardsigmoid,
    'EluBackward':         F.elu,
    'SeluBackward':        F.selu,
    'CeluBackward':        F.celu,
    'SoftplusBackward':    F.softplus,
    'SoftsignBackward':    F.softsign,
    'LogSigmoidBackward':  F.logsigmoid,
    'MishBackward':        F.mish,
}


def install_activation_yx(node, config):
    r"""Activation ``y/x`` rule (Achtibat et al. 2024, Prop. 3.2) from the
    saved output, recomputed from ``_ACTIVATION_FORWARD`` when the node did
    not save it; passthrough otherwise.
    """
    name = node.name()
    y = getattr(node, '_saved_result', None)
    x = getattr(node, '_saved_self', None)

    # Need x; if only y is saved, recovering x is generally not possible.
    if x is None:
        return install_passthrough(node, config)

    if y is None:
        fwd = None
        for key, fn in _ACTIVATION_FORWARD.items():
            if (key == name or key == name.rstrip('0123456789')):
                fwd = fn
                break
        if fwd is None:
            return install_passthrough(node, config)
        with torch.no_grad():
            y = fwd(x)

    eps = config.eps

    def _hook(gi, go, _x=x, _y=y, _eps=eps):
        R_out = go[0]
        R_in = lrp_utils.activation_yx(_x, _y, R_out, _eps)
        return as_grad_tuple(gi, {0: R_in})

    return node.register_hook(_hook)


# ---- passthrough and shape routing -----------------------------------

def install_passthrough(node, config):
    r"""``R_in = R_out`` into slot 0: activations, norms, softmax under the
    default field, sign-preserving unaries. ``None`` slots stay ``None``.
    """
    _trace(None, 'passthrough')

    def _hook(gi, go):
        return as_grad_tuple(gi, {0: go[0]})
    return node.register_hook(_hook)


def install_noop(node, config):
    r"""No-op installer: autograd-native VJP correctly routes
    :math:`R` for shape-routing ops (Reshape, View, Permute, ...).
    """
    pass


# ---- fused attention (consults the BmmBackward rule entry) ----

class _AsBmm:
    r"""Stand-in for one of the two products a fused attention node
    performs. Answers ``name()`` as a bmm so a ``'BmmBackward'`` entry
    reaches it. For ``A @ V`` it carries ``weights_operand: 0`` (the
    weights are on the left by construction); for ``Q @ K^T`` it carries
    no such fact, exactly like the score bmm of the decomposed graph.
    The real node's own facts are copied underneath either way."""

    __slots__ = ('metadata',)

    def __init__(self, real, weights_on_left=False):
        facts = dict(node_facts(real))
        if weights_on_left:
            facts['weights_operand'] = 0
        self.metadata = {'lrp': facts}

    def name(self):
        return 'BmmBackward0'


def _fused_product(config, node, live_left, live_right, weights_on_left):
    r"""The relevance function for one product ``L @ Rt`` inside a fused
    attention node: ``(L, Rt, R_out) -> (R_L, R_Rt)``, or ``None`` when
    neither operand comes from the input. Same family choice as
    :func:`_install_product`."""
    eps = config.eps
    rf = config.relevance_filter
    T = lambda t: t.transpose(-2, -1)
    fwd = lambda x, y: torch.matmul(x, y)
    bwd_a = lambda o, s: torch.matmul(s, T(o))
    bwd_b = lambda o, s: torch.matmul(T(o), s)
    if live_left and live_right:
        fn, kw = resolve_rule(config.rule, _AsBmm(node, weights_on_left), BMM_RULES)
        return lambda L, Rt, R: fn(L, Rt, R, eps, fwd, bwd_a, bwd_b, **kw)
    if not (live_left or live_right):
        return None
    rule_fn, kw = resolve_rule(config.rule, _Named(node, 'MmBackward0'), LINEAR_RULES)
    if live_left:
        def run(L, Rt, R):
            R_L, _ = run_linear_rule(L, Rt, None, R, rule_fn, kw, fwd, bwd_a, bwd_b, eps, relevance_filter=rf)
            return R_L, torch.zeros_like(Rt)
    else:
        fwd_r = lambda x, w: torch.matmul(w, x)
        bwd_ar = lambda w, s: torch.matmul(T(w), s)
        bwd_br = lambda x, s: torch.matmul(s, T(x))

        def run(L, Rt, R):
            R_Rt, _ = run_linear_rule(Rt, L, None, R, rule_fn, kw, fwd_r, bwd_ar, bwd_br, eps, relevance_filter=rf)
            return torch.zeros_like(L), R_Rt
    return run


def install_sdpa(node, config):
    r"""Fused scaled-dot-product attention node, used when
    :func:`autoLRP.set_decompose_attention` is off. Reconstructs the
    attention matrix from the saved query, key, value and logsumexp, then
    runs the same rules the decomposed graph would: each of the two
    products resolved by which operands come from the input
    (:func:`_fused_product`), the softmax addressed as ``SoftmaxBackward``
    through :class:`_Named`.
    Writes the query, key and value slots; GQA expands K and V and sums
    the relevance back over the repeated heads.
    """
    q = getattr(node, '_saved_query', None)
    k = getattr(node, '_saved_key', None)
    v = getattr(node, '_saved_value', None)
    lse = getattr(node, '_saved_logsumexp', None)
    if lse is None:
        lse = getattr(node, '_saved_log_sumexp', None)
    if q is None or k is None or v is None or lse is None:
        _warn_missing_state(node, 'this SDPA backend did not save query/key/value/logsumexp; use autoLRP.set_decompose_attention(True) for full rule coverage')
        return
    mask = getattr(node, '_saved_attn_mask', None)
    if mask is None:
        mask = getattr(node, '_saved_attn_bias', None)
    is_causal = bool(getattr(node, '_saved_is_causal', False))
    scale = getattr(node, '_saved_scale', None)
    # One node stands for softmax and two products; each product is
    # resolved on its own, as the decomposed graph would.
    ps = parents(node, skip_aliases=False)
    live_q, live_k, live_v = (i < len(ps) and reaches_input(ps[i]) for i in range(3))
    live_a = live_q or live_k
    _av = _fused_product(config, node, live_a, live_v, weights_on_left=True)
    _qk = _fused_product(config, node, live_q, live_k, weights_on_left=False)
    if _av is None and _qk is None:
        return
    softmax_name, _ = resolve(config.softmax, _Named(node, 'SoftmaxBackward'))
    _trace(None, f'softmax={softmax_name}')

    def _hook(gi, go, _q=q, _k=k, _v=v, _lse=lse, _mask=mask,
              _is_causal=is_causal, _scale=scale,
              _av=_av, _qk=_qk, _softmax=softmax_name):
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
            if _softmax == 'jacobian':
                R_scores = lrp_utils.softmax_jacobian(
                    fl(scores), fl(A), R_A, dim=-1)
            elif _softmax == 'detach':
                R_scores = lrp_utils.softmax_detach(fl(A), R_A)
            else:                                                 # passthrough
                R_scores = R_A
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


# ---------------------------------------------------------------------------
# Handler tables for the unary config fields
# ---------------------------------------------------------------------------

SOFTMAX_HANDLERS: dict = {
    'passthrough': install_passthrough,
    'jacobian':    install_softmax_jacobian,
    'detach':      install_softmax_detach,
}


LAYERNORM_HANDLERS: dict = {
    'identity':    install_layernorm_identity,
    'passthrough': install_passthrough,
    'yx':          install_layernorm_yx,
    'detach_std':  install_layernorm_detach_std,
}


ACTIVATION_HANDLERS: dict = {
    'passthrough': install_passthrough,
    'yx':          install_activation_yx,
}
