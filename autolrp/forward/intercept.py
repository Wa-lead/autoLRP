r"""``__torch_function__`` dispatch for :class:`autolrp.LRPTensor`.

``REWRITES`` maps a torch function name to a rewrite
``fn(func, args, kwargs) -> result | NotImplemented``; PyTorch gives
every calling form one name (``a + b`` and ``torch.add`` both arrive
as ``'add'``), and ``NotImplemented`` falls through to the native op.
Built in: the fused-attention decomposition (toggle
:func:`set_decompose_attention`), the state-saving wrappers of
:mod:`autolrp.forward.ops`, in-place to out-of-place remaps, and the
live copy of weight operands (:data:`WEIGHT_BEARING`). Register your
own with :func:`register_rewrite`; a registered name replaces the
built-in. The attention and wrapper built-ins are module-level
functions you can delegate to; the 22 in-place remaps are factory
closures reachable only through ``REWRITES``.
"""

import contextlib
import math
import os
import warnings
from typing import Callable, Dict

import torch
import torch.nn.functional as F

from .ops import Softmax, Add, Sub, Mean, Sum, Cumsum


# In-place methods rewritten out of place: an in-place node saves the
# mutated state. AUTOLRP_NO_INPLACE_REMAP=1 warns and dispatches natively.
_INPLACE_TO_OOP = {
    'relu_': 'relu', 'sigmoid_': 'sigmoid', 'tanh_': 'tanh',
    'abs_': 'abs', 'neg_': 'neg', 'exp_': 'exp', 'log_': 'log',
    'add_': 'add', 'sub_': 'sub', 'mul_': 'mul', 'div_': 'div',
    'pow_': 'pow', 'clamp_': 'clamp',
    'clamp_min_': 'clamp_min', 'clamp_max_': 'clamp_max',
    'square_': 'square', 'sqrt_': 'sqrt', 'reciprocal_': 'reciprocal',
    'addcmul_': 'addcmul', 'addcdiv_': 'addcdiv',
    'addmm_': 'addmm', 'addbmm_': 'addbmm',
}

_INPLACE_REMAP_DISABLED = (
    os.environ.get('AUTOLRP_NO_INPLACE_REMAP', '0').lower()
    in ('1', 'true', 'yes')
)
_INPLACE_REMAP_WARNED: set = set()


# Decompose fused attention into matmul/softmax/matmul (default) or keep
# the fused node for install_sdpa. Set before the forward pass.
_DECOMPOSE_ATTENTION = (
    os.environ.get('AUTOLRP_NO_DECOMPOSE_ATTENTION', '0').lower()
    not in ('1', 'true', 'yes')
)


def set_decompose_attention(enabled: bool) -> None:
    r"""Enable/disable forward decomposition of fused attention kernels.

    When enabled (default), ``scaled_dot_product_attention`` is expanded into
    matmul/softmax/matmul so the per-primitive LRP rules apply. When disabled,
    the fused op is kept and its backward node is handled by ``install_sdpa``.
    Must be called BEFORE the forward pass.
    """
    global _DECOMPOSE_ATTENTION
    _DECOMPOSE_ATTENTION = bool(enabled)


def get_decompose_attention() -> bool:
    r"""Return whether forward attention decomposition is currently enabled."""
    return _DECOMPOSE_ATTENTION


@contextlib.contextmanager
def decompose_attention(enabled: bool = True):
    r"""Temporarily set the attention-decomposition mode; restore on exit."""
    global _DECOMPOSE_ATTENTION
    prev = _DECOMPOSE_ATTENTION
    _DECOMPOSE_ATTENTION = bool(enabled)
    try:
        yield
    finally:
        _DECOMPOSE_ATTENTION = prev


# ---------------------------------------------------------------------------
# Rewrite registry
# ---------------------------------------------------------------------------

REWRITES: Dict[str, Callable] = {}


def register_rewrite(fname: str):
    r"""Decorator: register ``fn(func, args, kwargs) -> result |
    NotImplemented`` as the rewrite for torch calls named ``fname``.

    ``fname`` is the canonical ``func.__name__`` — PyTorch funnels
    operator, method, and functional forms of the same op to one name.
    Registering an existing name replaces it (last-write-wins).
    """
    def _decorator(fn: Callable) -> Callable:
        REWRITES[fname] = fn
        return fn
    return _decorator


def torch_function_handler(cls, func, types, args, kwargs):
    r"""``__torch_function__`` entry point: flip a functional
    ``inplace=True``, make weight operands live for :data:`WEIGHT_BEARING`
    ops, then run the registered rewrite or dispatch natively.
    """
    kwargs = kwargs or {}
    fname = getattr(func, '__name__', '')

    # Functional inplace=True kwarg (e.g. F.relu(x, inplace=True)) →
    # flip to inplace=False: the input tensor is NOT mutated, so a
    # caller relying on the mutation must use the return value.
    # Warned once per op, like the in-place method remaps.
    if kwargs.get('inplace'):
        _warn_key = f'{fname}(inplace=True)'
        if _warn_key not in _INPLACE_REMAP_WARNED:
            _INPLACE_REMAP_WARNED.add(_warn_key)
            warnings.warn(
                f"Functional '{fname}' with inplace=True remapped to "
                f"inplace=False for LRP graph compatibility; the input "
                f"tensor is NOT modified in place.",
                UserWarning, stacklevel=3)
        kwargs = dict(kwargs)
        kwargs['inplace'] = False

    if fname in WEIGHT_BEARING:
        args, kwargs = _make_weights_live(args, kwargs)

    rewrite = REWRITES.get(fname)
    if rewrite is not None:
        out = rewrite(func, args, kwargs)
        if out is not NotImplemented:
            return out

    return super(cls, cls).__torch_function__(func, types, args, kwargs)


# These ops save an operand only when the other one needs a gradient,
# so a frozen weight would leave the input unsaved. Every grad-free
# tensor operand becomes a detached copy that requires grad; the
# model's parameter is untouched, the copy's gradient is discarded.
WEIGHT_BEARING = frozenset({
    'linear', 'matmul', 'mm', 'bmm', 'addmm', 'baddbmm', 'addbmm',
    'einsum', 'tensordot',
    'conv1d', 'conv2d', 'conv3d',
    'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d',
})


def _live(t):
    if isinstance(t, torch.Tensor) and not t.requires_grad:
        return t.detach().requires_grad_(True)
    if isinstance(t, (list, tuple)) and any(isinstance(u, torch.Tensor) for u in t):
        return type(t)(_live(u) for u in t)
    return t


def _make_weights_live(args, kwargs):
    return tuple(_live(a) for a in args), {k: _live(v) for k, v in kwargs.items()}


# ---------------------------------------------------------------------------
# Built-in rewrites: fused-attention decomposition
# ---------------------------------------------------------------------------

@register_rewrite('scaled_dot_product_attention')
def decompose_sdpa(func, args, kwargs):
    r"""Rewrite fused SDPA into matmul/softmax/matmul (see
    :func:`_decompose_sdpa`). Declines when decomposition is toggled off,
    letting the fused node build for ``install_sdpa``."""
    if not _DECOMPOSE_ATTENTION:
        return NotImplemented
    return _decompose_sdpa(args, kwargs)


@register_rewrite('multi_head_attention_forward')
def decompose_mha(func, args, kwargs):
    r"""Rewrite fused multi-head attention into explicit projections,
    BMMs, softmax (see :func:`_decompose_mha_forward`)."""
    if not _DECOMPOSE_ATTENTION:
        return NotImplemented
    return _decompose_mha_forward(args, kwargs)


# ---------------------------------------------------------------------------
# Built-in rewrites: state-saving wrappers
# ---------------------------------------------------------------------------

@register_rewrite('softmax')
def wrap_softmax(func, args, kwargs):
    r"""Route softmax through :class:`Softmax` so both ``x`` and ``y``
    are saved (native saves only ``y``; the Jacobian rule needs both)."""
    if len(args) < 1 or not isinstance(args[0], torch.Tensor):
        return NotImplemented
    dim = args[1] if len(args) > 1 else kwargs.get('dim', -1)
    if dim is None:
        return NotImplemented
    dtype = kwargs.get('dtype', args[2] if len(args) > 2 else None)
    x = args[0]
    if dtype is not None:
        x = x.to(dtype)      # native semantics: compute AND return in dtype
    return Softmax.apply(x, dim)


def _two_tensors(args, kwargs):
    """The two tensor operands of an add or sub, with ``alpha`` folded
    into the second one (``torch.add(a, b, alpha=c)`` is ``a + c*b``).
    The fold is a multiplication by a constant, through which relevance
    passes unchanged. ``None`` when an operand is not a tensor."""
    if len(args) < 2:
        return None
    a, b = args[0], args[1]
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return None
    alpha = kwargs.get('alpha', 1)
    if alpha != 1:
        b = b * alpha
    return a, b


@register_rewrite('add')
def wrap_add(func, args, kwargs):
    r"""Route two-tensor add through :class:`Add` (saves operands for
    the proportional/residual split). A scalar operand declines to
    native, where the constant acts as a bias."""
    pair = _two_tensors(args, kwargs)
    if pair is None:
        return NotImplemented
    return Add.apply(*pair)


@register_rewrite('sub')
def wrap_sub(func, args, kwargs):
    r"""Route two-tensor sub through :class:`Sub`."""
    pair = _two_tensors(args, kwargs)
    if pair is None:
        return NotImplemented
    return Sub.apply(*pair)


def _wrap_reduction(fn_cls, args, kwargs):
    r"""Shared parsing for the mean/sum wrappers: cast the input for a
    ``dtype`` argument, pull ``dim``/``keepdim``, apply ``fn_cls``."""
    if len(args) < 1 or not isinstance(args[0], torch.Tensor):
        return NotImplemented
    a = args[0]
    if kwargs.get('dtype') is not None:
        a = a.to(kwargs['dtype'])       # the cast is a ToCopy node, noop
    dim = args[1] if len(args) >= 2 else kwargs.get('dim', None)
    keepdim = args[2] if len(args) >= 3 else kwargs.get('keepdim', False)
    return fn_cls.apply(a, dim, keepdim)


@register_rewrite('mean')
def wrap_mean(func, args, kwargs):
    r"""Route mean through :class:`Mean` (saves the pre-reduction
    activation for the |x|-proportional split). A ``dtype`` argument
    becomes a cast of the input first."""
    return _wrap_reduction(Mean, args, kwargs)


@register_rewrite('sum')
def wrap_sum(func, args, kwargs):
    r"""Route sum through :class:`Sum`."""
    return _wrap_reduction(Sum, args, kwargs)


@register_rewrite('cumsum')
def wrap_cumsum(func, args, kwargs):
    r"""Route cumsum through :class:`Cumsum` so the input is saved.
    A ``dtype`` argument becomes a cast of the input first (native
    casts the input before accumulating)."""
    if len(args) < 1 or not isinstance(args[0], torch.Tensor):
        return NotImplemented
    a = args[0]
    if kwargs.get('dtype') is not None:
        a = a.to(kwargs['dtype'])       # the cast is a ToCopy node, noop
    dim = args[1] if len(args) >= 2 else kwargs.get('dim', None)
    if dim is None:
        return NotImplemented
    return Cumsum.apply(a, dim)


# ---------------------------------------------------------------------------
# Built-in rewrites: in-place → out-of-place remaps
# ---------------------------------------------------------------------------

def _make_inplace_rewrite(ip_name: str, oop_name: str):
    def _rewrite(func, args, kwargs):
        if _INPLACE_REMAP_DISABLED:
            warnings.warn(
                f"In-place '{ip_name}' on an LRPTensor produces a graph "
                f"node autoLRP cannot trace; relevance at this layer "
                f"will be zero or incorrect. (Remap disabled via "
                f"AUTOLRP_NO_INPLACE_REMAP — unset to enable.)",
                UserWarning, stacklevel=4)
            return NotImplemented
        if ip_name not in _INPLACE_REMAP_WARNED:
            _INPLACE_REMAP_WARNED.add(ip_name)
            warnings.warn(
                f"In-place '{ip_name}' remapped to out-of-place "
                f"'{oop_name}' for LRP graph compatibility; the "
                f"original tensor is NOT modified in place. Set "
                f"AUTOLRP_NO_INPLACE_REMAP=1 to disable.",
                UserWarning, stacklevel=4)
        # Re-enters __torch_function__ under the out-of-place name, so
        # wrapper rewrites ('add', 'sub', ...) still apply downstream.
        return getattr(torch.Tensor, oop_name)(*args, **kwargs)
    _rewrite.__name__ = f'remap_{ip_name}'
    return _rewrite


for _ip, _oop in _INPLACE_TO_OOP.items():
    REWRITES[_ip] = _make_inplace_rewrite(_ip, _oop)


# -----------------------------------------------------------------------
# Attention decomposition
# -----------------------------------------------------------------------

# A finite constant: -inf times 0 is NaN in the gradient.
_MASK_FILL = -1.0e9


def _add_causal_and_mask(scores, is_causal, attn_mask, bool_true_attends):
    r"""Add the causal triangle and/or the attention mask to ``scores``
    as large negative constants (an add is a node our rules handle,
    unlike ``masked_fill``, and ``softmax(-1e9)`` is 0 in fp32/fp64).

    ``bool_true_attends`` sets the polarity of a bool mask: SDPA's
    ``True`` means *may attend* (fill the ``False`` positions), MHA's
    ``True`` means *do NOT attend* (fill the ``True`` positions). A
    float mask is already additive and is added unchanged (it may
    contain ``-inf``; callers that relied on that get the same softmax
    behavior).
    """
    if is_causal:
        L, S = scores.shape[-2], scores.shape[-1]
        scores = scores + torch.triu(
            torch.full((L, S), _MASK_FILL,
                       device=scores.device, dtype=scores.dtype),
            diagonal=1)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            masked_out = ~attn_mask if bool_true_attends else attn_mask
            attn_mask = masked_out.to(scores.dtype) * _MASK_FILL
        scores = scores + attn_mask
    return scores


def _decompose_sdpa(args, kwargs):
    r"""``scaled_dot_product_attention`` as matmul, scale, mask, softmax,
    matmul. Masks are added as large negative constants rather than
    ``masked_fill`` with ``-inf``, since ``softmax(-1e9)`` is 0 in fp32 and
    fp64 and an add is a node our rules handle. Dropout is applied
    whenever ``dropout_p > 0``, matching the native kernel.
    """
    query, key, value = args[0], args[1], args[2]
    attn_mask = kwargs.get('attn_mask', args[3] if len(args) > 3 else None)
    dropout_p = kwargs.get('dropout_p', args[4] if len(args) > 4 else 0.0)
    is_causal = kwargs.get('is_causal', args[5] if len(args) > 5 else False)
    scale = kwargs.get('scale', None)

    # GQA: repeat K/V heads to match Q; shapes are (B, H, T, D).
    h_q = query.shape[-3]
    h_kv = key.shape[-3]
    if h_q != h_kv:
        if h_q % h_kv != 0:
            raise ValueError(
                f"GQA: Q heads ({h_q}) not a multiple of K/V heads ({h_kv})")
        n_rep = h_q // h_kv
        key = key.repeat_interleave(n_rep, dim=-3)
        value = value.repeat_interleave(n_rep, dim=-3)

    d = query.shape[-1]
    if scale is None:
        scale = d ** -0.5

    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Native SDPA bool semantics: True = may attend.
    scores = _add_causal_and_mask(scores, is_causal, attn_mask,
                                  bool_true_attends=True)

    attn = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    return torch.matmul(attn, value)


def _decompose_mha_forward(args, kwargs):
    r"""Replace :func:`torch.nn.functional.multi_head_attention_forward`
    with an explicit-op decomposition.

    The fused C++ kernel is unrolled into its constituent linear
    projections, BMMs, softmax, and dropout so the LRP installers can
    attach hooks to each step.
    """
    query, key, value = args[0], args[1], args[2]
    embed_dim, num_heads = args[3], args[4]
    in_proj_weight, in_proj_bias = args[5], args[6]
    bias_k, bias_v, add_zero_attn = args[7], args[8], args[9]
    dropout_p = args[10]
    out_proj_weight, out_proj_bias = args[11], args[12]
    training = kwargs.get('training', args[13] if len(args) > 13 else True)
    key_padding_mask = kwargs.get('key_padding_mask', args[14] if len(args) > 14 else None)
    need_weights = kwargs.get('need_weights', args[15] if len(args) > 15 else True)
    attn_mask = kwargs.get('attn_mask', args[16] if len(args) > 16 else None)
    use_separate_proj_weight = kwargs.get('use_separate_proj_weight',
                                          args[17] if len(args) > 17 else False)
    q_proj_weight = kwargs.get('q_proj_weight', args[18] if len(args) > 18 else None)
    k_proj_weight = kwargs.get('k_proj_weight', args[19] if len(args) > 19 else None)
    v_proj_weight = kwargs.get('v_proj_weight', args[20] if len(args) > 20 else None)
    static_k = kwargs.get('static_k', args[21] if len(args) > 21 else None)
    static_v = kwargs.get('static_v', args[22] if len(args) > 22 else None)
    average_attn_weights = kwargs.get('average_attn_weights',
                                      args[23] if len(args) > 23 else True)
    is_causal = kwargs.get('is_causal', False)

    if bias_k is not None or bias_v is not None:
        raise NotImplementedError(
            "autoLRP attention decomposition does not support "
            "bias_k/bias_v (MultiheadAttention add_bias_kv=True)")
    if add_zero_attn:
        raise NotImplementedError(
            "autoLRP attention decomposition does not support "
            "add_zero_attn=True")
    if static_k is not None or static_v is not None:
        raise NotImplementedError(
            "autoLRP attention decomposition does not support "
            "static_k/static_v")

    is_batched = query.dim() == 3
    if not is_batched:
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

    T, B, E = query.shape
    H = num_heads
    D = E // H

    # QKV projection
    if not use_separate_proj_weight and in_proj_weight is not None:
        if query is key and key is value:
            qkv = F.linear(query, in_proj_weight, in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            w_q, w_k, w_v = in_proj_weight.chunk(3, dim=0)
            b_q, b_k, b_v = (in_proj_bias.chunk(3, dim=0)
                              if in_proj_bias is not None else (None, None, None))
            q = F.linear(query, w_q, b_q)
            k = F.linear(key, w_k, b_k)
            v = F.linear(value, w_v, b_v)
    else:
        b_q, b_k, b_v = (in_proj_bias.chunk(3, dim=0)
                          if in_proj_bias is not None else (None, None, None))
        q = F.linear(query, q_proj_weight, b_q)
        k = F.linear(key, k_proj_weight, b_k)
        v = F.linear(value, v_proj_weight, b_v)

    # Reshape to multi-head: (T, B, E) → (B*H, T, D)
    q = q.view(T, B * H, D).transpose(0, 1)
    k = k.view(k.shape[0], B * H, D).transpose(0, 1)
    v = v.view(v.shape[0], B * H, D).transpose(0, 1)

    # Masks are added rather than masked_fill'd so the backward node is
    # AddBackward (hookable) instead of MaskedFillBackward (not hookable).
    scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(D)

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() != 3:
            raise ValueError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported")
    # MHA bool semantics: True = do NOT attend.
    scores = _add_causal_and_mask(scores, is_causal, attn_mask,
                                  bool_true_attends=False)

    if key_padding_mask is not None:
        # key_padding_mask: True = mask-out. Same convention as torch's MHA.
        kpm_bool = key_padding_mask if key_padding_mask.dtype == torch.bool \
                   else key_padding_mask.bool()
        kpm = kpm_bool.unsqueeze(1).expand(-1, H, -1).reshape(B * H, 1, -1)
        scores = scores + kpm.to(scores.dtype) * _MASK_FILL

    attn_weights = torch.softmax(scores, dim=-1)
    if training and dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    out = torch.bmm(attn_weights, v)
    out = out.transpose(0, 1).reshape(T, B, E)
    out = F.linear(out, out_proj_weight, out_proj_bias)

    if not is_batched:
        out = out.squeeze(1)

    if need_weights:
        weights = attn_weights.view(B, H, T, -1)
        if average_attn_weights:
            weights = weights.mean(dim=1)
        return out, weights
    return out, None


