r"""``__torch_function__`` dispatch for :class:`autolrp.LRPTensor`.

Every torch call on an LRPTensor arrives at :func:`torch_function_handler`
under one name per op (``a + b``, ``a.add(b)`` and ``torch.add(a, b)``
all arrive as ``'add'``). The handler makes weight operands live, then
looks the name up in :data:`REWRITES` and calls the rewrite with the
call's own arguments; every rewrite has the signature of the op it
replaces. The built-in rewrites are the state-saving wrapped ops of
:mod:`autolrp.forward.ops`,
write fused ops out as ordinary ops (:mod:`autolrp.forward.decompose`),
and turn in-place methods into their out-of-place forms. Register your
own with :func:`register_rewrite`; a registered name replaces the built-in.
"""
import contextlib
import os
import warnings
from typing import Callable, Dict

import torch
import torch.nn.functional as F

from . import ops
from .decompose import decompose_sdpa, decompose_mha


# ---------------------------------------------------------------------------
# The handler
# ---------------------------------------------------------------------------

REWRITES: Dict[str, Callable] = {}

# Ops that combine the input with a parameter. Autograd saves the input
# only if the parameter needs a gradient, so a frozen parameter would
# leave the input unsaved. Every grad-free tensor operand becomes a
# detached alias that requires grad; the parameter itself is untouched,
# and the backward never computes the alias's gradient.
HAS_PARAMETER = frozenset({
    'linear', 'matmul', 'mm', 'bmm', 'addmm', 'baddbmm', 'addbmm', 'einsum', 'tensordot',
    'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d',
})

_WARNED: set = set()


def _warn_once(key, message):
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, UserWarning, stacklevel=4)


def _live(t):
    if isinstance(t, torch.Tensor) and not t.requires_grad:
        return t.detach().requires_grad_(True)
    if isinstance(t, (list, tuple)) and any(isinstance(u, torch.Tensor) for u in t):
        return type(t)(_live(u) for u in t)
    return t


def torch_function_handler(cls, func, types, args, kwargs):
    r"""``__torch_function__`` entry point. A functional ``inplace=True``
    becomes ``inplace=False`` (the input is not mutated; use the return
    value); weight operands of :data:`HAS_PARAMETER` ops are made live;
    then the registered rewrite runs, or the native op."""
    kwargs = dict(kwargs or {})
    fname = getattr(func, '__name__', '')
    if kwargs.get('inplace'):
        _warn_once(f'{fname}(inplace=True)',
                   f"'{fname}(inplace=True)' runs as inplace=False for LRP: the input is not modified in place.")
        kwargs['inplace'] = False
    if fname in HAS_PARAMETER:
        args = tuple(_live(a) for a in args)
        kwargs = {k: _live(v) for k, v in kwargs.items()}
    rewrite = REWRITES.get(fname)
    if rewrite is not None:
        return rewrite(*args, **kwargs)
    return super(cls, cls).__torch_function__(func, types, args, kwargs)


def register_rewrite(fname: str, fn: Callable = None):
    r"""Register ``fn`` for torch calls named ``fname``; it is called with
    the call's own arguments and must accept the torch op's signature.
    ``register_rewrite('gelu', fn)``, or as a decorator
    ``@register_rewrite('gelu')``. An existing name is replaced."""
    def _register(f):
        REWRITES[fname] = f
        return f
    return _register if fn is None else _register(fn)


# ---------------------------------------------------------------------------
# Rewrites. Each has the signature of the op it replaces and is called
# with the call's own arguments; the wrapped ops are ops.WRAPPED.
# ---------------------------------------------------------------------------

def rewrite_sdpa(*args, **kwargs):
    r"""Fused attention as matmul, scale, mask, softmax, matmul, unless
    decomposition is off (then the fused node stays for ``install_sdpa``)."""
    if not _DECOMPOSE_ATTENTION:
        return ops._native(F.scaled_dot_product_attention, *args, **kwargs)
    return decompose_sdpa(*args, **kwargs)


def rewrite_mha(*args, **kwargs):
    r"""``multi_head_attention_forward`` as its projections and attention."""
    if not _DECOMPOSE_ATTENTION:
        return ops._native(F.multi_head_attention_forward, *args, **kwargs)
    return decompose_mha(*args, **kwargs)


def rewrite_inplace(name):
    r"""An in-place method (``x.relu_()``, ``x.add_(y)``) as its
    out-of-place form: an in-place node saves the mutated state. The
    input is not modified; use the return value. Re-enters the handler
    under the new name, so the wrapped op applies.
    ``AUTOLRP_NO_INPLACE_REMAP=1`` keeps the in-place op and warns."""
    out_of_place = getattr(torch.Tensor, name[:-1])
    in_place = getattr(torch.Tensor, name)

    def rewrite(*args, **kwargs):
        if _NO_INPLACE_REMAP:
            _warn_once(name, f"In-place '{name}' kept: its node cannot be traced; relevance at this layer will be wrong.")
            return ops._native(in_place, *args, **kwargs)
        _warn_once(name, f"In-place '{name}' runs as '{name[:-1]}' for LRP: the input is not modified in place.")
        return out_of_place(*args, **kwargs)
    return rewrite


_INPLACE = ('relu_', 'sigmoid_', 'tanh_', 'abs_', 'neg_', 'exp_', 'log_', 'sqrt_', 'square_',
            'reciprocal_', 'pow_', 'clamp_', 'clamp_min_', 'clamp_max_',
            'add_', 'sub_', 'mul_', 'div_', 'addcmul_', 'addcdiv_', 'addmm_', 'addbmm_')

REWRITES.update({
    **ops.WRAPPED,
    'scaled_dot_product_attention': rewrite_sdpa,
    'multi_head_attention_forward': rewrite_mha,
    **{name: rewrite_inplace(name) for name in _INPLACE},
})


# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------

_NO_INPLACE_REMAP = os.environ.get('AUTOLRP_NO_INPLACE_REMAP', '0').lower() in ('1', 'true', 'yes')
_DECOMPOSE_ATTENTION = os.environ.get('AUTOLRP_NO_DECOMPOSE_ATTENTION', '0').lower() not in ('1', 'true', 'yes')


def set_decompose_attention(enabled: bool) -> None:
    r"""Decompose fused attention during the forward (default) or keep the
    fused node for ``install_sdpa``. Set before the forward pass."""
    global _DECOMPOSE_ATTENTION
    _DECOMPOSE_ATTENTION = bool(enabled)


def get_decompose_attention() -> bool:
    return _DECOMPOSE_ATTENTION


@contextlib.contextmanager
def decompose_attention(enabled: bool = True):
    r"""Temporarily set the attention-decomposition mode."""
    global _DECOMPOSE_ATTENTION
    prev = _DECOMPOSE_ATTENTION
    _DECOMPOSE_ATTENTION = bool(enabled)
    try:
        yield
    finally:
        _DECOMPOSE_ATTENTION = prev
