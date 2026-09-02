r"""``autograd.Function`` wrappers for ops whose native node saves less
than a rule needs: softmax saves only its output, add and sub save
nothing, mean, sum and cumsum do not save the input. Each wrapper's
backward equals the native gradient (bitwise, softmax within one
ulp), and each node is named after the op it replaces
(``AddBackward``), next to the native ``AddBackward0``. The wrappers
run only when an :class:`autolrp.LRPTensor` is in the call.
"""
import torch
from torch._C import DisableTorchFunctionSubclass

from ..backward.lrp_utils import reduce_to_shape


class Softmax(torch.autograd.Function):
    r"""Softmax that saves ``(x, y)``; node ``SoftmaxBackward``. The result
    is cast back to the input's subclass so downstream ops keep being
    intercepted.
    """

    @staticmethod
    def forward(ctx, x, dim):
        with DisableTorchFunctionSubclass():
            y = torch.softmax(x, dim=dim)
        ctx.save_for_backward(x, y)
        ctx.dim = dim
        if type(x) is not torch.Tensor:
            y = y.as_subclass(type(x))
        return y

    @staticmethod
    def backward(ctx, grad_out):
        _, y = ctx.saved_tensors
        return y * (grad_out - (grad_out * y).sum(dim=ctx.dim, keepdim=True)), None


class Add(torch.autograd.Function):
    r"""Add that saves ``(a, b)``; node ``AddBackward``. Entered for two
    tensors, with ``alpha`` already folded into ``b`` by the intercept.
    """

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        with DisableTorchFunctionSubclass():
            r = a + b
        for src in (a, b):
            if type(src) is not torch.Tensor:
                r = r.as_subclass(type(src))
                break
        return r

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        return reduce_to_shape(grad_out, a.shape), reduce_to_shape(grad_out, b.shape)


class Sub(torch.autograd.Function):
    r"""Sub that saves ``(a, b)``; node ``SubBackward``. The installer
    overwrites both slots with the split shares.
    """

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        with DisableTorchFunctionSubclass():
            r = a - b
        for src in (a, b):
            if type(src) is not torch.Tensor:
                r = r.as_subclass(type(src))
                break
        return r

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        return (reduce_to_shape(grad_out, a.shape),
                -reduce_to_shape(grad_out, b.shape))


class Mean(torch.autograd.Function):
    r"""Mean that saves ``(x,)`` for the ``|x|``-proportional split; node
    ``MeanBackward``.
    """

    @staticmethod
    def forward(ctx, x, dim, keepdim):
        ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.keepdim = keepdim
        with DisableTorchFunctionSubclass():
            if dim is None:
                r = x.mean(dim=None, keepdim=True) if keepdim else x.mean()
            else:
                r = x.mean(dim=dim, keepdim=keepdim)
        if type(x) is not torch.Tensor:
            r = r.as_subclass(type(x))
        return r

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        if ctx.dim is None:
            return grad_out.expand_as(x) / x.numel(), None, None
        if not ctx.keepdim:
            grad_out = grad_out.unsqueeze(ctx.dim) if isinstance(ctx.dim, int) \
                       else _unsqueeze_dims(grad_out, ctx.dim)
        if isinstance(ctx.dim, int):
            n = x.shape[ctx.dim]
        else:
            n = 1
            for d in ctx.dim:
                n *= x.shape[d]
        return grad_out.expand_as(x) / n, None, None


class Sum(torch.autograd.Function):
    r"""Sum that saves ``(x,)`` for the ``|x|``-proportional split; node
    ``SumBackward``.
    """

    @staticmethod
    def forward(ctx, x, dim, keepdim):
        ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.keepdim = keepdim
        with DisableTorchFunctionSubclass():
            if dim is None:
                r = x.sum(dim=None, keepdim=True) if keepdim else x.sum()
            else:
                r = x.sum(dim=dim, keepdim=keepdim)
        if type(x) is not torch.Tensor:
            r = r.as_subclass(type(x))
        return r

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        if ctx.dim is None:
            return grad_out.expand_as(x), None, None
        if not ctx.keepdim:
            grad_out = grad_out.unsqueeze(ctx.dim) if isinstance(ctx.dim, int) \
                       else _unsqueeze_dims(grad_out, ctx.dim)
        return grad_out.expand_as(x), None, None


def _unsqueeze_dims(t: torch.Tensor, dims) -> torch.Tensor:
    r"""Re-insert reduced dims into ``t``. Dims are normalized against the
    restored rank first so a mixed-sign tuple like ``(-3, 2)`` sorts by
    actual position, not by sign."""
    nd = t.dim() + len(dims)
    for d in sorted(x % nd for x in dims):
        t = t.unsqueeze(d)
    return t


class Cumsum(torch.autograd.Function):
    r"""Cumsum that saves ``x`` for the epsilon rule; node
    ``CumsumBackward``.
    """

    @staticmethod
    def forward(ctx, x, dim):
        ctx.save_for_backward(x)
        ctx.dim = dim
        with DisableTorchFunctionSubclass():
            r = x.cumsum(dim)
        if type(x) is not torch.Tensor:
            r = r.as_subclass(type(x))
        return r

    @staticmethod
    def backward(ctx, grad_out):
        d = ctx.dim
        return torch.flip(torch.cumsum(torch.flip(grad_out, (d,)), d), (d,)), None
