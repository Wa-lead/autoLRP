r"""``autograd.Function`` wrappers for ops whose native node saves less
than a rule needs: softmax saves only its output, add and sub save
nothing, mean, sum and cumsum do not save the input. Each wrapper's
backward equals the native gradient (bitwise, softmax within one
ulp), and each node is named after the op it replaces
(``AddBackward``), next to the native ``AddBackward0``. The wrappers
run only when an :class:`autolrp.LRPTensor` is in the call. Each is
exposed as a function with the signature of the torch op it replaces
(:func:`softmax`, :func:`add`, ...), so the intercept passes the call's
own arguments through; a form the wrapper does not save for (an add
with a constant, softmax without a dim) runs the native op.
"""
import torch
import torch.nn.functional as F
from torch._C import DisableTorchFunctionSubclass

from ..backward.lrp_utils import reduce_to_shape
from ..nodes import ELEMENTWISE, node_of


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
    tensors, with ``alpha`` folded into ``b`` by :func:`add`.
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
    overwrites both positions with the split shares.
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


def _native(fn, *args, **kwargs):
    """The native op, not intercepted, its result kept in the input's
    subclass: what a wrapper returns for a form it does not save for."""
    with DisableTorchFunctionSubclass():
        out = fn(*args, **kwargs)
    for t in args:
        if isinstance(t, torch.Tensor) and type(t) is not torch.Tensor and isinstance(out, torch.Tensor):
            return out.as_subclass(type(t))
    return out


# ---------------------------------------------------------------------------
# The wrapped ops, each with the signature of the torch op it replaces.
# The intercept calls them with the torch call's own arguments.
# ---------------------------------------------------------------------------

def softmax(input, dim=None, _stacklevel=3, dtype=None):
    # torch.softmax(input, dim, dtype) and F.softmax(input, dim, _stacklevel, dtype) both arrive here
    if isinstance(_stacklevel, torch.dtype):             # torch.softmax's third positional is the dtype
        dtype, _stacklevel = _stacklevel, 3
    if dim is None:                                      # the deprecated implicit-dim form
        return _native(torch.softmax, input, dim=dim, dtype=dtype)
    if dtype is not None:
        input = input.to(dtype)
    return Softmax.apply(input, dim)


def add(input, other, *, alpha=1):
    if not (isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor)):
        return _native(torch.add, input, other, alpha=alpha)   # a constant: a bias, the native node
    return Add.apply(input, other if alpha == 1 else other * alpha)


def sub(input, other, *, alpha=1):
    if not (isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor)):
        return _native(torch.sub, input, other, alpha=alpha)
    return Sub.apply(input, other if alpha == 1 else other * alpha)


def mean(input, dim=None, keepdim=False, *, dtype=None):
    if dtype is not None:                                # native casts before reducing
        input = input.to(dtype)
    return Mean.apply(input, dim, keepdim)


def sum(input, dim=None, keepdim=False, *, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    return Sum.apply(input, dim, keepdim)


def cumsum(input, dim, *, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    return Cumsum.apply(input, dim)


def elementwise_function(class_name):
    r"""An ``autograd.Function`` for one elementwise nonlinearity, node
    ``<class_name>Backward``. Saves ``(x, y)`` so the ``y/x`` rule has
    both, whatever the native node would have kept. The backward has no
    formula: it re-runs the op on the saved input under autograd and
    takes the vector-Jacobian product, which is the native gradient for
    any parameters. ``create_graph=True`` keeps the gradient differentiable
    for second-order callers; autograd drops that graph on its own when
    the outer backward is an ordinary one."""

    def forward(ctx, x, op, args, kwargs):
        with DisableTorchFunctionSubclass():          # our own call must not be intercepted again
            y = op(x, *args, **kwargs)
        ctx.save_for_backward(x, y)
        ctx.op, ctx.args, ctx.kwargs = op, args, kwargs
        if type(x) is not torch.Tensor:               # keep the LRPTensor type on the output
            y = y.as_subclass(type(x))
        return y

    def backward(ctx, grad_y):
        x, _ = ctx.saved_tensors
        with torch.enable_grad(), DisableTorchFunctionSubclass():   # grad mode is off inside a backward
            y = ctx.op(x, *ctx.args, **ctx.kwargs)
            grad_x, = torch.autograd.grad(y, x, grad_y, create_graph=True)
        return grad_x, None, None, None               # one entry per forward argument

    return type(class_name, (torch.autograd.Function,),
                {'forward': staticmethod(forward), 'backward': staticmethod(backward)})


def _torch_op(fname):
    """The torch callable behind a handler name."""
    if fname == 'log_sigmoid':
        return F.logsigmoid
    return getattr(F, fname, None) or getattr(torch, fname)


def elementwise_op(fname):
    """The wrapped form of one elementwise op: same call as the torch op,
    ``input`` first, the op's own parameters after (``negative_slope``,
    ``alpha``, ``min``/``max``, an exponent, ``approximate=``). A tensor
    among the parameters (``pow`` with a tensor exponent) makes the op
    two-operand: the native node."""
    op = _torch_op(fname)
    wrapper = elementwise_function(node_of(fname).removesuffix('Backward'))

    def wrapped(input, *args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != 'inplace'}
        if any(isinstance(v, torch.Tensor) for v in args + tuple(kwargs.values())):
            return _native(op, input, *args, **kwargs)
        return wrapper.apply(input, op, args, kwargs)
    wrapped.__name__ = fname
    return wrapped


# Handler name -> wrapped op. The intercept's table for these ops.
WRAPPED = {'softmax': softmax, 'add': add, 'sub': sub, 'mean': mean, 'sum': sum, 'cumsum': cumsum,
           **{fname: elementwise_op(fname) for fname in ELEMENTWISE}}
