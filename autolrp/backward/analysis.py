r"""Analyzers: facts a config can address a node by.

An analyzer is ``fn(node) -> value | None``: the fact's value when the
node carries it (``True`` for a plain tag, a position number for a side),
``None`` otherwise. Its registered name is the config key. :func:`run`
asks every analyzer about every walked node and writes the answers to
``node.metadata['lrp']``. Three ship: ``statistic_operand``,
``attention_weights``, ``input_conv``, ``bilinear``.
"""
from typing import Callable, Dict

import torch

from ..nodes import saved_tensors
from .graph import parents, reaches_input, reaches_input_without, is_input


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ANALYZERS: Dict[str, Callable] = {}


def register_analyzer(name, fn=None):
    r"""Register an analyzer: ``register_analyzer('name', fn)``, or as a
    decorator, ``@register_analyzer('name')``. ``fn(node)`` returns the
    fact's value for that node (``True`` for a plain tag, a position number
    for a side), or ``None`` when the node does not carry it. The name is
    the config key. Registering a name again overwrites it.
    """
    def _register(f):
        if not callable(f):
            raise TypeError(f"analyzer must be callable; got {type(f).__name__}")
        ANALYZERS[name] = f
        return f
    return _register if fn is None else _register(fn)


def run(plan) -> None:
    r"""Ask every registered analyzer about every walked node and write
    the answers onto ``node.metadata['lrp']`` under the analyzer's name."""
    if not plan or not ANALYZERS:
        return
    for node, _ in plan:
        try:
            md = node.metadata.setdefault('lrp', {})
        except (AttributeError, TypeError):
            continue                                     # test stand-ins without metadata
        for name, fn in ANALYZERS.items():
            value = fn(node)
            if value is not None:
                md[name] = value


# ---------------------------------------------------------------------------
# statistic_operand, and its cancellation test
# ---------------------------------------------------------------------------

_CANDIDATE_NODES = ('MulBackward', 'DivBackward', 'SubBackward')

_CANCEL_TOL = 1e-4      # fractional change in z per fractional change in src;
                        # true cancellations read ~1e-13, everything else >= ~0.4


def _recompute(node_name, a, b):
    """The node's output, rebuilt from its saved operands."""
    if 'Div' in node_name:
        return a / b
    if 'Sub' in node_name:
        return a - b
    return a * b


def cancels(z, src, tol: float = _CANCEL_TOL) -> bool:
    r"""``True`` when ``z`` does not move as ``src`` is scaled, or as a
    constant is added to every element of ``src`` (RMSNorm cancels the
    factor, mean-subtraction the constant).

    The movement of every element of ``z`` is read out with two calls:
    a first ``grad`` with an unfixed ``w`` records ``sum_i w[i] dz[i]/dsrc``,
    and differentiating ``(g * v).sum()`` with respect to ``w`` returns all
    ``(v * dz[i]/dsrc).sum()`` at once. ``tol`` is a relative rate: 1e-4
    means a 1 percent change of ``src`` moves ``z`` by under 0.0001
    percent.
    """
    w = torch.zeros_like(z, requires_grad=True)
    g = torch.autograd.grad(z, src, grad_outputs=w, create_graph=True,
                            allow_unused=True)[0]
    if g is None:
        return True                   # z does not depend on src: cannot move
    zn = float(z.detach().norm())
    sn = float(src.detach().norm())
    if zn == 0.0 or sn == 0.0:
        return True                   # nothing to measure against
    for v in (src, torch.ones_like(src)):
        dz = torch.autograd.grad((g * v).sum(), w, retain_graph=True)[0]
        if float(dz.detach().norm()) / zn / (float(v.detach().norm()) / sn) < tol:
            return True
    return False


def derived_operand(a_p, b_p):
    r"""Position of the operand that reaches the input only through its
    sibling, so it was computed from it, or ``None`` when both or neither
    does. A nomination, not a verdict: it says which side a statistic
    would be, not whether it is one. ``x * x.mean()`` has a derived side
    and cancels nothing.
    """
    a_alone = reaches_input_without(a_p, b_p)
    b_alone = reaches_input_without(b_p, a_p)
    if a_alone == b_alone:
        return None
    return 1 if a_alone else 0


@register_analyzer('statistic_operand')
def statistic_operand(node):
    r"""For a two-operand ``Mul``/``Div``/``Sub`` node, the position (0 or 1)
    of the operand that is a statistic of the other, else ``None``.
    """
    name = node.name()
    if not any(k in name for k in _CANDIDATE_NODES):
        return None
    ps = parents(node)                                   # aliases collapsed
    if len(ps) < 2 or not (reaches_input(ps[0]) and reaches_input(ps[1])):
        return None                                      # a scalar edge, a weight side, or off the input path
    position = derived_operand(ps[0], ps[1])                 # which side could be the statistic
    if position is None:
        return None                                      # x * x, or two independent operands
    a, b = (t.tensor for t in saved_tensors(node))
    if a is None or b is None:
        return None                                      # a native Sub saved nothing to measure with
    return position if cancels(_recompute(name, a, b), a if position == 1 else b) else None


@register_analyzer('input_conv')
def input_conv(node):
    r"""``True`` for a convolution that convolves the wrapped input itself
    (through views). The key for the z-box input rule, ``rule={**BASE,
    'input_conv': ('zbox', {'low': lo, 'high': hi})}``. A model that
    normalizes inside its forward before the first convolution has no
    such node; wrap the normalized tensor, or register your own fact.
    """
    if 'ConvolutionBackward' not in node.name():
        return None
    src = parents(node)[0]                               # what it convolves, aliases collapsed
    return True if src is not None and is_input(src) else None


# ---------------------------------------------------------------------------
# bilinear
# ---------------------------------------------------------------------------

_PRODUCT_NODES = ('AddmmBackward', 'MmBackward', 'BmmBackward', 'MulBackward', 'DivBackward')


def _from_input(node):
    """Per operand position: does it come from the wrapped input."""
    return [p is not None and reaches_input(p) for p in parents(node, skip_aliases=False)]


@register_analyzer('bilinear')
def bilinear(node):
    r"""``True`` for a matmul (addmm, mm, bmm) whose two operands both come
    from the wrapped input: a bilinear product, addressed by this fact
    (a two-sided rule, or a one-sided rule with ``side``)."""
    if not any(k in node.name() for k in ('AddmmBackward', 'MmBackward', 'BmmBackward')):
        return None
    return True if sum(_from_input(node)) == 2 else None


@register_analyzer('weight_operand')
def weight_operand(node):
    r"""On a product (matmul, mul, div) with exactly one operand from the
    wrapped input, the position of the other operand: its weight, a
    parameter, a constant or a frozen tensor. The installers send all
    relevance to the input side. A fact the installers read; not one a
    config needs to address."""
    if not any(k in node.name() for k in _PRODUCT_NODES):
        return None
    positions = [t.position for t in saved_tensors(node) if t.position is not None]
    live = _from_input(node)
    from_input = [i for i in positions if i < len(live) and live[i]]
    if len(from_input) != 1:
        return None
    return next(i for i in positions if i != from_input[0])


# ---------------------------------------------------------------------------
# attention_weights: which operand of a bilinear product is the softmax
# ---------------------------------------------------------------------------

_AVERAGE_TOL = 1e-4


def is_weighted_average(m, tol: float = _AVERAGE_TOL) -> bool:
    r"""``True`` when each row of ``m`` holds the weights of a weighted
    average: no weight negative, each row totalling 1. In ``m @ b`` such an
    ``m`` only picks points among the rows of ``b``, so everything in the
    product came from ``b``. The row axis is the one the multiply
    contracts; :func:`attention_weights` asks the question per operand and
    transposes for the second one. Reading ``m`` assumes it holds still
    while ``b`` moves, which fails when ``m`` is computed from ``b``
    (``softmax(V @ V.mT) @ V``); :func:`attention_weights` closes that case
    with an independence probe.
    """
    m = m.detach()
    return (float(m.min()) >= -tol
            and float((m.sum(-1) - 1.0).abs().max()) < tol)


@register_analyzer('attention_weights')
def attention_weights(node):
    r"""For a ``BmmBackward`` with both operands from the input, the position
    of the operand holding the weights of a weighted average
    (:func:`is_weighted_average`, asked per operand), else ``None``. A
    row-stochastic operand that the other operand depends on emits
    nothing. Only naming: ``('detach', {'by': 'attention_weights'})`` in
    the config is what detaches it, so ``bmm(A, V)`` and
    ``bmm(V.mT, A.mT)`` receive the same attribution.
    """
    if 'BmmBackward' not in node.name():
        return None
    ps = parents(node)
    if len(ps) < 2 or not (reaches_input(ps[0]) and reaches_input(ps[1])):
        return None                                      # a weight side: a linear layer, no role to name
    a, b = (t.tensor for t in saved_tensors(node))
    if a is None or b is None:
        return None
    first = is_weighted_average(a)
    second = is_weighted_average(b.transpose(-2, -1))
    if first == second:
        return None                                      # neither side looks like weights, or both do
    cand, other = (a, b) if first else (b, a)
    if not _independent(cand, other):
        return None                                      # weights computed from the values (softmax(V @ V.mT) @ V) are not weights
    return 0 if first else 1


def _independent(y, x):
    r"""``True`` when ``y`` does not depend on ``x`` through the graph
    (``torch.autograd.grad`` returns ``None`` under ``allow_unused``).
    Tensors outside each other's graphs, or non-grad tensors, count as
    independent."""
    if not (isinstance(y, torch.Tensor) and isinstance(x, torch.Tensor)
            and y.requires_grad and x.requires_grad):
        return True
    try:
        g = torch.autograd.grad(y.sum(), x, retain_graph=True,
                                allow_unused=True)[0]
    except RuntimeError:
        return True
    return g is None
