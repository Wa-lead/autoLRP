r"""Graph analysis: facts about autograd nodes.

An analyzer is ``fn(nodes) -> {node: fact}`` over the whole plan, run
between :func:`~autoLRP.backward.engine.walk` and
:func:`~autoLRP.backward.engine.execute`. It reads the graph and the
saved tensors, never module names, and writes ``node.metadata['lrp']``.
It writes only the fact it is registered as, so a config key can be
checked against :data:`ANALYZERS` at construction; the value may carry
data (``statistic_operand`` writes the slot to detach). Facts are
config keys: ``rule={**BASE, 'my_fact': ...}`` reaches the nodes an
analyzer registered as ``my_fact`` tagged, before the name entry.
"""
from typing import Callable, Dict

import torch


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ANALYZERS: Dict[str, Callable] = {}


def register_analyzer(name_or_fn=None):
    r"""Register an analyzer, ``@register_analyzer`` or
    ``@register_analyzer('name')``. It returns ``{node: spec}`` for the
    nodes that carry its fact; ``spec`` is the fact's value (``True`` for a
    plain tag, a slot number for a side), the registered name itself
    (shorthand for a ``True`` tag), or the dict ``{'name': value}`` with
    the registered name. Registering a name again overwrites it.
    """
    def _register(fn, name):
        ANALYZERS[name] = fn
        return fn
    if callable(name_or_fn):
        return _register(name_or_fn, name_or_fn.__name__)
    def _decorator(fn):
        return _register(fn, name_or_fn if name_or_fn is not None
                         else fn.__name__)
    return _decorator

def run(plan) -> None:
    r"""Execute every registered analyzer once over the walked graph and
    write the returned facts onto ``node.metadata['lrp']`` per the one
    contract documented on :func:`register_analyzer`."""
    if not plan or not ANALYZERS:
        return
    nodes = [n for n, _ in plan]
    for _id, fn in ANALYZERS.items():
        facts = fn(nodes)
        if not facts:
            continue
        for node, spec in facts.items():
            if isinstance(spec, str):
                spec = {spec: True}
            elif not isinstance(spec, dict):
                spec = {_id: spec}          # a bare value is the fact's value
            try:
                md = node.metadata.setdefault('lrp', {})
            except (AttributeError, TypeError):
                continue  # nodes without a metadata dict (test stand-ins)
            for k, v in spec.items():
                if k != _id:
                    raise ValueError(
                        f"analyzer '{_id}' wrote fact {k!r}; an analyzer "
                        f"writes only the fact it is registered as, so a "
                        f"config key can be checked against the registry")
                if k in md and md[k] != v:
                    raise ValueError(
                        f"fact '{k}' written twice with different values "
                        f"({md[k]!r} vs {v!r}); analyzers must not "
                        f"conflict on a fact name")
                md[k] = v

def node_facts(node) -> dict:
    r"""Return ``node.metadata['lrp']`` or an empty dict. Safe on
    stand-in nodes that lack ``metadata``."""
    md = getattr(node, 'metadata', None)
    if isinstance(md, dict):
        return md.get('lrp', {})
    return {}


# ---------------------------------------------------------------------------
# Which subgraphs reach a wrapped input
# ---------------------------------------------------------------------------


def is_weight_leaf(var) -> bool:
    """A leaf that carries no relevance of its own. Relevance flows to
    what the user wrapped with :func:`autoLRP.tensor`; every other leaf,
    an ``nn.Parameter``, a constant the forward intercept made live, a
    plain tensor with ``requires_grad``, is a weight."""
    return not getattr(var, '_lrp_init', False)


def leaf_reach(nodes) -> Dict[int, bool]:
    r"""``reach[id(fn)] = True`` iff ``fn``'s subgraph contains a wrapped
    input leaf (:func:`is_weight_leaf`). One post-order pass over the
    union of the given subgraphs.
    """
    reach: Dict[int, bool] = {}
    for root in nodes:
        if root is None or id(root) in reach:
            continue
        stack = [(root, False)]
        while stack:
            fn, post = stack.pop()
            if fn is None:
                continue
            fid = id(fn)
            if post:
                reach[fid] = any(
                    reach.get(id(p), False)
                    for p in parents(fn, skip_aliases=False)
                    if p is not None)
                continue
            if fid in reach:
                continue
            if 'AccumulateGrad' in fn.name():
                var = getattr(fn, 'variable', None)
                reach[fid] = not is_weight_leaf(var)
                continue
            reach[fid] = False           # placeholder; fixed on post-visit
            stack.append((fn, True))
            for p in parents(fn, skip_aliases=False):
                if p is not None and id(p) not in reach:
                    stack.append((p, False))
    return reach


def reaches_input(fn) -> bool:
    """``True`` iff ``fn`` reaches a wrapped input leaf. A parameter or a
    constant (including a constant made live by the forward intercept)
    does not; only the path from the user's ``tensor(...)`` does."""
    return _reaches_input_avoiding(fn, None)


def _reaches_input_avoiding(start_fn, forbidden_fn) -> bool:
    r"""``True`` iff ``start_fn`` reaches an input leaf without passing
    through ``forbidden_fn``.
    """
    if start_fn is None:
        return False
    seen = set()
    stack = [start_fn]
    while stack:
        fn = stack.pop()
        if fn is None or fn is forbidden_fn:
            continue
        fid = id(fn)
        if fid in seen:
            continue
        seen.add(fid)
        if 'AccumulateGrad' in fn.name():
            var = getattr(fn, 'variable', None)
            if not is_weight_leaf(var):
                return True
            continue                      # parameter leaf: keep searching
        for parent in parents(fn, skip_aliases=False):
            if parent is not None and parent is not forbidden_fn:
                stack.append(parent)
    return False


# ---------------------------------------------------------------------------
# statistic_operand
# ---------------------------------------------------------------------------

_CANDIDATE_FAMILIES = ('MulBackward', 'DivBackward', 'SubBackward')


def _skip_aliases(fn):
    r"""Collapse a chain of ``AliasBackward`` nodes to the first real op;
    the subclass inserts an alias at every op boundary, and anchoring a
    path test on the alias lets a sibling path slip past it.
    """
    while fn is not None and 'AliasBackward' in fn.name():
        nfs = getattr(fn, 'next_functions', ())
        fn = nfs[0][0] if nfs else None
    return fn


def parents(node, skip_aliases: bool = True):
    r"""Producing nodes of ``node``'s operand slots, in slot order; ``None``
    for a slot with no producer. With ``skip_aliases`` (default) chains
    of ``AliasBackward`` are collapsed to the first real op.
    """
    ps = [q for q, _ in getattr(node, 'next_functions', ())]
    return [_skip_aliases(q) for q in ps] if skip_aliases else ps


def operands(node):
    r"""Saved operand tensors ``(a, b)`` of a two-operand node, or
    ``(None, None)``; native ops save ``_saved_self``/``_saved_other``,
    our wrapped ops save through ``saved_tensors``.
    """
    a = getattr(node, '_saved_self', None)
    b = getattr(node, '_saved_other', None)
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a, b
    saved = getattr(node, 'saved_tensors', None)
    if saved and len(saved) >= 2 and all(isinstance(t, torch.Tensor) for t in saved[:2]):
        return saved[0], saved[1]
    return None, None


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


def _parameter_slot(a_p, b_p, reach):
    r"""Slot of an operand reaching no input, or ``None`` when both or
    neither does. Relevance sent to such a side lands nowhere.
    """
    a_live = reach.get(id(a_p), False)
    b_live = reach.get(id(b_p), False)
    if a_live == b_live:
        return None
    return 1 if a_live else 0


def _dominated_slot(a_p, b_p):
    r"""Slot of the operand that reaches a model input only through its
    sibling, or ``None`` when both or neither does.

    Nomination, not verdict: a dominated operand was computed from the
    other one, which says where the zeros would go, not whether zeros
    are right. ``x * x.mean()`` is dominated and cancels nothing.
    """
    a_alone = _reaches_input_avoiding(a_p, b_p)
    b_alone = _reaches_input_avoiding(b_p, a_p)
    if a_alone == b_alone:
        return None
    return 1 if a_alone else 0


@register_analyzer('statistic_operand')
def statistic_operand(nodes) -> Dict[object, dict]:
    r"""For each two-operand ``Mul``/``Div``/``Sub`` node, name the operand
    that is a statistic of the other; fact value is its slot, 0 or 1.

    Two grounds, cheapest first. A side that reaches no wrapped input at
    all (parameters, constants): relevance sent there lands nowhere. A
    confirmed cancellation: with both sides from the input, the side that
    still reaches the input when the other is removed is the source, and
    :func:`cancels` measures whether the product discards a property of
    it (scale or level). Two independent operands emit nothing.

    Only naming: ``BASE`` carries ``{'statistic_operand': ('detach',
    {'by': 'statistic_operand'})}``, which detaches the named side;
    overriding that key changes what runs there.
    """
    reach = leaf_reach(nodes)
    out: Dict[object, dict] = {}
    for n in nodes:
        name = n.name()
        if not any(k in name for k in _CANDIDATE_FAMILIES):
            continue
        ps = parents(n)                              # aliases collapsed
        if len(ps) < 2 or ps[0] is None or ps[1] is None or ps[0] is ps[1]:
            continue                                 # scalar edge, or x * x
        a, b = operands(n)
        if a is None or b is None:
            continue                                 # native Sub saves nothing
        a_p, b_p = ps[0], ps[1]

        slot = _parameter_slot(a_p, b_p, reach)      # ground 1: role
        if slot is not None:
            out[n] = {'statistic_operand': slot}
            continue

        slot = _dominated_slot(a_p, b_p)             # nomination
        if slot is None:
            continue
        src = a if slot == 1 else b
        if cancels(_recompute(name, a, b), src):     # ground 2: measurement
            out[n] = {'statistic_operand': slot}
    return out


# ---------------------------------------------------------------------------
# input_conv
# ---------------------------------------------------------------------------


@register_analyzer('input_conv')
def input_conv(nodes) -> Dict[object, str]:
    r"""Tag a ``ConvolutionBackward`` that reads the model input: its saved
    input has at most four channels and a leaf producer. Used as the key
    for the z-box input rule, ``rule={**BASE, 'input_conv': ('zbox',
    {'low': lo, 'high': hi})}``.
    """
    out: Dict[object, str] = {}
    for n in nodes:
        if 'ConvolutionBackward' not in n.name():
            continue
        saved_inp = getattr(n, '_saved_input', None)
        if saved_inp is None or saved_inp.ndim < 4 or saved_inp.shape[1] > 4:
            continue
        for parent_fn in parents(n, skip_aliases=False):
            if parent_fn is not None and 'AccumulateGrad' in parent_fn.name():
                out[n] = 'input_conv'
                break
    return out


# ---------------------------------------------------------------------------
# weights_operand
# ---------------------------------------------------------------------------

_AVERAGE_TOL = 1e-4


def is_weighted_average(m, tol: float = _AVERAGE_TOL) -> bool:
    r"""``True`` when each row of ``m`` holds the weights of a weighted
    average: no weight negative, each row totalling 1. In ``m @ b`` such an
    ``m`` only picks points among the rows of ``b``, so everything in the
    product came from ``b``. The row axis is the one the multiply
    contracts; :func:`weights_operand` asks the question per operand and
    transposes for the second one. Reading ``m`` assumes it holds still
    while ``b`` moves, which fails when ``m`` is computed from ``b``
    (``softmax(V @ V.mT) @ V``); :func:`weights_operand` closes that case
    with an independence probe.
    """
    m = m.detach()
    return (float(m.min()) >= -tol
            and float((m.sum(-1) - 1.0).abs().max()) < tol)


@register_analyzer('weights_operand')
def weights_operand(nodes) -> Dict[object, dict]:
    r"""For each ``BmmBackward`` with both operands from the input, name the
    operand holding the weights of a weighted average
    (:func:`is_weighted_average`, asked per operand); fact value is its
    slot. A row-stochastic operand that the other operand depends on
    emits no fact. Only naming: ``('detach', {'by': 'weights_operand'})``
    in the config is what detaches it, so ``bmm(A, V)`` and
    ``bmm(V.mT, A.mT)`` receive the same attribution.
    """
    out: Dict[object, dict] = {}
    reach = leaf_reach(nodes)
    for n in nodes:
        if 'BmmBackward' not in n.name():
            continue
        ps = parents(n, skip_aliases=False)
        if len(ps) < 2 or ps[0] is None or ps[1] is None or ps[0] is ps[1]:
            continue
        # Both operands must come from the input: with one, the node is
        # a linear layer whose weight is the other operand, and there is
        # no role to name.
        if not (reach.get(id(ps[0]), False) and reach.get(id(ps[1]), False)):
            continue
        a = getattr(n, '_saved_self', None)
        b = getattr(n, '_saved_mat2', None)
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) \
                or a is b:
            continue
        first = is_weighted_average(a)
        second = is_weighted_average(b.transpose(-2, -1))
        if first == second:
            continue
        # A row-stochastic operand the other operand depends on
        # (softmax(V @ V.mT) @ V) is not "the weights"; emit nothing.
        cand, other = (a, b) if first else (b, a)
        if not _independent(cand, other):
            continue
        out[n] = {'weights_operand': 0 if first else 1}
    return out


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
