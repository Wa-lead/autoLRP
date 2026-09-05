r"""Reading the autograd graph: the producers of a node's operands,
which subgraphs reach a wrapped input, a node's facts and saved
operands, and a layer's bias. Every other backward file asks these
questions; none of them decides anything about relevance.
"""
from collections import deque
from typing import Dict, Optional

import torch


def node_facts(node) -> dict:
    r"""Return ``node.metadata['lrp']`` or an empty dict. Safe on
    stand-in nodes that lack ``metadata``."""
    md = getattr(node, 'metadata', None)
    if isinstance(md, dict):
        return md.get('lrp', {})
    return {}


def is_weight_leaf(var) -> bool:
    """A leaf that carries no relevance of its own. Relevance flows to
    what the user wrapped with :func:`autolrp.tensor`; every other leaf,
    an ``nn.Parameter``, a constant the forward intercept made live, a
    plain tensor with ``requires_grad``, is a weight."""
    return not getattr(var, '_lrp_init', False)


def is_leaf(node) -> bool:
    r"""``True`` iff ``node`` is a leaf accumulator (``AccumulateGrad``), the
    node autograd gives a tensor nothing computed: a parameter, a constant
    made live, or the wrapped input."""
    return 'AccumulateGrad' in node.name()


def is_input_leaf(node) -> bool:
    r"""``True`` iff ``node`` is the leaf of a tensor the user wrapped with
    :func:`autolrp.tensor`; parameter and constant leaves are weights."""
    return is_leaf(node) and not is_weight_leaf(getattr(node, 'variable', None))


def topo_order(nodes):
    r"""Every node reachable from ``nodes`` through :func:`parents`, each
    once, parents before children (iterative post-order DFS). The list
    holds the node wrappers, so anything keyed on ``id(node)`` stays valid
    while it is alive."""
    order, seen = [], set()
    for root in nodes:
        if root is None or id(root) in seen:
            continue
        stack = [(root, False)]
        while stack:
            node, post = stack.pop()
            if post:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in parents(node, skip_aliases=False):
                if p is not None and id(p) not in seen:
                    stack.append((p, False))
    return order


def bfs_order(nodes):
    r"""Every node reachable from ``nodes`` through :func:`parents`, each
    once, breadth-first from them (children before parents): the
    plan order the engine, ``explain`` and the analyzers see."""
    order, seen, queue = [], set(), deque(nodes)
    while queue:
        node = queue.popleft()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        order.append(node)
        for p in parents(node, skip_aliases=False):
            if p is not None and id(p) not in seen:
                queue.append(p)
    return order


def leaf_reach(nodes) -> Dict[int, bool]:
    r"""``reach[id(node)] = True`` iff ``node``'s subgraph contains a wrapped
    input leaf (:func:`is_weight_leaf`): a fold over :func:`topo_order`,
    so every parent is decided before its children."""
    reach: Dict[int, bool] = {}
    for node in topo_order(nodes):
        if is_leaf(node):
            reach[id(node)] = is_input_leaf(node)
        else:
            reach[id(node)] = any(reach.get(id(p), False)
                                for p in parents(node, skip_aliases=False)
                                if p is not None)
    return reach


def reaches_input(node) -> bool:
    """``True`` iff ``node`` reaches a wrapped input leaf. A parameter or a
    constant (including a constant made live by the forward intercept)
    does not; only the path from the user's ``tensor(...)`` does."""
    return _reaches_input_avoiding(node, None)


def _reaches_input_avoiding(start, forbidden) -> bool:
    r"""``True`` iff ``start`` reaches an input leaf without passing
    through ``forbidden``. A search that stops at the first input leaf,
    not a fold: it runs once per installed node.
    """
    if start is None:
        return False
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node is None or node is forbidden:
            continue
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        if is_leaf(node):
            if is_input_leaf(node):
                return True
            continue                      # parameter leaf: keep searching
        for parent in parents(node, skip_aliases=False):
            if parent is not None and parent is not forbidden:
                stack.append(parent)
    return False


def _skip_aliases(node):
    r"""Collapse a chain of ``AliasBackward`` nodes to the first real op;
    the subclass inserts an alias at every op boundary, and anchoring a
    path test on the alias lets a sibling path slip past it.
    """
    while node is not None and 'AliasBackward' in node.name():
        nfs = getattr(node, 'next_functions', ())
        node = nfs[0][0] if nfs else None
    return node


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


def find_bias(node, slot: int) -> Optional[torch.Tensor]:
    r"""The 1-D bias tensor in operand ``slot`` of ``node`` (0 for
    ``Addmm``, 2 for ``Convolution``), detached, or ``None`` when the slot
    is empty, computed, or not a vector.
    """
    ps = parents(node, skip_aliases=False)
    if len(ps) <= slot:
        return None
    bias_node = ps[slot]
    if bias_node is not None and is_leaf(bias_node):
        v = bias_node.variable
        if v.ndim == 1:
            return v.detach()
    return None


def live_slots(node):
    """Operand slots of ``node`` that relevance can flow to: the ones with a
    producer. A constant operand (a Python number, a tensor without grad)
    has none, so ``next_functions`` holds ``None`` there."""
    return [i for i, p in enumerate(parents(node, skip_aliases=False))
            if p is not None]


def input_slots(node, slots):
    """Which of ``slots`` hold an operand that reaches the wrapped input."""
    ps = parents(node, skip_aliases=False)
    return [i for i in slots if i < len(ps) and reaches_input(ps[i])]
