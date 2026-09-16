r"""Reading the autograd graph: the producers of a node's operands,
which subgraphs reach a wrapped input, a node's facts, and a layer's
bias. What a node saved is read in :mod:`autolrp.nodes`, beside the
table that says where each kind keeps it. Nothing here decides anything
about relevance.
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


def is_leaf(node) -> bool:
    r"""``True`` iff ``node`` is a leaf accumulator (``AccumulateGrad``), the
    node autograd gives a tensor nothing computed: a parameter, a constant
    made live, or the wrapped input."""
    return 'AccumulateGrad' in node.name()


def is_input(node) -> bool:
    r"""``True`` iff ``node`` is the leaf of a tensor the user wrapped with
    :func:`autolrp.tensor`. Relevance flows to inputs; every other leaf, a
    parameter, a constant the forward intercept made live, a plain tensor
    with ``requires_grad``, is a weight."""
    return is_leaf(node) and bool(getattr(getattr(node, 'variable', None), '_lrp_init', False))


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


def bfs_order(nodes, removed=None):
    r"""Every node reachable from ``nodes`` through :func:`parents`, each
    once, breadth-first (children before parents): the plan order the
    engine, ``explain`` and the analyzers see. Lazy, so a caller that
    stops early walks only what it looked at. ``removed`` is a node the
    walk refuses to enter, which cuts its subgraph out."""
    seen, queue = {}, deque(nodes)                 # id -> node: holds the wrappers, so an id is never reused mid-walk
    while queue:
        node = queue.popleft()
        if node is None or node is removed or id(node) in seen:
            continue
        seen[id(node)] = node
        yield node
        queue.extend(parents(node, skip_aliases=False))


def reaches_input_map(nodes) -> Dict[int, bool]:
    r""":func:`reaches_input` for every node at once: ``map[id(node)]`` is
    ``True`` iff ``node``'s subgraph contains an input leaf. A fold over
    :func:`topo_order`, so every parent is decided before its children."""
    reach: Dict[int, bool] = {}
    for node in topo_order(nodes):
        if is_leaf(node):
            reach[id(node)] = is_input(node)
        else:
            reach[id(node)] = any(reach.get(id(p), False)
                                for p in parents(node, skip_aliases=False)
                                if p is not None)
    return reach


def reaches_input(node) -> bool:
    """``True`` iff relevance sent to ``node`` can arrive at a wrapped input:
    an input leaf lies under it. A parameter or a constant (including one
    the forward intercept made live) does not."""
    return any(is_input(n) for n in bfs_order([node]))


def reaches_input_without(start, removed) -> bool:
    r""":func:`reaches_input` with the subgraph under ``removed`` cut out."""
    return any(is_input(n) for n in bfs_order([start], removed))


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
    r"""Producing nodes of ``node``'s operand positions, in position order; ``None``
    for a position with no producer. With ``skip_aliases`` (default) chains
    of ``AliasBackward`` are collapsed to the first real op.
    """
    ps = [q for q, _ in getattr(node, 'next_functions', ())]
    return [_skip_aliases(q) for q in ps] if skip_aliases else ps


def find_bias(node, position: int) -> Optional[torch.Tensor]:
    r"""The 1-D bias tensor in operand ``position`` of ``node`` (0 for
    ``Addmm``, 2 for ``Convolution``), detached, or ``None`` when the position
    is empty, computed, or not a vector.
    """
    ps = parents(node, skip_aliases=False)
    if len(ps) <= position:
        return None
    bias_node = ps[position]
    if bias_node is not None and is_leaf(bias_node):
        v = bias_node.variable
        if v.ndim == 1:
            return v.detach()
    return None
