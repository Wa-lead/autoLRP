r"""Graph walker and hook runner: :func:`walk` lists the nodes and their
installers, :func:`execute` installs the hooks and runs ``backward``,
:func:`graph_lrp` does both for :meth:`autolrp.LRPTensor.lrp`. The
attributed scalar is chosen by slicing the output first
(``out[0, k].lrp()``).
"""
import warnings
from typing import Callable, List, Optional, Tuple

import torch

from . import analysis
from .strategies import (
    Strategy, EXPLICIT_STRATEGY, match_installer, is_shape_node,
    build_strategy,
)
from ..config import LRPConfig
from .analysis import leaf_reach, parents


PlanItem = Tuple[object, Optional[Callable]]   # (node, installer or None)


# ---------------------------------------------------------------------------
# plan report
# ---------------------------------------------------------------------------

def plan_report(plan) -> dict:
    r"""``{'rule': [...], 'native_fallback': [...], 'leaves': int}`` over a
    plan: which node names have an installer and which fall to the native
    gradient.
    """
    report = {'rule': [], 'native_fallback': [], 'leaves': 0}
    for node, installer in plan:
        name = node.name()
        if 'AccumulateGrad' in name:
            report['leaves'] += 1
        elif installer is None:
            report['native_fallback'].append(name)
        else:
            report['rule'].append(name)
    return report


_UNMATCHED_WARNED: set = set()


# ---------------------------------------------------------------------------
# walk
# ---------------------------------------------------------------------------

def walk(output: torch.Tensor,
         strategy: Optional[Strategy] = None,
         config=None) -> List[PlanItem]:
    r"""BFS over the autograd graph from ``output.grad_fn``; returns
    ``[(node, installer)]`` in BFS order, ``installer=None`` where no
    strategy pattern matches. ``config`` is accepted for symmetry and not
    read.
    """
    if strategy is None:
        strategy = EXPLICIT_STRATEGY

    plan: List[PlanItem] = []
    if output.grad_fn is None:
        return plan

    visited = set()
    queue = [output.grad_fn]
    while queue:
        node = queue.pop(0)
        if node is None or id(node) in visited:
            continue
        visited.add(id(node))
        _, installer = match_installer(node.name(), strategy)
        plan.append((node, installer))
        for parent in parents(node, skip_aliases=False):
            if parent is not None and id(parent) not in visited:
                queue.append(parent)

    # An unmatched node matters only on the path to a wrapped input;
    # one that reaches no input (an embedding lookup of ids, a
    # parameter-only branch) receives relevance that lands nowhere.
    reach = leaf_reach([n for n, _ in plan])
    for node, installer in plan:
        name = node.name()
        if (installer is None and 'AccumulateGrad' not in name
                and reach.get(id(node), False)
                and name not in _UNMATCHED_WARNED):
            _UNMATCHED_WARNED.add(name)
            warnings.warn(
                f"autoLRP: no installer matched {name!r}; the native "
                f"gradient runs there. Register one via "
                f"autolrp.register_installer.", UserWarning, stacklevel=3)
    return plan


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

def explain(output: torch.Tensor, config=None, strategy=None):
    r"""What would run at every node of the graph under ``output``, as a
    list of ``(node_name, key, what)`` rows in walk order. ``key`` is the
    config entry that addressed the node (a fact or the node name) and
    ``what`` the rule function's name; for nodes without a rule entry
    ``key`` is ``None`` and ``what`` names the installer, ``'native
    gradient'`` when nothing installs. The hooks are installed and
    removed again; no backward runs. :func:`explain_summary` formats the
    rows as a printable table string.
    """
    from . import install as _install
    if config is None:
        config = LRPConfig()
    plan = walk(output, strategy=strategy or build_strategy(config), config=config)
    analysis.run(plan)
    rows, handles = [], []
    _install._TRACE = trace = []
    try:
        for node, installer in plan:
            name = node.name()
            if 'AccumulateGrad' in name:
                continue
            if installer is None:
                rows.append((name, None, 'native gradient'))
                continue
            before = len(trace)
            h = installer(node, config)
            if h is not None:
                handles.extend(h if isinstance(h, (list, tuple)) else [h])
            if len(trace) > before:
                for key, what in trace[before:]:
                    rows.append((name, key, what))
            elif h is None:
                rows.append((name, None, 'native gradient'))
            else:
                rows.append((name, None, installer.__name__))
    finally:
        _install._TRACE = None
        for h in handles:
            h.remove()
    return rows


def explain_summary(rows) -> str:
    r"""One line per distinct ``(node, key, what)`` with its count."""
    from collections import Counter
    c = Counter(rows)
    w = max((len(n) for n, _, _ in c), default=4)
    lines = [f"{'count':>5}  {'node':<{w}}  {'key':<20}  what"]
    for (n, k, what), cnt in sorted(c.items(), key=lambda kv: (-kv[1], kv[0][0])):
        lines.append(f"{cnt:>5}  {n:<{w}}  {str(k):<20}  {what}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------

def _unit_seed(t: torch.Tensor) -> torch.Tensor:
    r"""Return a :math:`+1` seed tensor matching ``t``'s shape.

    Positive :math:`R` values then indicate inputs that increase the
    seeded scalar; negative :math:`R` indicates inputs that decrease it.
    Matches the zennit/captum convention.
    """
    return torch.ones_like(t.detach())


def _backward(output: torch.Tensor) -> None:
    r"""Seed ``output`` with :math:`+1` and run backward.

    Uses ``retain_graph=True`` so the :class:`LRPTensor` input-side
    hook can store the incoming gradient as ``.relevance``.
    """
    output.backward(gradient=_unit_seed(output), retain_graph=True)


# ---------------------------------------------------------------------------
# Capture prehooks: read-only per-node taps for ``capture_layers``
# ---------------------------------------------------------------------------
# Fired at every non-shape planned node BEFORE the installer's hook runs,
# so the captured value is the original incoming relevance.

_CAPTURE_PREHOOK_WARNED: set = set()


def _install_capture_prehooks(plan, store):
    r"""Prehooks that record each node's incoming relevance into ``store``
    under ``(node_name, plan_index)``; shape-routing nodes are skipped.
    Returns the handles. A node whose ``register_prehook`` raises is
    warned about once and left out of ``store``.
    """
    handles: list = []
    for idx, (node, installer) in enumerate(plan):
        if installer is None:
            continue
        name = node.name()
        if is_shape_node(name):
            continue
        key = (name, idx)

        def _tap(go, _key=key, _store=store):
            if go and go[0] is not None:
                _store[_key] = go[0].detach().clone()
            return None

        try:
            handles.append(node.register_prehook(_tap))
        except Exception as exc:
            if name not in _CAPTURE_PREHOOK_WARNED:
                _CAPTURE_PREHOOK_WARNED.add(name)
                warnings.warn(
                    f"autoLRP: could not register a capture prehook on "
                    f"{name!r} ({exc}); its relevance will be missing from "
                    f"layer_relevances despite config.capture_layers. "
                    f"Upgrade torch if Node.register_prehook is "
                    f"unavailable.", UserWarning, stacklevel=3)
    return handles


def execute(plan: List[PlanItem], output: torch.Tensor,
            config, layer_relevances: Optional[dict] = None) -> None:
    r"""Install the hooks of ``plan``, run ``backward`` from ``output``, and
    remove the hooks (in a ``finally``). With ``config.capture_layers``
    the per-node relevances are recorded into ``layer_relevances``.
    """
    handles: list = []

    # Registered before the installers, so they see the incoming R_out.
    store = (layer_relevances
             if (config is not None and config.capture_layers
                 and layer_relevances is not None) else None)
    if store is not None:
        handles.extend(_install_capture_prehooks(plan, store))

    try:
        for node, installer in plan:
            if installer is None:
                continue
            result = installer(node, config)
            if result is None:
                continue
            if isinstance(result, (list, tuple)):
                handles.extend(result)
            else:
                handles.append(result)
        _backward(output)
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# graph_lrp -- entry point used by tensor.py
# ---------------------------------------------------------------------------

def graph_lrp(output: torch.Tensor,
              config=None,
              layer_relevances: Optional[dict] = None,
              strategy: Optional[Strategy] = None):
    r"""Walk, analyze, install, backward. ``config`` defaults to
    ``LRPConfig()``. Returns ``layer_relevances`` when
    ``config.capture_layers`` is set, else ``None``.
    """
    if config is None:
        config = LRPConfig()
    if strategy is None:
        strategy = build_strategy(config)

    # Warn-once state is per call, so a second model warns again.
    from .install import _MISSING_STATE_WARNED
    from .rules import _GAMMA_DEGENERATION_WARNED
    from ..forward.intercept import _INPLACE_REMAP_WARNED
    _UNMATCHED_WARNED.clear()
    _CAPTURE_PREHOOK_WARNED.clear()
    _MISSING_STATE_WARNED.clear()
    _GAMMA_DEGENERATION_WARNED.clear()
    _INPLACE_REMAP_WARNED.clear()

    plan = walk(output, strategy=strategy, config=config)
    # A wrapped input leaf must be reachable. `_lrp_init` is set by
    # tensor() and dropped by detach(), so a `.detach().requires_grad_()`
    # inside the forward, which severs the path, fails this check.
    if not any('AccumulateGrad' in n.name()
               and getattr(getattr(n, 'variable', None), '_lrp_init', False)
               and not isinstance(getattr(n, 'variable', None),
                                  torch.nn.Parameter)
               for n, _ in plan):
        raise RuntimeError(
            "autoLRP: the autograd graph under this output contains no "
            "wrapped-input leaf -- the gradient road from the output back "
            "to an autolrp.tensor(...) input was severed "
            "(torch.no_grad() inside the forward, a .detach() or "
            ".detach().requires_grad_(True) re-attachment, or a numpy "
            "round-trip). Relevance cannot reach the wrapped input.")
    analysis.run(plan)          # ANALYZE: write structural facts to node.metadata
    execute(plan, output, config, layer_relevances=layer_relevances)

    return layer_relevances if config.capture_layers else None
