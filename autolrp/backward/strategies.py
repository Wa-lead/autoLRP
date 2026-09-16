r"""Strategy table: ``{pattern: installer}`` matched by longest substring
of ``grad_fn.name()``, so ``'LeakyReluBackward'`` beats
``'ReluBackward'`` whatever the order. This is the internal table;
user config keys are matched exactly (:mod:`autolrp.backward.resolve`).
``None`` or a missing key means the native gradient runs. Extend with
:func:`register_installer` or :func:`merge`.
"""
from typing import Callable, Dict, Optional

from .install import (
    install_matmul, install_conv, install_sdpa,
    install_mean_or_sum, install_norm, install_cumsum, install_mul, install_div,
    install_add, install_softmax, install_layernorm, install_elementwise,
    install_passthrough, install_noop,
)
from ..nodes import (PRODUCT_NODES, MUL_NODES, ADD_NODES, REDUCTION_NODES,
                     CUMSUM_NODES, ELEMENTWISE_NODES, LAYERNORM_NODES, SOFTMAX_NODES,
                     SDPA_NODES, PASSTHROUGH_NODES, ROUTING_NODES, SHAPE_NODES)


Strategy = Dict[str, Optional[Callable]]
Installer = Callable


# Matched by longest substring; insertion order only breaks exact-length
# ties (see the module docstring). Every name comes from autolrp.nodes;
# the two-operand kinds map name by name because each name has its own
# installer, and the check below keeps them complete.
EXPLICIT_STRATEGY: Strategy = {
    'AddmmBackward':          install_matmul,
    'MmBackward':             install_matmul,
    'BmmBackward':            install_matmul,
    'ConvolutionBackward':    install_conv,
    **{n: install_sdpa for n in SDPA_NODES},

    'AddBackward':            install_add,
    'SubBackward':            install_add,
    'MulBackward':            install_mul,
    'DivBackward':            install_div,

    'MeanBackward':           install_mean_or_sum,
    'SumBackward':            install_mean_or_sum,
    'LinalgVectorNormBackward': install_norm,
    'NormBackward':           install_norm,
    'CumsumBackward':         install_cumsum,

    # One-operand kinds: the rule in config.rule.
    **{n: install_elementwise for n in ELEMENTWISE_NODES},
    **{n: install_layernorm for n in LAYERNORM_NODES},
    **{n: install_softmax for n in SOFTMAX_NODES},

    # Nodes the library does not attribute through.
    **{n: install_passthrough for n in PASSTHROUGH_NODES},
    **{n: install_noop for n in ROUTING_NODES + SHAPE_NODES},
}

_missing = set(PRODUCT_NODES + MUL_NODES + ADD_NODES + REDUCTION_NODES + CUMSUM_NODES) - set(EXPLICIT_STRATEGY)
if _missing:
    raise RuntimeError(f"nodes.py names rule-bearing kinds the strategy does not install: {sorted(_missing)}")


# Live registry consulted by :func:`match_installer` and
# :func:`~autolrp.backward.engine.walk`; starts as a copy of :data:`EXPLICIT_STRATEGY`
# and is extended in place by :func:`register_installer`.
INSTALLERS: Dict[str, Installer] = dict(EXPLICIT_STRATEGY)


def match_installer(grad_fn_name: str,
                    strategy: Strategy = INSTALLERS):
    r"""``(pattern, installer)`` of the longest pattern that is a substring of
    ``grad_fn_name``, or ``(None, None)``. The default table is the live
    :data:`INSTALLERS` registry, so :func:`register_installer` entries match.
    """
    best_pattern, best_installer, best_len = None, None, -1
    for pattern, installer in strategy.items():
        # Longest match wins; insertion order breaks length ties.
        if pattern in grad_fn_name and len(pattern) > best_len:
            best_pattern, best_installer, best_len = pattern, installer, len(pattern)
    return best_pattern, best_installer


def is_shape_node(grad_fn_name: str) -> bool:
    r"""Return ``True`` if ``grad_fn_name`` is a shape op: the longest
    :data:`EXPLICIT_STRATEGY` pattern matching it is in
    :data:`~autolrp.nodes.SHAPE_NODES`. Longest-match keeps a routing op
    with a shape-op substring out (``'IndexSelectBackward0'`` resolves to
    ``'IndexSelectBackward'``, not ``'SelectBackward'``).
    """
    pattern, _ = match_installer(grad_fn_name, EXPLICIT_STRATEGY)
    return pattern in SHAPE_NODES


def merge(base: Strategy, *overrides: Strategy) -> Strategy:
    r"""``base`` with ``overrides`` layered on, later arguments winning;
    new keys are appended.
    """
    result = dict(base)
    for o in overrides:
        result.update(o)
    return result


# ---------------------------------------------------------------------------
# Public extension API
# ---------------------------------------------------------------------------

def register_installer(name: str, fn: Optional[Installer] = None) -> Callable:
    r"""Register ``fn`` in :data:`INSTALLERS` for ``grad_fn`` names
    containing ``name``, as a call (``register_installer('GluBackward',
    fn)``) or a decorator (``@register_installer('GluBackward')``). An
    existing entry is overwritten. ``fn(node, config)`` returns a hook
    handle or ``None``. The entry is honored by :func:`match_installer`'s
    default table and by :func:`~autolrp.backward.engine.walk`; :data:`EXPLICIT_STRATEGY`
    stays unchanged.
    """
    def _register(f: Installer) -> Installer:
        INSTALLERS[name] = f
        return f
    return _register if fn is None else _register(fn)
