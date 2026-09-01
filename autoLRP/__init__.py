r"""autoLRP: layer-wise relevance propagation on the autograd graph.

Wrap the input, run the model, pick the output scalar, call ``.lrp()``::

    import autoLRP
    from autoLRP import LRPConfig, BASE

    x = autoLRP.tensor(image)
    out = model(x)
    out[0, pred].lrp()                     # BASE: epsilon on linear
    heatmap = x.relevance                  # families, proportional elsewhere

Every rule-bearing node in the graph is addressed by its name without
the version digit, or by a fact an analyzer attached to it. ``BASE`` is
the printed starting table; override entries on it::

    out[0, pred].lrp(config=LRPConfig(rule={**BASE, 'BmmBackward': 'uniform'}))
    out[0, pred].lrp(config=LRPConfig(attn='attnlrp'))

Anything that is not a node name or a registered fact, a rule the
node's family cannot run, or a node no entry addresses, is an error.
"""
__version__ = '0.1.0'

from .compat import run_selfcheck
run_selfcheck()

from .config import LRPConfig, BASE
from .tensor import LRPTensor, tensor
from .forward.intercept import (
    register_rewrite, REWRITES,
    set_decompose_attention, get_decompose_attention, decompose_attention,
)
from .backward.strategies import (
    register_installer, installer, merge, match_installer, is_shape_node,
    EXPLICIT_STRATEGY, INSTALLERS,
)
from .backward.analysis import (
    register_analyzer, ANALYZERS, node_facts,
)
from .backward.engine import graph_lrp, walk, plan_report, explain, explain_summary
from .recipes import bilrp, clrp
from . import eval

__all__ = [
    'LRPConfig', 'BASE', 'LRPTensor', 'tensor',
    'register_rewrite', 'REWRITES',
    'set_decompose_attention', 'get_decompose_attention',
    'decompose_attention',
    'register_installer', 'installer', 'merge', 'match_installer',
    'is_shape_node', 'EXPLICIT_STRATEGY', 'INSTALLERS',
    'register_analyzer', 'ANALYZERS', 'node_facts',
    'graph_lrp', 'walk', 'plan_report', 'explain', 'explain_summary',
    'bilrp', 'clrp', 'eval',
]
