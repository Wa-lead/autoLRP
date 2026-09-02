r"""Strategy table: ``{pattern: installer}`` matched by longest substring
of ``grad_fn.name()``, so ``'LeakyReluBackward'`` beats
``'ReluBackward'`` whatever the order. This is the internal table;
user config keys are matched exactly (:mod:`autolrp.backward.resolve`).
``None`` or a missing key means the native gradient runs. Extend with
:func:`register_installer` or :func:`merge`.
"""
from typing import Callable, Dict, Optional

from .install import (
    install_addmm, install_mm, install_conv, install_bmm, install_sdpa,
    install_mean_or_sum, install_norm, install_cumsum, install_mul, install_div,
    install_add, install_passthrough, install_noop,
    SOFTMAX_HANDLERS, LAYERNORM_HANDLERS, ACTIVATION_HANDLERS,
)
from .resolve import resolve


Strategy = Dict[str, Optional[Callable]]
Installer = Callable


# ---------------------------------------------------------------------------
# Canonical pattern groups. Each tuple is spliced into EXPLICIT_STRATEGY
# below and is the single source for the derived tables
# (:func:`is_shape_node`, :data:`UNARY_NODES`, :func:`build_strategy`),
# so the listings cannot drift apart.
# ---------------------------------------------------------------------------

_ACTIVATION_PATTERNS = (
    'LeakyReluBackward', 'ReluBackward', 'GeluBackward', 'SiluBackward',
    'TanhBackward', 'SigmoidBackward', 'HardtanhBackward', 'HardswishBackward',
    'HardsigmoidBackward', 'EluBackward', 'SeluBackward', 'CeluBackward',
    'SoftplusBackward', 'SoftsignBackward', 'LogSigmoidBackward', 'MishBackward',
)

_LAYERNORM_PATTERNS = ('NativeLayerNormBackward', 'LayerNormBackward')

_SOFTMAX_PATTERNS = ('LogSoftmaxBackward', 'SoftmaxBackward')

# Shape-routing ops: the autograd-native VJP is the LRP identity, and the
# layer-capture prehooks skip them (:func:`is_shape_node`), since shape
# routing is not a meaningful LRP propagation point.
SHAPE_PATTERNS = (
    'ReshapeAliasBackward', 'ViewBackward', 'ReshapeBackward',
    'TransposeBackward', 'PermuteBackward', 'SqueezeBackward',
    'UnsqueezeBackward', 'ExpandBackward', 'CatBackward', 'StackBackward',
    'SplitBackward', 'SplitWithSizesBackward', 'NarrowBackward',
    'SliceBackward', 'IndexBackward', 'SelectBackward', 'AliasBackward',
    'CloneBackward', 'ToCopyBackward', 'ContiguousBackward',
    'UnbindBackward', 'TBackward', 'AsStridedBackward',
)


# Matched by longest substring; insertion order only breaks exact-length
# ties (see the module docstring).
EXPLICIT_STRATEGY: Strategy = {
    # Linear family.
    'AddmmBackward':          install_addmm,
    'BmmBackward':            install_bmm,
    'ConvolutionBackward':    install_conv,
    'MmBackward':             install_mm,

    # Fused attention nodes, present when decomposition is off.
    'ScaledDotProductEfficientAttention': install_sdpa,
    'ScaledDotProductFlashAttention':     install_sdpa,
    'ScaledDotProductCudnnAttention':     install_sdpa,   # H100+ backend (torch>=2.5)

    # Our add/sub carry the native names, so one key reaches both; a
    # native add or sub saved nothing and keeps its gradient.
    'AddBackward':            install_add,
    'SubBackward':            install_add,
    'MulBackward':            install_mul,
    'DivBackward':            install_div,
    'NegBackward':            install_passthrough,   # -x: relevance unchanged
    'RsubBackward':           install_passthrough,   # c - x: relevance to x unchanged

    # Reductions, same pattern.
    'MeanBackward':           install_mean_or_sum,
    'SumBackward':            install_mean_or_sum,
    'LinalgVectorNormBackward': install_norm,
    'NormBackward':           install_norm,
    'CumsumBackward':         install_cumsum,

    # Selections and routings: the native gradient is the routing.
    'MaxBackward':            install_noop,
    'MinBackward':            install_noop,
    'AmaxBackward':           install_noop,
    'AminBackward':           install_noop,
    'WhereBackward':          install_noop,
    'MaskedFillBackward':     install_noop,
    'UpsampleNearest':        install_noop,
    'UpsampleBilinear':       install_noop,
    'RepeatBackward':         install_noop,
    'FlipBackward':           install_noop,
    'RollBackward':           install_noop,
    'ConstantPadNdBackward':  install_noop,
    'IndexSelectBackward':    install_noop,
    'GatherBackward':         install_noop,
    'SortBackward':           install_noop,
    'TopkBackward':           install_noop,

    # Activations -- passthrough (overrides autograd's ReLU mask).
    **{p: install_passthrough for p in _ACTIVATION_PATTERNS},

    # Normalizations -- passthrough by default
    **{p: install_passthrough for p in _LAYERNORM_PATTERNS},
    'NativeBatchNormBackward':   install_passthrough,
    'CudnnBatchNormBackward':    install_passthrough,
    'BatchNormBackward':         install_passthrough,
    'NativeGroupNormBackward':   install_passthrough,
    'GroupNormBackward':         install_passthrough,
    'InstanceNormBackward':      install_passthrough,

    # Softmax.
    **{p: install_passthrough for p in _SOFTMAX_PATTERNS},

    # Dropout is inert at eval time.
    'NativeDropoutBackward':  install_passthrough,
    'DropoutBackward':        install_passthrough,

    # Elementwise unaries: relevance passes through unchanged.
    'AbsBackward':            install_passthrough,
    'SinBackward':            install_passthrough,
    'CosBackward':            install_passthrough,
    'TanBackward':            install_passthrough,

    # Math unaries: relevance unchanged (the rsqrt gradient would flip
    # its sign at every RMSNorm).
    'PowBackward':            install_passthrough,
    'SqrtBackward':           install_passthrough,
    'RsqrtBackward':          install_passthrough,
    'ExpBackward':            install_passthrough,
    'LogBackward':            install_passthrough,
    'ClampBackward':          install_passthrough,

    # Fused RMSNorm (PyTorch >= 2.4); manual decompositions hit the
    # math-unary entries above instead.
    'RmsNormBackward':        install_passthrough,
    'NativeRmsNormBackward':  install_passthrough,

    # Pooling: native VJP is the standard LRP treatment (avg -> uniform
    # spread; max -> winner-take-all routing).
    'AdaptiveAvgPool':        install_noop,
    'AvgPool':                install_noop,
    'AdaptiveMaxPool':        install_noop,
    'MaxPool':                install_noop,
    # repeat_interleave (GQA head expansion): native VJP sums relevance
    # over the repeated groups -- exactly the LRP fold.
    'RepeatInterleaveBackward': install_noop,

    # Shape-routing ops: autograd-native VJP is the LRP identity.
    **{p: install_noop for p in SHAPE_PATTERNS},
}


# Live registry consulted by :func:`match_installer` and
# :func:`build_strategy`; starts as a copy of :data:`EXPLICIT_STRATEGY`
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
    r"""Return ``True`` if ``grad_fn_name`` is a shape-routing op: the
    longest :data:`EXPLICIT_STRATEGY` pattern matching it is a
    :data:`SHAPE_PATTERNS` entry. Longest-match keeps a routing op with a
    shape-op substring out (``'IndexSelectBackward0'`` resolves to
    ``'IndexSelectBackward'``, not ``'SelectBackward'``).
    """
    pattern, _ = match_installer(grad_fn_name, EXPLICIT_STRATEGY)
    return pattern in SHAPE_PATTERNS


def merge(base: Strategy, *overrides: Strategy) -> Strategy:
    r"""``base`` with ``overrides`` layered on, later arguments winning;
    new keys are appended.
    """
    result = dict(base)
    for o in overrides:
        result.update(o)
    return result


# ---------------------------------------------------------------------------
# Config-driven strategy building
# ---------------------------------------------------------------------------

# Node names each unary config field may use as dict keys, taken from the
# canonical pattern groups at the top of the module.
UNARY_NODES = {
    'softmax':    _SOFTMAX_PATTERNS,
    'layernorm':  _LAYERNORM_PATTERNS,
    'activation': _ACTIVATION_PATTERNS,
}


def _dispatcher(config_attr: str, handlers: Dict[str, Callable]) -> Callable:
    r"""Build a per-node dispatcher closure for one soft-axis config field.

    The returned installer reads ``config.<config_attr>`` at hook time,
    resolves it against the current autograd node, and delegates to the
    matching entry in ``handlers``.
    """
    def install(node, config):
        spec = getattr(config, config_attr)
        name, _ = resolve(spec, node)
        handler = name if callable(name) else handlers[name]
        from . import install as _install
        n = len(_install._TRACE) if _install._TRACE is not None else 0
        h = handler(node, config)
        if _install._TRACE is not None:           # explain(): report the field, not the handler's own line
            del _install._TRACE[n:]
            label = name if isinstance(name, str) else getattr(name, '__name__', 'callable')
            _install._TRACE.append((None, f"{config_attr}={label}"))
        return h
    install.__name__ = f'install_dispatched_{config_attr}'
    return install


def build_strategy(config) -> Strategy:
    r"""The strategy for ``config``: :data:`INSTALLERS` with the softmax,
    layernorm and activation dispatchers overlaid where the entry is still
    :func:`install_passthrough`, so a registered installer is kept.
    """
    s = dict(INSTALLERS)

    def _overlay(pat: str, dispatch: Callable) -> None:
        if s.get(pat) is install_passthrough:
            s[pat] = dispatch

    softmax_dispatch = _dispatcher('softmax', SOFTMAX_HANDLERS)
    for pat in _SOFTMAX_PATTERNS:
        _overlay(pat, softmax_dispatch)

    ln_dispatch = _dispatcher('layernorm', LAYERNORM_HANDLERS)
    for pat in _LAYERNORM_PATTERNS:
        _overlay(pat, ln_dispatch)

    act_dispatch = _dispatcher('activation', ACTIVATION_HANDLERS)
    for pat in _ACTIVATION_PATTERNS:
        _overlay(pat, act_dispatch)

    return s


# ---------------------------------------------------------------------------
# Public extension API
# ---------------------------------------------------------------------------

def register_installer(name: str, fn: Optional[Installer] = None) -> Callable:
    r"""Register ``fn`` in :data:`INSTALLERS` for ``grad_fn`` names
    containing ``name``, as a call (``register_installer('GluBackward',
    fn)``) or a decorator (``@register_installer('GluBackward')``). An
    existing entry is overwritten. ``fn(node, config)`` returns a hook
    handle or ``None``. The entry is honored by :func:`match_installer`'s
    default table and by :func:`build_strategy`; :data:`EXPLICIT_STRATEGY`
    stays unchanged.
    """
    if fn is None:
        return installer(name)
    INSTALLERS[name] = fn
    return fn


def installer(name: str) -> Callable[[Installer], Installer]:
    r"""Decorator form of :func:`register_installer`.

    ``@installer('MyBackward')`` above a function definition registers
    that function and returns it unchanged.
    """
    def _decorator(fn: Installer) -> Installer:
        register_installer(name, fn)
        return fn
    return _decorator
