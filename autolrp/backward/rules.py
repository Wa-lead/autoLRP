r"""LRP rules: every function that turns a node's output relevance into
its input relevance, one table per kind of node.

Two-operand nodes, ``rule(a, b, R_out, eps, fwd, bwd_a, bwd_b, *,
attribute, **params) -> (R_a, R_b)``, ``None`` for an operand the rule
does not attribute. ``fwd(a, b)`` is the op without bias, ``bwd_a(b, s)``
its VJP into ``a`` for an output-shaped ``s``, ``bwd_b(a, s)`` its VJP
into ``b``. Product rules use the kernels, ``s = R / z`` then
``a * bwd_a(b, s)``; split rules (mul, div, add, sub) work on operands
that share ``z``'s shape and are called without kernels. ``fwd`` arrives memoized
from the hook, so calling ``fwd(a, b)`` for ``z`` costs nothing more.

One-operand nodes, ``rule(x, y, R_out, eps, **saved) -> R_in``. ``x`` is
the node's input, ``y`` its output (``None`` when the node did not keep
it), ``saved`` the rest of what the node saved: ``dim`` for a softmax,
``normalized_shape``, ``weight``, ``bias``, ``mean``, ``rstd`` for a
layer norm, ``dim`` and ``keepdim`` for a reduction.

``attribute`` is ``'lhs'`` (the first operand, the second its weight),
``'rhs'``, or ``'both'`` (each with half the relevance, the Euler budget
of a bilinear product); a split rule attributing one side gives it the
whole of ``R``. Each rule's signature carries its own default, ``'lhs'``
for the product rules, ``'both'`` for the split rules. The entry may set
it; the node's facts may (:mod:`autolrp.backward.resolve`); otherwise the
rule's default stands.

One route from a node to its rule: the node's name without its version
digit selects a :class:`RuleTable` in :data:`RULES_FOR`; the config's
``rule`` entry for that node names one function in the table. Every
node that runs a rule is a key of :data:`RULES_FOR`, and the keys are
exactly the node names a config may use.
"""
import inspect
import warnings
from typing import Callable, Dict

import torch
import torch.nn.functional as F

from .lrp_utils import stabilize, apply_bias_split
from ..nodes import (PRODUCT_NODES, MUL_NODES, ADD_NODES, SOFTMAX_NODES,
                     LAYERNORM_NODES, REDUCTION_NODES, CUMSUM_NODES, ELEMENTWISE_NODES)


class RuleTable(dict):
    """``{rule name: function}`` for one kind of node. ``default`` is the
    ``BASE`` entry for its nodes, and what a ``detach`` entry runs on the
    kept operand. ``two_operand`` is read off the rules: they take
    ``attribute``."""

    def __init__(self, default: str, entries: Dict[str, Callable]):
        super().__init__(entries)
        if default not in entries:
            raise ValueError(f"table default {default!r} is not in the table")
        self.default = default
        self.two_operand = all('attribute' in inspect.signature(f).parameters for f in entries.values())


_GAMMA_DEGENERATION_WARNED: set = set()


def _warn_gamma_degeneration(gamma_val, fn_name):
    # Once per (value, rule): below the cutoff the epsilon rule runs instead.
    key = (float(gamma_val), fn_name)
    if key in _GAMMA_DEGENERATION_WARNED:
        return
    _GAMMA_DEGENERATION_WARNED.add(key)
    warnings.warn(
        f"autoLRP: {fn_name}(gamma={gamma_val}) runs the epsilon rule "
        f"instead: gamma <= 0.01 perturbs the clamped weights by at most "
        f"1% -- a small but nonzero effect this cutoff drops. Set "
        f"gamma > 0.01 for the gamma rule, or name 'epsilon' in the "
        f"entry to make the intent explicit.", UserWarning, stacklevel=4)


# ---------------------------------------------------------------------------
# Product rules: addmm, mm, bmm, convolution
# ---------------------------------------------------------------------------

def _sides(attribute, R_out):
    r"""Which operands a product rule attributes, and the relevance each
    gets: ``'lhs'`` the first with all of it, ``'rhs'`` the second,
    ``'both'`` both with half each (the Euler budget of a bilinear
    product). Returns ``(attribute a, attribute b, R)``."""
    if attribute == 'lhs':
        return True, False, R_out
    if attribute == 'rhs':
        return False, True, R_out
    if attribute == 'both':
        return True, True, R_out * 0.5
    raise ValueError(f"attribute must be 'lhs', 'rhs' or 'both', got {attribute!r}")


def epsilon(a, b, R_out, eps, fwd, bwd_a, bwd_b, *, attribute='lhs'):
    r"""LRP-epsilon (Bach et al. 2015, 2): ``R_a = a * bwd_a(b, R / z)``,
    and the same for ``b`` with ``a`` as its weight."""
    do_a, do_b, R = _sides(attribute, R_out)
    z = fwd(a, b)
    with torch.no_grad():
        s = R / stabilize(z, eps)
        return (a * bwd_a(b, s) if do_a else None,
                b * bwd_b(a, s) if do_b else None)


def zplus(a, b, R_out, eps, fwd, bwd_a, bwd_b, *, attribute='lhs'):
    r"""LRP-z+ (Bach et al. 2015): positive contributions only, the
    alpha-beta rule with ``alpha = 1``, ``beta = 0``."""
    do_a, do_b, R = _sides(attribute, R_out)
    with torch.no_grad():
        a_pos, a_neg = a.clamp(min=0), a.clamp(max=0)
        b_pos, b_neg = b.clamp(min=0), b.clamp(max=0)
        z_pos = fwd(a_pos, b_pos) + fwd(a_neg, b_neg)
        s_pos = R / stabilize(z_pos, eps)
        return ((a_pos * bwd_a(b_pos, s_pos) + a_neg * bwd_a(b_neg, s_pos)) if do_a else None,
                (b_pos * bwd_b(a_pos, s_pos) + b_neg * bwd_b(a_neg, s_pos)) if do_b else None)


def alpha_beta(a, b, R_out, eps, fwd, bwd_a, bwd_b, *, alpha, beta, attribute='lhs'):
    r"""LRP-alpha-beta (Bach et al. 2015, 2.2): positive contributions
    scaled by ``alpha``, negative by ``-beta``, with ``alpha - beta = 1``
    (``ValueError`` otherwise)."""
    if abs((alpha - beta) - 1.0) > 1e-6:
        raise ValueError(
            f"alpha_beta requires alpha - beta == 1, got "
            f"alpha={alpha}, beta={beta} (diff={alpha - beta})")
    do_a, do_b, R = _sides(attribute, R_out)
    a_pos, a_neg = a.clamp(min=0), a.clamp(max=0)
    b_pos, b_neg = b.clamp(min=0), b.clamp(max=0)
    z_pos = fwd(a_pos, b_pos) + fwd(a_neg, b_neg)
    s_pos = alpha * R / stabilize(z_pos, eps)
    R_a = (a_pos * bwd_a(b_pos, s_pos) + a_neg * bwd_a(b_neg, s_pos)) if do_a else None
    R_b = (b_pos * bwd_b(a_pos, s_pos) + b_neg * bwd_b(a_neg, s_pos)) if do_b else None
    if beta != 0:
        z_neg = fwd(a_pos, b_neg) + fwd(a_neg, b_pos)
        s_neg = -beta * R / stabilize(z_neg, eps)
        if do_a:
            R_a = R_a + (a_pos * bwd_a(b_neg, s_neg) + a_neg * bwd_a(b_pos, s_neg))
        if do_b:
            R_b = R_b + (b_pos * bwd_b(a_neg, s_neg) + b_neg * bwd_b(a_pos, s_neg))
    return R_a, R_b


def gamma(a, b, R_out, eps, fwd, bwd_a, bwd_b, *, gamma, attribute='lhs'):
    r"""LRP-gamma, sign-symmetric: the attributed operand is split by sign
    and the other, its weight, has its same-sign paths amplified by
    ``gamma``, so mixed-sign inputs are handled; equals
    :func:`gamma_montavon` when the attributed operand is ``>= 0``.
    Degenerates to :func:`epsilon` for ``gamma <= 0.01`` (warns once)."""
    if gamma <= 0.01:
        _warn_gamma_degeneration(gamma, 'gamma')
        return epsilon(a, b, R_out, eps, fwd, bwd_a, bwd_b, attribute=attribute)
    do_a, do_b, R = _sides(attribute, R_out)
    z = fwd(a, b)
    with torch.no_grad():
        R_a = R_b = None
        if do_a:
            x_pos, x_neg = a.clamp(min=0), a.clamp(max=0)
            w_pos, w_neg = b + gamma * b.clamp(min=0), b + gamma * b.clamp(max=0)
            z_pp = fwd(x_pos, w_pos) + fwd(x_neg, w_neg)
            z_pn = fwd(x_pos, w_neg) + fwd(x_neg, w_pos)
            s_pos = (z > 0).to(z.dtype) * R / stabilize(z_pp, eps)
            s_neg = (z < 0).to(z.dtype) * R / stabilize(z_pn, eps)
            R_a = (x_pos * (bwd_a(w_pos, s_pos) + bwd_a(w_neg, s_neg))
                   + x_neg * (bwd_a(w_neg, s_pos) + bwd_a(w_pos, s_neg)))
        if do_b:
            x_pos, x_neg = b.clamp(min=0), b.clamp(max=0)
            w_pos, w_neg = a + gamma * a.clamp(min=0), a + gamma * a.clamp(max=0)
            z_pp = fwd(w_pos, x_pos) + fwd(w_neg, x_neg)
            z_pn = fwd(w_neg, x_pos) + fwd(w_pos, x_neg)
            s_pos = (z > 0).to(z.dtype) * R / stabilize(z_pp, eps)
            s_neg = (z < 0).to(z.dtype) * R / stabilize(z_pn, eps)
            R_b = (x_pos * (bwd_b(w_pos, s_pos) + bwd_b(w_neg, s_neg))
                   + x_neg * (bwd_b(w_neg, s_pos) + bwd_b(w_pos, s_neg)))
        return R_a, R_b


def gamma_montavon(a, b, R_out, eps, fwd, bwd_a, bwd_b, *, gamma, attribute='lhs'):
    r"""LRP-gamma as in Montavon et al. 2019 (10.2.3) and zennit:
    ``w' = w + gamma * w+``, then the epsilon rule with ``w'``. Assumes
    the attributed operand ``>= 0``; on mixed-sign inputs it amplifies
    positive-weight paths whatever the contribution sign, see
    :func:`gamma`."""
    if gamma <= 0.01:
        _warn_gamma_degeneration(gamma, 'gamma_montavon')
        return epsilon(a, b, R_out, eps, fwd, bwd_a, bwd_b, attribute=attribute)
    do_a, do_b, R = _sides(attribute, R_out)
    with torch.no_grad():
        R_a = R_b = None
        if do_a:
            w = b + gamma * b.clamp(min=0)
            R_a = a * bwd_a(w, R / stabilize(fwd(a, w), eps))
        if do_b:
            w = a + gamma * a.clamp(min=0)
            R_b = b * bwd_b(w, R / stabilize(fwd(w, b), eps))
        return R_a, R_b


def zbox(a, b, R_out, eps, fwd, bwd_a, bwd_b, *, low, high, attribute='lhs'):
    r"""z-box rule for a bounded input domain ``[low, high]`` (Montavon
    et al. 2017, 3.1), meant for the input layer, e.g. normalized
    pixels; the bounds are the first operand's, so it attributes that
    one only."""
    if attribute != 'lhs':
        raise ValueError("zbox bounds the first operand: attribute must be 'lhs'")
    z = fwd(a, b)
    with torch.no_grad():
        L = torch.full_like(a, low)
        H = torch.full_like(a, high)
        b_pos, b_neg = b.clamp(min=0), b.clamp(max=0)
        s = R_out / stabilize(z - fwd(L, b_pos) - fwd(H, b_neg), eps)
        return (a * bwd_a(b, s) - L * bwd_a(b_pos, s) - H * bwd_a(b_neg, s)), None


def gradient_input(a, b, R_out, eps, fwd, bwd_a, bwd_b, *, attribute='lhs'):
    r"""Gradient times input, ``a * bwd_a(b, R)``: no ``z`` in the
    denominator, so it does not conserve. Attributed ``'both'`` ways on a
    bilinear product it is LXT's uniform attention rule."""
    do_a, do_b, R = _sides(attribute, R_out)
    return (a * bwd_a(b, R) if do_a else None,
            b * bwd_b(a, R) if do_b else None)


# ---------------------------------------------------------------------------
# Split rules: mul, div, add, sub (operands share z's shape). Attributing
# one side gives it the whole of R; 'both' splits.
# ---------------------------------------------------------------------------

def _split(attribute, R_out, both, a, b):
    """``(R_a, R_b)``: one attributed side takes all of ``R_out``; both
    take the split ``both()`` computes, which needs both operands saved."""
    if attribute == 'lhs':
        return R_out, None
    if attribute == 'rhs':
        return None, R_out
    if attribute == 'both':
        if a is None or b is None:
            raise ValueError("attribute='both' on an operand the node did not save: "
                             "the other operand is a constant or a frozen tensor")
        return both()
    raise ValueError(f"attribute must be 'lhs', 'rhs' or 'both', got {attribute!r}")


def proportional(a, b, R_out, eps, *_, attribute='both'):
    r"""Split ``R_out`` between two operands in proportion to their
    magnitudes."""
    def both():
        with torch.no_grad():
            aa, bb = a.abs(), b.abs()
            d = aa + bb + eps
            return (aa / d) * R_out, (bb / d) * R_out
    return _split(attribute, R_out, both, a, b)


def equal(a, b, R_out, eps, *_, attribute='both'):
    r"""Half of ``R_out`` to each operand."""
    return _split(attribute, R_out, lambda: (R_out * 0.5, R_out * 0.5), a, b)


def fixed(a, b, R_out, eps, *_, p, attribute='both'):
    r"""``p * R`` to the first operand, ``(1 - p) * R`` to the second."""
    return _split(attribute, R_out, lambda: (p * R_out, (1.0 - p) * R_out), a, b)


# ---------------------------------------------------------------------------
# One-operand nodes: rule(x, y, R_out, eps, **saved) -> R_in
# ---------------------------------------------------------------------------

def passthrough(x, y, R_out, eps, **_):
    r"""``R_in = R_out``: the node is transparent to relevance."""
    return R_out


def elementwise_yx(x, y, R_out, eps, **_):
    r"""``y/x`` rule for an elementwise nonlinearity (Achtibat et al. 2024,
    Prop. 3.2): ``R_in = R_out * y / x``. Stated there for any elementwise
    nonlinearity, so it applies to ``exp`` and ``sqrt`` as to ``gelu``."""
    return R_out * y / stabilize(x, eps)


def softmax_jacobian(x, y, R_out, eps, *, dim=-1, **_):
    r"""Softmax rule of Achtibat et al. 2024 (AttnLRP, Prop. 3.1):
    ``R_in = x * (R_out - y * sum_dim(R_out))``, the Taylor form at the
    input point, not the plain VJP. Row sums are not conserved; the
    hidden bias keeps part of the relevance."""
    return x * (R_out - y * R_out.sum(dim=dim, keepdim=True))


def softmax_gate(x, y, R_out, eps, **_):
    r"""The softmax output as a constant gate: ``R_in = y * R_out``. Not
    conservative; sharpens attention maps in some recipes."""
    return y * R_out


def _normalized(x, normalized_shape, eps, mean=None, rstd=None):
    """``(x - mean) / std`` over the normalized dims, from the node's
    saved ``mean``/``rstd`` when present."""
    if mean is not None and rstd is not None:
        m = mean.reshape(mean.shape + (1,) * (x.ndim - mean.ndim))
        r = rstd.reshape(rstd.shape + (1,) * (x.ndim - rstd.ndim))
        return (x - m) * r
    dims = tuple(range(-len(normalized_shape), 0))
    xc = x - x.mean(dim=dims, keepdim=True)
    return xc / torch.sqrt(xc.pow(2).mean(dim=dims, keepdim=True) + eps)


def layernorm_identity(x, y, R_out, eps, *, normalized_shape, weight=None,
                       bias=None, mean=None, rstd=None, **_):
    r"""Layer norm with mean, std and the affine weight treated as
    statistics (Achtibat et al. 2024, Prop. 3.4): relevance passes
    unchanged except for the bias, which takes its magnitude share as the
    decomposed graph's add split would. Identity when ``bias`` is ``None``."""
    if bias is None:
        return R_out
    with torch.no_grad():
        xn = _normalized(x, normalized_shape, eps, mean, rstd)
        y_nb = xn * weight if weight is not None else xn
        return apply_bias_split(R_out, y_nb, bias, eps)


def layernorm_yx(x, y, R_out, eps, *, normalized_shape, weight=None,
                 bias=None, **_):
    r"""Layer norm ``y/x`` rule (Ali et al. 2022, 4):
    ``R_in = R_out * LN(x) / x``."""
    with torch.no_grad():
        y = F.layer_norm(x, normalized_shape, weight, bias)
    return R_out * y / stabilize(x, eps)


def layernorm_detach_std(x, y, R_out, eps, *, normalized_shape, weight=None,
                         bias=None, mean=None, rstd=None, **_):
    r"""Layer norm with only the standard deviation held constant: what the
    decomposed graph gives when the division by ``std`` is detached but the
    centering is not. The bias takes its share, the weight passes through,
    ``x - mean(x)`` splits proportionally, and the mean's share returns over
    ``x`` by the reduction rule. Conserves. (LXT detaches the std the same
    way and propagates through the centering with its gradient rule.)"""
    with torch.no_grad():
        dims = tuple(range(-len(normalized_shape), 0))
        xn = _normalized(x, normalized_shape, eps, mean, rstd)
        y_nb = xn * weight if weight is not None else xn
        R = apply_bias_split(R_out, y_nb, bias, eps)            # since z = y_nb + bias we first remove bias share, not here it is the R share or xn
        m = x.mean(dim=dims, keepdim=True).expand_as(x)          # xn = x - mean(x), so we first need compute the relevance for both brnahces and move the mean() relevance to x
        R_x, R_m = proportional(x, m, R, eps)                    # x - mean(x): a two-operand sub
        R_via_mean = reduction_proportional(                     # mean(x): its share back over x
            x, None, R_m.sum(dim=dims, keepdim=True), eps, dim=dims, keepdim=True)
        return R_x + R_via_mean


def reduction_proportional(x, y, R_out, eps, *, dim=None, keepdim=False, **_):
    r"""A reduction (mean, sum, norm): each reduced element takes the
    share ``|x_i| / sum |x|`` of the output it was reduced into. ``dim``
    as the op received it: ``None`` or an empty tuple for a full
    reduction, an int, or a tuple of ints."""
    if isinstance(dim, int):
        dim = (dim,)
    elif dim is not None and len(dim) == 0:
        dim = None
    with torch.no_grad():
        ax = x.abs()
        if dim is None:
            return ax / (ax.sum() + eps) * R_out
        # a native node stores a negative dim as an unsigned 64-bit value
        dims = tuple((d - (1 << 64) if d >= (1 << 63) else d) % x.ndim for d in dim)
        ratios = ax / (ax.sum(dim=dims, keepdim=True) + eps)
        if not keepdim:
            shape = list(x.shape)
            for d in dims:
                shape[d] = 1
            R_out = R_out.reshape(shape)
        return ratios * R_out


# ---------------------------------------------------------------------------
# Rule tables
# ---------------------------------------------------------------------------

def cumsum_epsilon(x, y, R_out, eps, *, dim, **_):
    r"""``cumsum``: a linear map with 0/1 weights, ``y_j = sum_{i<=j} x_i``.
    The epsilon rule gives ``R_i = x_i * sum_{j>=i} R_j / y_j``, a reversed
    cumulative sum of ``R / y``; the native gradient would hand every
    ``x_i`` the full ``R_j`` of each later output."""
    if y is None:
        y = torch.cumsum(x, dim)
    s = R_out / stabilize(y, eps)
    return x * torch.flip(torch.cumsum(torch.flip(s, (dim,)), dim), (dim,))


# Every product kind (addmm, mm, bmm, convolution). A rule attributes an
# operand with the other as its weight; ``attribute`` says which, or
# both with half the relevance each.
PRODUCT_RULES = RuleTable('epsilon', {
    'epsilon':          epsilon,
    'zplus':            zplus,
    'gamma':            gamma,
    'gamma_montavon':   gamma_montavon,
    'alpha_beta':       alpha_beta,
    'zbox':             zbox,
    'gradient_input':   gradient_input,
})

MUL_RULES = RuleTable('proportional', {
    'proportional': proportional,
})

ADD_RULES = RuleTable('proportional', {
    'proportional': proportional,
    'equal':        equal,
    'fixed':        fixed,
})

SOFTMAX_RULES = RuleTable('passthrough', {
    'passthrough': passthrough,
    'jacobian':    softmax_jacobian,
    'gate':        softmax_gate,
})

LAYERNORM_RULES = RuleTable('identity', {
    'identity':    layernorm_identity,
    'passthrough': passthrough,
    'yx':          layernorm_yx,
    'detach_std':  layernorm_detach_std,
})

ELEMENTWISE_RULES = RuleTable('passthrough', {
    'passthrough': passthrough,
    'yx':          elementwise_yx,
})

REDUCTION_RULES = RuleTable('proportional', {
    'proportional': reduction_proportional,
})

CUMSUM_RULES = RuleTable('epsilon', {
    'epsilon': cumsum_epsilon,
})

# The one virtual name: ``('detach', {'by': fact})`` on a two-operand
# table runs the table's default (``rule=`` to choose another) attributing
# the operand the fact does not name; on a node without the fact, the
# default with the graph's own attribute.
VIRTUAL_RULES = frozenset({'detach'})

# Node name (no version digit) to its rule table: the one map behind
# resolution, validation and ``BASE``. The names live in ``autolrp.nodes``.
RULES_FOR: Dict[str, RuleTable] = {
    **{name: PRODUCT_RULES for name in PRODUCT_NODES},
    **{name: MUL_RULES for name in MUL_NODES},
    **{name: ADD_RULES for name in ADD_NODES},
    **{name: SOFTMAX_RULES for name in SOFTMAX_NODES},
    **{name: LAYERNORM_RULES for name in LAYERNORM_NODES},
    **{name: ELEMENTWISE_RULES for name in ELEMENTWISE_NODES},
    **{name: REDUCTION_RULES for name in REDUCTION_NODES},
    **{name: CUMSUM_RULES for name in CUMSUM_NODES},
}
