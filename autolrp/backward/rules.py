r"""LRP rules: every function that turns a node's output relevance into
its input relevance, one table per kind of node.

Two-operand nodes, ``rule(a, b, R_out, eps, fwd, bwd_a, bwd_b, **params)
-> (R_a, R_b)``. ``fwd(a, b)`` is the op without bias, ``bwd_a(b, s)``
its VJP into ``a`` for an output-shaped ``s``, ``bwd_b(a, s)`` its VJP
into ``b``; the linear family names them ``x``, ``w``, ``bwd_x``,
``bwd_w``. Contribution rules (linear, bilinear) use the kernels,
``s = R / z`` then ``a * bwd_a(b, s)``; split rules (mul, div, add, sub)
work on operands that share ``z``'s shape and ignore the kernels. The
linear rules return ``(R_x, None)``: the weight slot keeps its native
gradient. ``fwd`` arrives memoized from ``run_linear_rule``, so calling
``fwd(x, w)`` for ``z`` costs nothing more.

One-operand nodes, ``rule(x, y, R_out, eps, **saved) -> R_in``. ``x`` is
the node's input, ``y`` its output (``None`` when the node did not keep
it), ``saved`` the rest of what the node saved: ``dim`` for a softmax,
``normalized_shape``, ``weight``, ``bias``, ``mean``, ``rstd`` for a
layer norm, ``dim`` and ``keepdim`` for a reduction.

Names are positional: ``detach_lhs`` zeros the operand written on the
left of that op, always. Which operand a config means is decided
before the call, by the ``('detach', {'by': <fact>})`` entry.
"""
import warnings
from typing import Callable, Dict

import torch
import torch.nn.functional as F

from .lrp_utils import stabilize, apply_bias_split


class Family(dict):
    """A rule table plus the one name it falls back to when a
    ``('detach', {'by': <fact>})`` entry addresses a node that does not
    carry that fact."""

    def __init__(self, default: str, entries: Dict[str, Callable]):
        super().__init__(entries)
        if default not in entries:
            raise ValueError(f"family default {default!r} is not in the table")
        self.default = default


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
# Linear family: addmm, mm, convolution, and any product with one
# operand from the input
# ---------------------------------------------------------------------------

def epsilon(x, w, R_out, eps, fwd, bwd_x, bwd_w):
    r"""LRP-epsilon (Bach et al. 2015, 2): ``R_in = x * bwd_x(w, R / z)``."""
    z = fwd(x, w)
    with torch.no_grad():
        return x * bwd_x(w, R_out / stabilize(z, eps)), None


def zplus(x, w, R_out, eps, fwd, bwd_x, bwd_w):
    r"""LRP-z+ (Bach et al. 2015): positive contributions only, the
    alpha-beta rule with ``alpha = 1``, ``beta = 0``."""
    with torch.no_grad():
        x_pos, x_neg = x.clamp(min=0), x.clamp(max=0)
        w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
        z_pos = fwd(x_pos, w_pos) + fwd(x_neg, w_neg)
        s_pos = R_out / stabilize(z_pos, eps)
        return (x_pos * bwd_x(w_pos, s_pos)
                + x_neg * bwd_x(w_neg, s_pos)), None


def alpha_beta(x, w, R_out, eps, fwd, bwd_x, bwd_w, *, alpha, beta):
    r"""LRP-alpha-beta (Bach et al. 2015, 2.2): positive contributions
    scaled by ``alpha``, negative by ``-beta``, with ``alpha - beta = 1``
    (``ValueError`` otherwise)."""
    if abs((alpha - beta) - 1.0) > 1e-6:
        raise ValueError(
            f"alpha_beta requires alpha - beta == 1, got "
            f"alpha={alpha}, beta={beta} (diff={alpha - beta})")
    x_pos, x_neg = x.clamp(min=0), x.clamp(max=0)
    w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
    z_pos = fwd(x_pos, w_pos) + fwd(x_neg, w_neg)
    s_pos = alpha * R_out / stabilize(z_pos, eps)
    R_in = (x_pos * bwd_x(w_pos, s_pos)
            + x_neg * bwd_x(w_neg, s_pos))
    if beta != 0:
        z_neg = fwd(x_pos, w_neg) + fwd(x_neg, w_pos)
        s_neg = -beta * R_out / stabilize(z_neg, eps)
        R_in = R_in + (x_pos * bwd_x(w_neg, s_neg)
                       + x_neg * bwd_x(w_pos, s_neg))
    return R_in, None


def gamma(x, w, R_out, eps, fwd, bwd_x, bwd_w, *, gamma):
    r"""LRP-gamma, sign-symmetric: both ``x`` and ``w`` are split by sign
    and same-sign paths are amplified by ``gamma``, so mixed-sign inputs
    are handled; equals :func:`gamma_montavon` when ``x >= 0``.
    Degenerates to :func:`epsilon` for ``gamma <= 0.01`` (warns once)."""
    if gamma <= 0.01:
        _warn_gamma_degeneration(gamma, 'gamma')
        return epsilon(x, w, R_out, eps, fwd, bwd_x, bwd_w)
    z = fwd(x, w)
    with torch.no_grad():
        x_pos, x_neg = x.clamp(min=0), x.clamp(max=0)
        w_pos = w + gamma * w.clamp(min=0)
        w_neg = w + gamma * w.clamp(max=0)
        z_pp = fwd(x_pos, w_pos) + fwd(x_neg, w_neg)
        z_pn = fwd(x_pos, w_neg) + fwd(x_neg, w_pos)
        s_pos = (z > 0).to(z.dtype) * R_out / stabilize(z_pp, eps)
        s_neg = (z < 0).to(z.dtype) * R_out / stabilize(z_pn, eps)
        return (x_pos * (bwd_x(w_pos, s_pos) + bwd_x(w_neg, s_neg))
                + x_neg * (bwd_x(w_neg, s_pos) + bwd_x(w_pos, s_neg))), None


def gamma_montavon(x, w, R_out, eps, fwd, bwd_x, bwd_w, *, gamma):
    r"""LRP-gamma as in Montavon et al. 2019 (10.2.3) and zennit:
    ``w' = w + gamma * w+``, then the epsilon rule with ``w'``. Assumes
    ``x >= 0``; on mixed-sign inputs it amplifies positive-weight paths
    whatever the contribution sign, see :func:`gamma`."""
    if gamma <= 0.01:
        _warn_gamma_degeneration(gamma, 'gamma_montavon')
        return epsilon(x, w, R_out, eps, fwd, bwd_x, bwd_w)
    with torch.no_grad():
        w_prime = w + gamma * w.clamp(min=0)
        s = R_out / stabilize(fwd(x, w_prime), eps)
        return x * bwd_x(w_prime, s), None


def zbox(x, w, R_out, eps, fwd, bwd_x, bwd_w, *, low, high):
    r"""z-box rule for a bounded input domain ``[low, high]`` (Montavon
    et al. 2017, 3.1), meant for the input layer, e.g. normalized
    pixels."""
    z = fwd(x, w)
    with torch.no_grad():
        L = torch.full_like(x, low)
        H = torch.full_like(x, high)
        w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
        s = R_out / stabilize(z - fwd(L, w_pos) - fwd(H, w_neg), eps)
        return (x * bwd_x(w, s)
                - L * bwd_x(w_pos, s)
                - H * bwd_x(w_neg, s)), None


# ---------------------------------------------------------------------------
# Bilinear family: a product with both operands from the input
# ---------------------------------------------------------------------------

def epsilon_bmm(a, b, R_out, eps, fwd, bwd_a, bwd_b):
    r"""Both operands receive the epsilon share, out of a ``2z`` budget
    (bilinear epsilon, AttnLRP Prop. 3.3)."""
    s = R_out / stabilize(2.0 * fwd(a, b), eps)
    return a * bwd_a(b, s), b * bwd_b(a, s)


def uniform_bmm(a, b, R_out, eps, fwd, bwd_a, bwd_b):
    r"""Gradient times input, halved, on each operand (LXT uniform)."""
    return a * bwd_a(b, R_out) / 2.0, b * bwd_b(a, R_out) / 2.0


def detach_lhs_bmm(a, b, R_out, eps, fwd, bwd_a, bwd_b):
    r"""Zeros for the first operand; the second takes the full epsilon
    share."""
    s = R_out / stabilize(fwd(a, b), eps)
    return torch.zeros_like(a), b * bwd_b(a, s)


def detach_rhs_bmm(a, b, R_out, eps, fwd, bwd_a, bwd_b):
    r"""Zeros for the second operand; the first takes the full epsilon
    share."""
    s = R_out / stabilize(fwd(a, b), eps)
    return a * bwd_a(b, s), torch.zeros_like(b)


# ---------------------------------------------------------------------------
# Split rules: mul, div, add, sub (operands share z's shape)
# ---------------------------------------------------------------------------

def proportional(a, b, R_out, eps, *_):
    r"""Split by magnitude, ``|a| / (|a| + |b|)`` to the first operand."""
    aa, bb = a.abs(), b.abs()
    d = aa + bb + eps
    return (aa / d) * R_out, (bb / d) * R_out


def detach_lhs(a, b, R_out, eps, *_):
    r"""Zeros to the first operand, all relevance to the second."""
    return torch.zeros_like(R_out), R_out


def detach_rhs(a, b, R_out, eps, *_):
    r"""All relevance to the first operand, zeros to the second."""
    return R_out, torch.zeros_like(R_out)


def equal(a, b, R_out, eps, *_):
    r"""``R / 2`` to each operand (Otsuki et al. 2024; for ResNets)."""
    return 0.5 * R_out, 0.5 * R_out


def fixed(a, b, R_out, eps, *_, p):
    r"""``p * R`` to the first operand, ``(1 - p) * R`` to the second."""
    return p * R_out, (1.0 - p) * R_out


# ---------------------------------------------------------------------------
# One-operand nodes: rule(x, y, R_out, eps, **saved) -> R_in
# ---------------------------------------------------------------------------

def passthrough(x, y, R_out, eps, **_):
    r"""``R_in = R_out``: the node is transparent to relevance."""
    return R_out


def activation_yx(x, y, R_out, eps, **_):
    r"""``y/x`` rule for a nonlinearity (Achtibat et al. 2024, Prop. 3.2):
    ``R_in = R_out * y / x``."""
    return R_out * y / stabilize(x, eps)


def softmax_jacobian(x, y, R_out, eps, *, dim=-1, **_):
    r"""Softmax rule of Achtibat et al. 2024 (AttnLRP, Prop. 3.1):
    ``R_in = x * (R_out - y * sum_dim(R_out))``, the Taylor form at the
    input point, not the plain VJP. Row sums are not conserved; the
    hidden bias keeps part of the relevance."""
    return x * (R_out - y * R_out.sum(dim=dim, keepdim=True))


def softmax_detach(x, y, R_out, eps, **_):
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
        R = apply_bias_split(R_out, y_nb, bias, eps)            # + bias: its share leaves
        m = x.mean(dim=dims, keepdim=True).expand_as(x)          # * weight and / std: pass through
        R_x, R_m = proportional(x, m, R, eps)                    # x - mean(x): a two-operand sub
        R_via_mean = reduction_proportional(                     # mean(x): its share back over x
            x, None, R_m.sum(dim=dims, keepdim=True), eps, dim=dims, keepdim=True)
        return R_x + R_via_mean


def reduction_proportional(x, y, R_out, eps, *, dim=None, keepdim=False, **_):
    r"""A reduction (mean, sum, norm): each reduced element takes the
    share ``|x_i| / sum |x|`` of the output it was reduced into. ``dim``
    is ``None`` for a full reduction or a tuple of ints."""
    with torch.no_grad():
        ax = x.abs()
        if dim is None:
            return ax / (ax.sum() + eps) * R_out
        dims = tuple(d if d < x.ndim else d - (1 << 64) for d in dim)
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

LINEAR_RULES = Family('epsilon', {
    'epsilon':        epsilon,
    'zplus':          zplus,
    'gamma':          gamma,
    'gamma_montavon': gamma_montavon,
    'alpha_beta':     alpha_beta,
    'zbox':           zbox,
})

BMM_RULES = Family('epsilon', {
    'epsilon':     epsilon_bmm,
    'uniform':     uniform_bmm,
    'detach_lhs':  detach_lhs_bmm,
    'detach_rhs':  detach_rhs_bmm,
})

MUL_RULES = Family('proportional', {
    'proportional': proportional,
    'detach_lhs':   detach_lhs,
    'detach_rhs':   detach_rhs,
})

ADD_RULES = Family('proportional', {
    'proportional': proportional,
    'equal':        equal,
    'fixed':        fixed,
    'detach_lhs':   detach_lhs,
    'detach_rhs':   detach_rhs,
})

SOFTMAX_RULES = Family('passthrough', {
    'passthrough': passthrough,
    'jacobian':    softmax_jacobian,
    'detach':      softmax_detach,
})

LAYERNORM_RULES = Family('identity', {
    'identity':    layernorm_identity,
    'passthrough': passthrough,
    'yx':          layernorm_yx,
    'detach_std':  layernorm_detach_std,
})

ACTIVATION_RULES = Family('passthrough', {
    'passthrough': passthrough,
    'yx':          activation_yx,
})

REDUCTION_RULES = Family('proportional', {
    'proportional': reduction_proportional,
})

# The one virtual name, resolved per node to detach_lhs/detach_rhs from
# the slot fact named by its required by= kwarg.
VIRTUAL_RULES = frozenset({'detach'})

# Node name (no version digit) to table; these are the config keys.
FAMILIES: Dict[str, Family] = {
    'AddmmBackward':       LINEAR_RULES,
    'MmBackward':          LINEAR_RULES,
    'ConvolutionBackward': LINEAR_RULES,
    'BmmBackward':         BMM_RULES,
    'MulBackward':         MUL_RULES,
    'DivBackward':         MUL_RULES,
    'AddBackward':         ADD_RULES,
    'SubBackward':         ADD_RULES,
}
