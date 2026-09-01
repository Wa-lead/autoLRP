r"""LRP rules, one call convention for all four tables::

    rule(a, b, R_out, eps, fwd, bwd_a, bwd_b, **params) -> (R_a, R_b)

A rule decides how the relevance of ``z = op(a, b)`` divides between
the operands and how each share spreads over the operand's elements.
Contribution rules (linear, bmm) use the op's forward and VJPs,
``s = R/z`` then ``a * bwd_a(b, s)``; split rules (mul, div, add, sub)
work on operands that share ``z``'s shape and ignore the kernels.
Linear rules return ``(R_x, None)``: the weight slot keeps its native
gradient.
"""
import warnings

import torch
from typing import Callable, Dict

from .lrp_utils import stabilize, apply_bias_split


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
# Contribution rules (use the op's forward and VJPs)
#
# Every caller (run_linear_rule, the install hooks) already runs rules
# under torch.no_grad(), so no rule opens its own no_grad block.
# ---------------------------------------------------------------------------

def epsilon(x, w, R_out, eps, fwd, bwd_a, bwd_b):
    r"""LRP-:math:`\epsilon` rule.

    Bach et al. 2015, "On Pixel-Wise Explanations…", §2.

    .. math::
        R_\mathrm{in} = x \odot \bwd\!\left(w,\;
            \frac{R_\mathrm{out}}{\mathrm{stab}(z, \epsilon)}\right).
    """
    z = fwd(x, w)          # memoized by the caller: shared with bias-split
    s = R_out / stabilize(z, eps)
    return x * bwd_a(w, s), None


def zplus(x, w, R_out, eps, fwd, bwd_a, bwd_b):
    r"""LRP-:math:`z^+` rule (positive contributions only).

    Bach et al. 2015 (LRP-:math:`z^+`, special case of
    :math:`\alpha\beta` with :math:`\alpha = 1`, :math:`\beta = 0`).
    """
    x_pos, x_neg = x.clamp(min=0), x.clamp(max=0)
    w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
    z_pos = fwd(x_pos, w_pos) + fwd(x_neg, w_neg)
    s_pos = R_out / stabilize(z_pos, eps)
    return (x_pos * bwd_a(w_pos, s_pos)
            + x_neg * bwd_a(w_neg, s_pos)), None


def alpha_beta(x, w, R_out, eps, fwd, bwd_a, bwd_b, *,
                alpha, beta):
    r"""LRP-alpha-beta (Bach et al. 2015, 2.2): positive contributions scaled
    by ``alpha``, negative by ``-beta``, with ``alpha - beta = 1``
    (``ValueError`` otherwise).
    """
    if abs((alpha - beta) - 1.0) > 1e-6:
        raise ValueError(
            f"alpha_beta requires alpha - beta == 1, got "
            f"alpha={alpha}, beta={beta} (diff={alpha - beta})")
    x_pos, x_neg = x.clamp(min=0), x.clamp(max=0)
    w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
    z_pos = fwd(x_pos, w_pos) + fwd(x_neg, w_neg)
    s_pos = alpha * R_out / stabilize(z_pos, eps)
    R_in = (x_pos * bwd_a(w_pos, s_pos)
            + x_neg * bwd_a(w_neg, s_pos))
    if beta != 0:
        z_neg = fwd(x_pos, w_neg) + fwd(x_neg, w_pos)
        s_neg = -beta * R_out / stabilize(z_neg, eps)
        R_in = R_in + (x_pos * bwd_a(w_neg, s_neg)
                       + x_neg * bwd_a(w_pos, s_neg))
    return R_in, None


def gamma(x, w, R_out, eps, fwd, bwd_a, bwd_b, *, gamma):
    r"""LRP-gamma, sign-symmetric: both ``x`` and ``w`` are split by sign and
    same-sign paths are amplified by ``gamma``, so mixed-sign inputs are
    handled; equals :func:`gamma_montavon` when ``x >= 0``. Degenerates to
    :func:`epsilon` for ``gamma <= 0.01`` (warns once).
    """
    if gamma <= 0.01:   # heuristic cutoff, not a cited constant
        _warn_gamma_degeneration(gamma, 'gamma')
        return epsilon(x, w, R_out, eps, fwd, bwd_a, bwd_b)
    z = fwd(x, w)
    x_pos, x_neg = x.clamp(min=0), x.clamp(max=0)
    w_pos = w + gamma * w.clamp(min=0)
    w_neg = w + gamma * w.clamp(max=0)
    z_pp = fwd(x_pos, w_pos) + fwd(x_neg, w_neg)
    z_pn = fwd(x_pos, w_neg) + fwd(x_neg, w_pos)
    s_pos = (z > 0).to(z.dtype) * R_out / stabilize(z_pp, eps)
    s_neg = (z < 0).to(z.dtype) * R_out / stabilize(z_pn, eps)
    return (x_pos * (bwd_a(w_pos, s_pos)
                     + bwd_a(w_neg, s_neg))
            + x_neg * (bwd_a(w_neg, s_pos)
                       + bwd_a(w_pos, s_neg))), None


def gamma_montavon(x, w, R_out, eps, fwd, bwd_a, bwd_b, *, gamma):
    r"""LRP-gamma as in Montavon et al. 2019 (10.2.3) and zennit:
    ``w' = w + gamma * w+``, then the epsilon rule with ``w'``. Assumes
    ``x >= 0``; on mixed-sign inputs it amplifies positive-weight paths
    whatever the contribution sign, see :func:`gamma`.
    """
    if gamma <= 0.01:   # heuristic cutoff, not a cited constant
        _warn_gamma_degeneration(gamma, 'gamma_montavon')
        return epsilon(x, w, R_out, eps, fwd, bwd_a, bwd_b)
    w_prime = w + gamma * w.clamp(min=0)
    z_prime = fwd(x, w_prime)
    s = R_out / stabilize(z_prime, eps)
    return x * bwd_a(w_prime, s), None


def zbox(x, w, R_out, eps, fwd, bwd_a, bwd_b, *, low, high):
    r"""z-box rule for a bounded input domain ``[low, high]`` (Montavon et al.
    2017, 3.1), meant for the input layer, e.g. normalized pixels.
    """
    z = fwd(x, w)
    L = torch.full_like(x, low)
    H = torch.full_like(x, high)
    w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
    z_box = (z
             - fwd(L, w_pos) - fwd(H, w_neg))   # z IS fwd(x, w), memoized
    s = R_out / stabilize(z_box, eps)
    return (x * bwd_a(w, s)
            - L * bwd_a(w_pos, s)
            - H * bwd_a(w_neg, s)), None


def epsilon_bmm(a, b, R_out, eps, fwd, bwd_a, bwd_b):
    r"""Both operands receive the epsilon share; 2z budget (bilinear
    epsilon, AttnLRP Prop. 3.3)."""
    z = fwd(a, b)
    s = R_out / stabilize(2.0 * z, eps)
    return a * bwd_a(b, s), b * bwd_b(a, s)


def detach_lhs_bmm(a, b, R_out, eps, fwd, bwd_a, bwd_b):
    r"""Zeros for the first operand; the second takes the full epsilon share.
    Which operand is first was decided before the call (positionally, or
    by a slot fact through ``('detach', {'by': <fact>})``).
    """
    z = fwd(a, b)
    s = R_out / stabilize(z, eps)
    return torch.zeros_like(a), b * bwd_b(a, s)


def detach_rhs_bmm(a, b, R_out, eps, fwd, bwd_a, bwd_b):
    r"""Mirror of :func:`detach_lhs_bmm`: zeros for the second
    operand."""
    z = fwd(a, b)
    s = R_out / stabilize(z, eps)
    return a * bwd_a(b, s), torch.zeros_like(b)


def uniform_bmm(a, b, R_out, eps, fwd, bwd_a, bwd_b):
    r"""Gradient-times-input halves on both operands (LXT uniform;
    composes to the 4/4/2 Q/K/V split)."""
    return a * bwd_a(b, R_out) / 2.0, b * bwd_b(a, R_out) / 2.0


# ---------------------------------------------------------------------------
# Split rules (operands share z's shape)
# ---------------------------------------------------------------------------

def detach_rhs(a, b, R_out, eps, *_):
    r"""All relevance to the first operand, zeros to the second; which
    operand is second was decided before the call.
    """
    return R_out, torch.zeros_like(R_out)


def detach_lhs(a, b, R_out, eps, *_):
    r"""Mirror of :func:`detach_rhs`: zeros to the first operand."""
    return torch.zeros_like(R_out), R_out


def proportional(a, b, R_out, eps, *_):
    r"""Split by magnitude |a|/(|a|+|b|); identity projector."""
    aa, bb = a.abs(), b.abs()
    d = aa + bb + eps
    return (aa / d) * R_out, (bb / d) * R_out


def _expand_like(t, R_out):
    if t.shape == R_out.shape:
        return t
    return t.reshape(R_out.shape) if t.numel() == R_out.numel() \
        else t.expand_as(R_out)


def residual_proportional(a, b, R_out, eps, *_):
    r"""Magnitude split with residual-shape broadcasting (Bach et al.
    2015 default; may underweight ResNet blocks where BN+ReLU shrinks
    |F(x)|)."""
    return proportional(_expand_like(a, R_out), _expand_like(b, R_out),
                        R_out, eps)


def residual_equal(a, b, R_out, eps, *_):
    r"""Fixed split R/2 each (Otsuki et al. 2024; recommended for
    ResNets)."""
    return 0.5 * R_out, 0.5 * R_out


def residual_fixed(a, b, R_out, eps, *_, p):
    r"""Fixed split: first operand p*R, second (1-p)*R."""
    return p * R_out, (1.0 - p) * R_out


# ---------------------------------------------------------------------------
# Rule tables
# ---------------------------------------------------------------------------

class Family(dict):
    """A rule table plus the one name it falls back to when a
    ``('detach', {'by': <fact>})`` entry addresses a node that does
    not carry that fact."""

    def __init__(self, default: str, entries: Dict[str, Callable]):
        super().__init__(entries)
        if default not in entries:
            raise ValueError(f"family default {default!r} is not in the table")
        self.default = default


LINEAR_RULES = Family('epsilon', {
    'epsilon':        epsilon,
    'zplus':          zplus,
    'gamma':          gamma,
    'gamma_montavon': gamma_montavon,
    'alpha_beta':     alpha_beta,
    'zbox':           zbox,
})

# Names are positional: 'statistic' and similar words live in facts
# and config entries, never here.
MUL_RULES = Family('proportional', {
    'proportional': proportional,
    'detach_lhs':   detach_lhs,
    'detach_rhs':   detach_rhs,
})

BMM_RULES = Family('epsilon', {
    'epsilon':     epsilon_bmm,
    'uniform':     uniform_bmm,
    'detach_lhs':  detach_lhs_bmm,
    'detach_rhs':  detach_rhs_bmm,
})

ADD_RULES = Family('proportional', {
    'proportional': residual_proportional,
    'equal':        residual_equal,
    'fixed':        residual_fixed,
    'detach_lhs':   detach_lhs,
    'detach_rhs':   detach_rhs,
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
