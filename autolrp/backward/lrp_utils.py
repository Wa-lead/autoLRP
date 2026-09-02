r"""Machinery under the rules: stabilize, the memoized forward, the bias
split, ``run_linear_rule``, the kernel builders, shape reduction, the
fused-attention reconstruction, and the unary policies (softmax,
layernorm, reduction, activation). ``rules`` imports this file, never
the reverse.
"""
import torch
import torch.nn.functional as F
from typing import Callable, Optional, Tuple




def stabilize(z: torch.Tensor, eps: float) -> torch.Tensor:
    r"""Sign-preserving :math:`\epsilon` offset.

    Returns :math:`z + \epsilon \cdot \mathrm{sign}(z)`, with the sign
    of zero entries treated as :math:`+1`. Used to avoid division by
    zero in LRP rules while preserving the sign of ``z``.
    """
    s = z.sign()
    s[s == 0] = 1.0
    return z + eps * s


def apply_bias_split(R_out: torch.Tensor, z_no_bias: torch.Tensor,
                     bias, eps: float) -> torch.Tensor:
    r"""Scale ``R_out`` by ``|z| / (|z| + |bias|)``, the share of the output
    that the linear part produced; the bias keeps the rest. ``R_out``
    unchanged when ``bias`` is ``None``.
    """
    if bias is None:
        return R_out
    z_abs = z_no_bias.abs()
    b_abs = bias.abs()
    if b_abs.shape != z_abs.shape:
        try:
            b_abs = b_abs.expand_as(z_abs)
        except RuntimeError:
            return R_out
    return (z_abs / (z_abs + b_abs + eps)) * R_out


def cache_pair(f):
    r"""Identity-keyed memo for a two-argument kernel: the vanilla
    ``z = fwd(x, w)`` is computed once and shared between the bias split
    and the rule; substituted-operand calls (``fwd(x_pos, w_pos)``) miss
    the cache and compute normally."""
    cache = {}

    def g(a, b):
        key = (id(a), id(b))
        if key not in cache:
            cache[key] = f(a, b)
        return cache[key]
    return g


def _topk_filter(t: torch.Tensor, k: float) -> torch.Tensor:
    r"""Keep the top ``k`` fraction of signed values per batch element,
    zeroing the rest; no-op for ``k >= 1``. Breaks conservation, sharpens
    maps on pixel-flipping benchmarks.
    """
    if k >= 1.0:
        return t
    if t.dim() < 2:
        return t
    flat = t.flatten(start_dim=1)
    n = flat.size(-1)
    keep = max(1, int(k * n))
    top = flat.topk(keep, dim=-1)
    out = torch.zeros_like(flat)
    out.scatter_(-1, top.indices, top.values)
    return out.view(t.shape)


# ---------------------------------------------------------------------------
# Linear-family math
# ---------------------------------------------------------------------------

def run_linear_rule(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor],
    R_out: torch.Tensor,
    rule_fn: Callable,
    rule_kwargs: dict,
    fwd: Callable,
    bwd_a: Callable,
    bwd_b: Callable,
    eps: float,
    relevance_filter: float = 1.0,
) -> torch.Tensor:
    r"""One linear-family step: ``z = fwd(x, w)`` without bias, bias split
    of ``R_out``, ``rule_fn(x, w, R, eps, fwd, bwd_a, bwd_b, **kwargs)``,
    then the optional top-fraction filter. Returns ``(R_in, R_w)``; the
    linear rules give ``R_w = None`` so the weight slot keeps its native
    gradient.
    """
    with torch.no_grad():
        fwd = cache_pair(fwd)          # rule's z = fwd(x, w) is THIS call
        z_no_bias = fwd(x, w)
        R_scaled = apply_bias_split(R_out, z_no_bias, bias, eps)
        R_in, R_w = rule_fn(
            x, w, R_scaled, eps,
            fwd, bwd_a, bwd_b, **rule_kwargs)
        if relevance_filter < 1.0:
            R_in = _topk_filter(R_in, relevance_filter)
        return R_in, R_w



def mm_ops():
    r"""``(fwd, bwd_a, bwd_b)`` for :math:`x @ w`: forward, VJP into x
    (substitutable w), VJP into w (substitutable x)."""
    return (lambda x, w: x @ w,
            lambda w, s: s @ w.transpose(-2, -1),
            lambda x, s: x.transpose(-2, -1) @ s)


def conv_ops(ndim: int, stride, padding, dilation, groups, x_shape):
    r"""``(fwd, bwd_a, bwd_b)`` for an ``ndim``-dimensional convolution. The
    input VJP uses ``torch.nn.grad.convNd_input`` with ``x_shape``, which
    recovers the input shape under any stride; ``conv_transposeNd`` would
    need an ``output_padding`` for that.
    """
    if ndim == 1:
        fwd = lambda x, w: F.conv1d(x, w, None, stride, padding, dilation, groups)
        bwd = lambda w, s: torch.nn.grad.conv1d_input(
            x_shape, w, s, stride=stride, padding=padding,
            dilation=dilation, groups=groups)
        bwd_w = lambda x, s, _ws=None: torch.nn.grad.conv1d_weight(
            x, _ws if _ws is not None else s.shape, s, stride=stride,
            padding=padding, dilation=dilation, groups=groups)
    elif ndim == 2:
        fwd = lambda x, w: F.conv2d(x, w, None, stride, padding, dilation, groups)
        bwd = lambda w, s: torch.nn.grad.conv2d_input(
            x_shape, w, s, stride=stride, padding=padding,
            dilation=dilation, groups=groups)
        bwd_w = lambda x, s, _ws=None: torch.nn.grad.conv2d_weight(
            x, _ws if _ws is not None else s.shape, s, stride=stride,
            padding=padding, dilation=dilation, groups=groups)
    else:
        fwd = lambda x, w: F.conv3d(x, w, None, stride, padding, dilation, groups)
        bwd = lambda w, s: torch.nn.grad.conv3d_input(
            x_shape, w, s, stride=stride, padding=padding,
            dilation=dilation, groups=groups)
        bwd_w = lambda x, s, _ws=None: torch.nn.grad.conv3d_weight(
            x, _ws if _ws is not None else s.shape, s, stride=stride,
            padding=padding, dilation=dilation, groups=groups)
    return fwd, bwd, bwd_w


# ---------------------------------------------------------------------------
# Softmax variants
# ---------------------------------------------------------------------------

def conv_transposed_ops(ndim: int, stride, padding, output_padding,
                        dilation, groups, x, w):
    r"""Forward/backward kernels for a transposed convolution. The
    forward is ``conv_transposeNd``; both VJPs are taken by autograd on
    that forward at the saved ``x`` and ``w``, so every stride, padding,
    output_padding and groups combination is covered without a
    hand-written adjoint."""
    ct = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}[ndim]
    fwd = lambda xx, ww: ct(xx, ww, None, stride, padding, output_padding, groups, dilation)

    def bwd_a(ww, s):
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            return torch.autograd.grad(fwd(xx, ww.detach()), xx, s)[0]

    def bwd_b(xx, s, _ws=None):
        with torch.enable_grad():
            ww = w.detach().requires_grad_(True)
            return torch.autograd.grad(fwd(xx.detach(), ww), ww, s)[0]

    return fwd, bwd_a, bwd_b


def reduction_share(activation, R_out, dim, keepdim, eps):
    r"""Mean/Sum share: distribute R_out proportional to |activation|
    along the reduced dims (unary policy for reduction nodes)."""
    with torch.no_grad():
        argabs = activation.abs()
        if dim is not None:
            ndim = activation.ndim
            dims = tuple(d if d < ndim else d - (1 << 64) for d in dim)
            denom = argabs.sum(dim=dims, keepdim=True) + eps
            ratios = argabs / denom
            if not keepdim:
                shape = list(activation.shape)
                for d in sorted(dims):
                    shape[d] = 1
                R_out = R_out.reshape(shape)
        else:
            denom = argabs.sum() + eps
            ratios = argabs / denom
    return ratios * R_out


def activation_yx(x, y, R_out, eps):
    r"""Generic y/x rule for nonlinearities (Achtibat et al. 2024,
    Prop. 3.2): R_in = R_out * y / stab(x)."""
    return R_out * y / stabilize(x, eps)


def softmax_jacobian(x: torch.Tensor,
                          s: torch.Tensor,
                          R_out: torch.Tensor,
                          dim: int) -> torch.Tensor:
    r"""Softmax rule of Achtibat et al. 2024 (AttnLRP, Prop. 3.1):
    ``R_in = x * (R_out - s * sum_dim(R_out))``, the Taylor form at the
    input point, not the plain VJP. Row sums are not conserved; the
    hidden bias keeps part of the relevance.
    """
    return x * (R_out - s * R_out.sum(dim=dim, keepdim=True))


def softmax_detach(s: torch.Tensor,
                        R_out: torch.Tensor) -> torch.Tensor:
    r"""Heuristic gated softmax propagation: :math:`R_\mathrm{in} = s \cdot
    R_\mathrm{out}`.

    Drops off-diagonal Jacobian terms; not bias-conservative. Sharpens
    attention attributions in some transformer recipes at the cost of
    breaking the conservation property.
    """
    return s * R_out


# ---------------------------------------------------------------------------
# y/x rule (activation / layernorm)
# ---------------------------------------------------------------------------

def layernorm_yx(x: torch.Tensor,
                      normalized_shape,
                      weight: Optional[torch.Tensor],
                      bias: Optional[torch.Tensor],
                      R_out: torch.Tensor,
                      eps: float) -> torch.Tensor:
    r"""LayerNorm :math:`y/x` rule: :math:`R_\mathrm{in} = R_\mathrm{out}
    \cdot \mathrm{LN}(x) / \mathrm{stab}(x, \epsilon)`.

    Ali et al. 2022, "XAI for Transformers: Better Explanations through
    Conservative Propagation", ICML (identity / :math:`y/x` form for
    LayerNorm; cf. §4).
    """
    with torch.no_grad():
        y = F.layer_norm(x, normalized_shape, weight, bias)
    return R_out * y / stabilize(x, eps)


def layernorm_detach_std(x: torch.Tensor,
                               normalized_shape,
                               weight: Optional[torch.Tensor],
                               bias: Optional[torch.Tensor],
                               R_out: torch.Tensor,
                               eps: float) -> torch.Tensor:
    r"""LayerNorm with the standard deviation held constant (Achtibat et al.
    2024, Eq. 9, as in LXT): ``R_in_j = (R_out_j w_j - mean(R_out w)) /
    std``. The bias has no gradient and drops out.
    """
    with torch.no_grad():
        # Normalised dim is the last ``len(normalized_shape)``; for
        # ViT/BERT it's the trailing hidden dim.
        norm_dims = tuple(range(-len(normalized_shape), 0))
        mean = x.mean(dim=norm_dims, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=norm_dims, keepdim=True)
        std = (var + eps).sqrt()
        if weight is not None:
            R_w = R_out * weight
        else:
            R_w = R_out
        # Mean over the normalised dim, broadcastable back.
        R_w_mean = R_w.mean(dim=norm_dims, keepdim=True)
        return (R_w - R_w_mean) / stabilize(std, eps)



def layernorm_identity(x, normalized_shape, weight, bias, R_out, eps,
                       mean=None, rstd=None):
    r"""Fused LayerNorm with mean, std and the affine weight treated as
    statistics: relevance passes unchanged except for the bias, which
    takes its magnitude share as the decomposed graph's Add split would.
    Uses the node's saved ``mean``/``rstd`` when present, so the share is
    exact; recomputes them otherwise. Identity when ``bias`` is ``None``.
    """
    if bias is None:
        return R_out
    with torch.no_grad():
        nd = len(normalized_shape)
        if mean is not None and rstd is not None:
            m = mean.reshape(mean.shape + (1,) * (x.ndim - mean.ndim))
            r = rstd.reshape(rstd.shape + (1,) * (x.ndim - rstd.ndim))
            xn = (x - m) * r
        else:
            norm_dims = tuple(range(-nd, 0))
            mu = x.mean(dim=norm_dims, keepdim=True)
            xc = x - mu
            xn = xc / torch.sqrt(
                xc.pow(2).mean(dim=norm_dims, keepdim=True) + eps)
        y_nb = xn * weight if weight is not None else xn
        aa, bb = y_nb.abs(), bias.abs().expand_as(y_nb)
        return (aa / (aa + bb + eps)) * R_out

def reduce_to_shape(t: torch.Tensor, target_shape) -> torch.Tensor:
    r"""Sum-reduce ``t`` so its shape matches ``target_shape``.

    Inverse of broadcasting: a gradient computed in the broadcast
    output space is summed along the broadcast axes to recover the
    operand-side shape.
    """
    while t.dim() > len(target_shape):
        t = t.sum(dim=0)
    for i, sz in enumerate(target_shape):
        if sz == 1 and t.shape[i] != 1:
            t = t.sum(dim=i, keepdim=True)
    return t


# ---------------------------------------------------------------------------
# Fused attention reconstruction
# ---------------------------------------------------------------------------


def reconstruct_sdpa(q, k, v, lse, mask, is_causal, scale):
    r"""Rebuild the fused attention forward from a SDPA backward node's
    saved state: GQA head expansion, scaling, mask/causal application,
    and the softmax output recovered from the logsumexp identity
    :math:`A = \exp(\mathrm{scores} - \mathrm{lse})`. Pure math -- so it
    is testable directly against a plain softmax of the same inputs.

    Returns ``(A, scores, k_e, v_e, sc, n_rep)``."""
    B, Hq, T, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    n_rep = Hq // Hkv
    sc = scale if scale is not None else (D ** -0.5)
    k_e = k.repeat_interleave(n_rep, dim=1) if n_rep > 1 else k
    v_e = v.repeat_interleave(n_rep, dim=1) if n_rep > 1 else v
    scores = torch.matmul(q, k_e.transpose(-2, -1)) * sc
    if mask is not None:
        mm_ = mask
        if mm_.dtype == torch.bool:
            mm_ = torch.where(mm_, scores.new_zeros(()),
                              scores.new_full((), -1.0e9))
        scores = scores + mm_
    elif is_causal:
        scores = scores + torch.triu(
            scores.new_full((T, S), -1.0e9), diagonal=1)
    A = torch.exp(scores - lse[..., :T].unsqueeze(-1))
    return A, scores, k_e, v_e, sc, n_rep

