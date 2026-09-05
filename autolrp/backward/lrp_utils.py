r"""Machinery under the rules. Nothing here decides how relevance is
split; these are the tensor helpers the rules and installers share:
the stabilizer, broadcast reduction, the bias split, the linear-family
driver ``run_linear_rule``, the kernel builders for matmul and
convolution, and the fused-attention reconstruction. ``rules`` imports
this file, never the reverse.
"""
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Elementwise helpers
# ---------------------------------------------------------------------------

def stabilize(z: torch.Tensor, eps: float) -> torch.Tensor:
    r"""``z + eps * sign(z)``, with the sign of zero taken as ``+1``, so a
    division by ``z`` never divides by zero and never flips a sign."""
    s = z.sign()
    s[s == 0] = 1.0
    return z + eps * s


def reduce_to_shape(t: torch.Tensor, target_shape) -> torch.Tensor:
    r"""Sum ``t`` over its broadcast axes until it has ``target_shape``:
    the inverse of broadcasting, for a share computed in the output
    space of a broadcast op."""
    while t.dim() > len(target_shape):
        t = t.sum(dim=0)
    for i, sz in enumerate(target_shape):
        if sz == 1 and t.shape[i] != 1:
            t = t.sum(dim=i, keepdim=True)
    return t


def apply_bias_split(R_out: torch.Tensor, z_no_bias: torch.Tensor,
                     bias, eps: float) -> torch.Tensor:
    r"""Scale ``R_out`` by ``|z| / (|z| + |bias|)``, the share of the
    output the linear part produced; the bias keeps the rest. ``R_out``
    unchanged when ``bias`` is ``None``. ``bias`` must broadcast to
    ``z_no_bias``: a conv bias is ``(1, C, 1, ...)``, not ``(C,)``."""
    if bias is None:
        return R_out
    z_abs = z_no_bias.abs()
    try:
        b_abs = bias.abs().expand_as(z_abs)
    except RuntimeError:
        raise ValueError(
            f"bias of shape {tuple(bias.shape)} does not broadcast to the "
            f"output of shape {tuple(z_abs.shape)}; reshape it to "
            f"(1, C, 1, ...) for a convolution") from None
    return (z_abs / (z_abs + b_abs + eps)) * R_out


def topk_filter(t: torch.Tensor, k: float) -> torch.Tensor:
    r"""Keep the top ``k`` fraction of signed values per batch element,
    zeroing the rest; no-op for ``k >= 1``. Breaks conservation, sharpens
    maps on pixel-flipping benchmarks."""
    if k >= 1.0 or t.dim() < 2:
        return t
    flat = t.flatten(start_dim=1)
    keep = max(1, int(k * flat.size(-1)))
    top = flat.topk(keep, dim=-1)
    out = torch.zeros_like(flat)
    out.scatter_(-1, top.indices, top.values)
    return out.view(t.shape)


# ---------------------------------------------------------------------------
# The linear-family driver
# ---------------------------------------------------------------------------

def cache_pair(f):
    r"""Identity-keyed memo for a two-argument kernel, for one call of
    :func:`run_linear_rule`: ``z = fwd(x, w)`` is computed once and shared
    by the bias split and the rule; a call with substituted operands
    (``fwd(x_pos, w_pos)``) misses the cache and computes normally."""
    cache = {}

    def g(a, b):
        key = (id(a), id(b))
        if key not in cache:
            cache[key] = f(a, b)
        return cache[key]
    return g


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
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""One linear-family step: ``z = fwd(x, w)`` without bias, the bias
    split of ``R_out``, ``rule_fn(x, w, R, eps, fwd, bwd_a, bwd_b,
    **kwargs)``, then the optional top-fraction filter. ``fwd`` is
    memoized for this call, so the rule's own ``fwd(x, w)`` is the same
    ``z``. Returns ``(R_in, R_w)``; the linear rules give ``R_w = None``
    so the weight slot keeps its native gradient."""
    with torch.no_grad():
        fwd = cache_pair(fwd)
        z = fwd(x, w)
        R_scaled = apply_bias_split(R_out, z, bias, eps)
        R_in, R_w = rule_fn(x, w, R_scaled, eps, fwd, bwd_a, bwd_b,
                            **rule_kwargs)
        if relevance_filter < 1.0:
            R_in = topk_filter(R_in, relevance_filter)
        return R_in, R_w


# ---------------------------------------------------------------------------
# Kernel builders: (fwd, bwd_a, bwd_b) for an op
#   fwd(x, w)    the op without bias
#   bwd_a(w, s)  VJP of fwd into x, for an output-shaped s
#   bwd_b(x, s)  VJP of fwd into w
# ---------------------------------------------------------------------------

def mm_ops():
    r"""Kernels for ``x @ w``."""
    return (lambda x, w: x @ w,
            lambda w, s: s @ w.transpose(-2, -1),
            lambda x, s: x.transpose(-2, -1) @ s)


def conv_ops(ndim: int, stride, padding, dilation, groups, x_shape):
    r"""Kernels for an ``ndim``-dimensional convolution. The input VJP
    uses ``torch.nn.grad.convNd_input`` with ``x_shape``, which recovers
    the input shape under any stride."""
    conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[ndim]
    grad_in = {1: torch.nn.grad.conv1d_input, 2: torch.nn.grad.conv2d_input,
               3: torch.nn.grad.conv3d_input}[ndim]
    grad_w = {1: torch.nn.grad.conv1d_weight, 2: torch.nn.grad.conv2d_weight,
              3: torch.nn.grad.conv3d_weight}[ndim]
    fwd = lambda x, w: conv(x, w, None, stride, padding, dilation, groups)
    bwd_a = lambda w, s: grad_in(x_shape, w, s, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups)
    bwd_b = lambda x, s, _ws=None: grad_w(
        x, _ws if _ws is not None else s.shape, s, stride=stride,
        padding=padding, dilation=dilation, groups=groups)
    return fwd, bwd_a, bwd_b


def conv_transposed_ops(ndim: int, stride, padding, output_padding,
                        dilation, groups, x, w):
    r"""Kernels for a transposed convolution. Both VJPs are taken by
    autograd on the forward at the saved ``x`` and ``w``, so every
    stride, padding, output_padding and groups combination is covered
    without a hand-written adjoint."""
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


# ---------------------------------------------------------------------------
# Fused attention reconstruction
# ---------------------------------------------------------------------------

def reconstruct_sdpa(q, k, v, lse, mask, is_causal, scale):
    r"""Rebuild a fused attention forward from its backward node's saved
    state: GQA head expansion, scaling, mask or causal fill, and the
    softmax output from the logsumexp identity ``A = exp(scores - lse)``.
    Returns ``(A, scores, k_e, v_e, scale, n_rep)``."""
    B, Hq, T, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    n_rep = Hq // Hkv
    sc = scale if scale is not None else (D ** -0.5)
    k_e = k.repeat_interleave(n_rep, dim=1) if n_rep > 1 else k
    v_e = v.repeat_interleave(n_rep, dim=1) if n_rep > 1 else v
    scores = torch.matmul(q, k_e.transpose(-2, -1)) * sc
    if mask is not None:
        m = mask
        if m.dtype == torch.bool:
            m = torch.where(m, scores.new_zeros(()), scores.new_full((), -1.0e9))
        scores = scores + m
    elif is_causal:
        scores = scores + torch.triu(scores.new_full((T, S), -1.0e9), diagonal=1)
    A = torch.exp(scores - lse[..., :T].unsqueeze(-1))
    return A, scores, k_e, v_e, sc, n_rep
