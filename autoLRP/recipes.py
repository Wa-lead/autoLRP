r"""Recipes that combine several LRP runs: :func:`bilrp` (Eberle et al.
2022) and :func:`clrp` (Gu et al. 2018). No new propagation rules.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn

from .config import LRPConfig, BASE
from .tensor import tensor


def bilrp(
    model: nn.Module,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    *,
    n_dims: Optional[int] = None,
    project: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    reduce_each: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    config: Optional[LRPConfig] = None,
    return_similarity: bool = False,
    generator: Optional[torch.Generator] = None,
) -> "torch.Tensor | tuple[torch.Tensor, float]":
    r"""BiLRP pairwise attribution (Eberle et al. 2022):
    ``R_pair = sum_m phi_m(x_a) phi_m(x_b) r_a^(m) (x) r_b^(m)``, where
    ``phi_m`` are the embedding (or projection) dimensions and ``r^(m)``
    the per-side relevance maps reduced by ``reduce_each``.

    Args:
        model: called twice per dimension.
        x_a, x_b: inputs with batch size 1, not pre-wrapped.
        n_dims: dimensions summed over; below the embedding size a random
            Gaussian projection is appended as a layer.
        project: output to ``(B, D)``; default ``flatten(start_dim=1)``.
        reduce_each: relevance to ``(B, n)``; default ``flatten``.
        config: default ``LRPConfig()``; conservation (paper Prop. 1)
            holds for the epsilon and gamma families.
        return_similarity: also return ``sum_m phi_m(x_a) phi_m(x_b)``,
            the conservation target.
        generator: RNG for the random projection (drawn when
            ``n_dims < emb_dim``), making results reproducible.

    Returns ``(n_a, n_b)``, or ``(R_pair, similarity)``.
    """
    if x_a.shape[0] != 1 or x_b.shape[0] != 1:
        raise ValueError(
            "bilrp requires batch size 1 for x_a and x_b, got "
            f"{x_a.shape[0]} and {x_b.shape[0]}; the seed sums over the "
            "batch while only entry 0 of the relevance is kept.")
    config = config or LRPConfig()
    if project is None:
        project = lambda out: out.flatten(start_dim=1) if out.ndim > 2 else out
    if reduce_each is None:
        reduce_each = lambda r: r.flatten(start_dim=1)

    with torch.no_grad():
        emb_dim = project(model(x_a)).shape[-1]

    if n_dims is None or n_dims >= emb_dim:
        n_dims = emb_dim
        P = None
    else:
        # The projection is a layer; LRP runs through it like any other.
        P = torch.randn(emb_dim, n_dims, device=x_a.device,
                        generator=generator) / math.sqrt(n_dims)

    def _scalar(x_wrapped, m):
        emb = project(model(x_wrapped))
        if P is not None:
            emb = emb @ P
        return emb[..., m].sum()

    R_pair = None
    similarity = 0.0
    for m in range(n_dims):
        # Unit seed per side; phi_m(x_a) phi_m(x_b) is multiplied in below
        # (propagation is linear in the seed).
        xa = tensor(x_a.detach())
        s_a = _scalar(xa, m)
        w_a = float(s_a.detach().item())
        s_a.lrp(config=config)
        r_a = reduce_each(xa.relevance.detach())[0]

        xb = tensor(x_b.detach())
        s_b = _scalar(xb, m)
        w_b = float(s_b.detach().item())
        s_b.lrp(config=config)
        r_b = reduce_each(xb.relevance.detach())[0]

        if R_pair is None:
            R_pair = torch.zeros(r_a.numel(), r_b.numel(), device=x_a.device)
        R_pair += (w_a * w_b) * torch.outer(r_a, r_b)
        similarity += w_a * w_b

    if return_similarity:
        return R_pair, similarity
    return R_pair


def clrp(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    *,
    config: Optional[LRPConfig] = None,
    low: float = -1.0,
    high: float = 1.0,
) -> torch.Tensor:
    r"""Contrastive LRP (Gu et al. 2018, CLRP1):
    ``R = max(0, R_target / |sum R_target| - R_dual / |sum R_dual|)``,
    ``R_dual`` seeded with the dual mask ``|out_c|`` on every class but
    ``target`` — the seed ``sum_c out_c |out_c|`` distributes over classes
    proportionally to ``|out_c * mask_c| = out_c^2``, so each non-target
    class effectively contributes in proportion to its squared logit.
    Keeps what contributed to ``target`` beyond the other classes. ``x``
    is not pre-wrapped; ``config`` defaults to z+ with the z-box input
    rule bounded by ``low``/``high``. Pass the actual input-domain bounds
    (e.g. per the ImageNet normalization, roughly ``[-2.1, 2.6]``); the
    ``[-1, 1]`` default is only a generic fallback. ``low``/``high`` are
    ignored when ``config`` is given. Returns a non-negative map shaped
    like ``x``.
    """
    config = config or LRPConfig(rule={
        **BASE,
        'input_conv': ('zbox', {'low': low, 'high': high}),
        'ConvolutionBackward': 'zplus',
        'AddmmBackward': 'zplus',
        'MmBackward': 'zplus'})
    target = int(target)

    xa = tensor(x.detach())
    out_a = model(xa)
    out_a[0, target].lrp(config=config)
    R_target = xa.relevance.detach()

    xb = tensor(x.detach())
    out_b = model(xb)
    mask = out_b.detach().abs()
    mask[..., target] = 0.0
    (out_b * mask).sum().lrp(config=config)
    R_dual = xb.relevance.detach()

    eps = 1e-9
    R_target_n = R_target / (R_target.sum().abs() + eps)
    R_dual_n   = R_dual   / (R_dual.sum().abs()   + eps)
    return torch.clamp(R_target_n - R_dual_n, min=0.0)
