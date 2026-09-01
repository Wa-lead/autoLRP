r"""Faithfulness metrics for any attribution ``R`` broadcastable to the
input: :func:`perturbation_curve` (Petsiuk et al. 2018), :func:`aopc`
(Samek et al. 2017), :func:`sanity_check_cascade` (Adebayo et al.
2018), :func:`sensitivity_correlation` (Ancona et al. 2018; Bhatt et
al. 2020).
"""
from __future__ import annotations

from typing import Callable, Optional, Union, Dict, List

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'perturbation_curve',
    'aopc',
    'sanity_check_cascade',
    'sensitivity_correlation',
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _broadcast_relevance(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""Broadcast ``R`` to the shape of ``x``."""
    if R.shape == x.shape:
        return R
    try:
        return R.expand_as(x)
    except RuntimeError as e:
        raise ValueError(
            f"R shape {tuple(R.shape)} not broadcastable to x shape "
            f"{tuple(x.shape)}: {e}"
        )


def _baseline_tensor(x: torch.Tensor, baseline) -> torch.Tensor:
    r"""``baseline`` as a tensor shaped like ``x``: ``'zero'``, ``'mean'``
    (per channel for ``(B, C, ...)``), a float, or a broadcastable tensor.
    """
    if isinstance(baseline, str):
        if baseline == 'zero':
            return torch.zeros_like(x)
        if baseline == 'mean':
            if x.ndim >= 3:                              # (B, C, *) → per-channel mean
                reduce_dims = tuple(range(2, x.ndim))
                ch_mean = x.mean(dim=reduce_dims, keepdim=True)
                return ch_mean.expand_as(x)
            return torch.full_like(x, x.mean().item())   # fallback: global mean
        raise ValueError(f"Unknown baseline string {baseline!r}; "
                          f"choices: 'zero', 'mean'")
    if isinstance(baseline, (int, float)):
        return torch.full_like(x, float(baseline))
    if torch.is_tensor(baseline):
        return baseline.to(device=x.device, dtype=x.dtype).expand_as(x)
    raise TypeError(
        f"baseline must be 'zero'|'mean'|float|tensor; got {type(baseline).__name__}"
    )


def _score(logits: torch.Tensor, target: int, score_fn) -> float:
    r"""Reduce ``score_fn(logits, target)`` to a Python float."""
    return float(score_fn(logits, target).item())


def _default_score_fn(logits: torch.Tensor, target: int) -> torch.Tensor:
    r"""Softmax probability of the target class."""
    flat = logits[0] if logits.ndim > 1 else logits
    return F.softmax(flat, dim=-1)[target]


def _prepare_perturbation(
    model: nn.Module,
    x: torch.Tensor,
    R: torch.Tensor,
    baseline,
    target: Optional[int],
    score_fn,
):
    r"""Shared setup for the perturbation metrics: check batch size 1,
    default ``score_fn``, broadcast+detach ``R``, build the baseline
    tensor, run one clean forward, and default ``target`` to its argmax.
    Returns ``(x, R_bc, base, score_fn, target, logits_clean)``.
    """
    if x.shape[0] != 1:
        raise ValueError(f"x must have batch dim 1; got {tuple(x.shape)}")
    score_fn = score_fn or _default_score_fn

    R_bc = _broadcast_relevance(R.detach(), x.detach())
    x = x.detach()
    base = _baseline_tensor(x, baseline)

    with torch.no_grad():
        logits_clean = model(x)
    if target is None:
        # Take argmax over the last axis of the first batch entry.
        target = int(logits_clean.reshape(logits_clean.shape[0], -1)[0].argmax().item())
    return x, R_bc, base, score_fn, target, logits_clean


def _bool_mask(indices: torch.Tensor, n_elem: int, x: torch.Tensor) -> torch.Tensor:
    r"""Boolean mask shaped like ``x`` that is True at the flat ``indices``."""
    mask = torch.zeros(n_elem, dtype=torch.bool, device=x.device)
    mask[indices] = True
    return mask.reshape(x.shape)


# ---------------------------------------------------------------------------
# perturbation_curve  (Petsiuk 2018)
# ---------------------------------------------------------------------------

def perturbation_curve(
    model: nn.Module,
    x: torch.Tensor,
    R: torch.Tensor,
    *,
    mode: str = 'deletion',
    n_steps: int = 20,
    baseline: Union[str, float, torch.Tensor] = 'zero',
    target: Optional[int] = None,
    score_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> Dict[str, np.ndarray]:
    r"""Deletion or insertion curve (Petsiuk et al. 2018): positions ranked
    by ``|R|``, the top ``k`` percent replaced by ``baseline``
    (``'deletion'``) or added back to it (``'insertion'``), the model
    score read after each step. ``x`` has batch size 1; ``target``
    defaults to the argmax, ``score_fn(logits, target)`` to the softmax
    probability. Returns ``{'fractions', 'scores', 'auc', 'mode',
    'target'}`` with ``n_steps + 1`` points.
    """
    if mode not in ('deletion', 'insertion'):
        raise ValueError(f"mode must be 'deletion' or 'insertion'; got {mode!r}")
    x, R_bc, base, score_fn, target, _ = _prepare_perturbation(
        model, x, R, baseline, target, score_fn)

    # Rank input elements by |R|. Sort descending.
    flat_R = R_bc.abs().reshape(-1)
    n_elem = flat_R.numel()
    order = torch.argsort(flat_R, descending=True)

    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    scores = np.empty_like(fractions)

    for i, frac in enumerate(fractions):
        k = int(round(frac * n_elem))
        # Mask shape == x shape; entries marked True are the top-k by |R|.
        mask = _bool_mask(order[:k], n_elem, x)

        if mode == 'deletion':
            x_pert = torch.where(mask, base, x)
        else:                                    # insertion
            x_pert = torch.where(mask, x, base)

        with torch.no_grad():
            logits = model(x_pert)
        scores[i] = _score(logits, target, score_fn)

    # np.trapz was removed in NumPy 2.0 in favor of np.trapezoid.
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    auc = float(trapz(scores, fractions))
    return {
        'fractions': fractions, 'scores': scores, 'auc': auc,
        'mode': mode, 'target': target,
    }


# ---------------------------------------------------------------------------
# aopc  (Bach 2015 / Samek 2017)
# ---------------------------------------------------------------------------

def aopc(
    model: nn.Module,
    x: torch.Tensor,
    R: torch.Tensor,
    *,
    mode: str = 'deletion',
    n_steps: int = 20,
    baseline: Union[str, float, torch.Tensor] = 'zero',
    target: Optional[int] = None,
    score_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> float:
    r"""Area over the perturbation curve (Samek et al. 2017),
    ``mean_l (f(x_0) - f(x_l))`` over the :func:`perturbation_curve`
    steps: both modes rank positions by ``|R|`` descending;
    ``'deletion'`` removes most-relevant-first from ``x`` (``x_0 = x``),
    ``'insertion'`` inserts most-relevant-first into the baseline
    (``x_0`` = baseline). Same arguments as :func:`perturbation_curve`;
    for deletion, larger is a sharper drop.
    """
    curve = perturbation_curve(
        model, x, R,
        mode=mode,
        n_steps=n_steps,
        baseline=baseline,
        target=target,
        score_fn=score_fn,
    )
    return float((curve['scores'][0] - curve['scores']).mean())


# ---------------------------------------------------------------------------
# sanity_check_cascade  (Adebayo 2018)
# ---------------------------------------------------------------------------

def _similarity(a: torch.Tensor, b: torch.Tensor, metric: str) -> float:
    r"""Scalar similarity in :math:`[-1, 1]` between two relevance
    tensors of equal numel (flattened internally): ``'cosine'``,
    ``'spearman'`` (Pearson on ranks), or ``'ssim'`` — which, despite
    the key, is 1 minus a normalized L2 distance, not structural
    similarity (no luminance/contrast/structure terms or windows).
    """
    a_f = a.detach().reshape(-1).double()
    b_f = b.detach().reshape(-1).double()
    if metric == 'cosine':
        denom = a_f.norm() * b_f.norm()
        return float((a_f @ b_f / denom.clamp_min(1e-12)).item())
    if metric == 'spearman':
        # Rank-correlation: Pearson on ranked values.
        ar = a_f.argsort().argsort().double()
        br = b_f.argsort().argsort().double()
        ar -= ar.mean(); br -= br.mean()
        denom = ar.norm() * br.norm()
        return float((ar @ br / denom.clamp_min(1e-12)).item())
    if metric == 'ssim':
        # Not SSIM proper: 1 - normalized L2 distance.
        d = (a_f - b_f).norm() / (a_f.norm() + b_f.norm() + 1e-12)
        return float((1.0 - d).item())
    raise ValueError(f"unknown similarity metric {metric!r}; "
                      f"choices: 'cosine', 'spearman', 'ssim'")


def sanity_check_cascade(
    model: nn.Module,
    x: torch.Tensor,
    attribute_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    *,
    similarity: str = 'spearman',
    randomize: str = 'cumulative',
    layer_filter: Optional[Callable[[nn.Module], bool]] = None,
) -> Dict[str, List]:
    r"""Cascade weight randomization (Adebayo et al. 2018): randomize the
    layers top to bottom (``'cumulative'``, or ``'independent'`` one at a
    time), recompute ``attribute_fn(model, x)`` after each, and report its
    similarity (``'cosine'``, ``'spearman'``, or ``'ssim'`` — 1 minus a
    normalized L2 distance, not structural similarity) to the original.
    ``layer_filter`` selects the modules (default: those with a
    ``weight``). Weights are restored on return. Returns
    ``{'layer_names', 'similarities'}``.
    """
    if similarity not in ('cosine', 'spearman', 'ssim'):
        raise ValueError(f"unknown similarity {similarity!r}")
    if randomize not in ('cumulative', 'independent'):
        raise ValueError(f"unknown randomize mode {randomize!r}")
    layer_filter = layer_filter or (
        lambda m: hasattr(m, 'weight')
                  and isinstance(m.weight, torch.nn.Parameter)
                  and m.weight.numel() > 0
    )

    R_original = attribute_fn(model, x).detach()

    # Discover the layers to randomize, top-to-bottom.
    named = [(n, m) for n, m in model.named_modules() if layer_filter(m)]
    named = list(reversed(named))                # top → bottom

    # Save originals so we can restore at the end (and between steps for
    # 'independent' mode).
    saved = {n: m.weight.detach().clone() for n, m in named}

    names, sims = [], []
    try:
        for n, m in named:
            with torch.no_grad():
                # Match the layer's own weight scale; the 1e-3 floor is a
                # heuristic guard so near-constant weight tensors (std ~ 0)
                # still get non-degenerate random weights.
                m.weight.copy_(torch.randn_like(m.weight) * m.weight.detach().std().clamp_min(1e-3))
            R = attribute_fn(model, x).detach()
            sims.append(_similarity(R, R_original, similarity))
            names.append(n)
            if randomize == 'independent':
                with torch.no_grad():
                    m.weight.copy_(saved[n])
    finally:
        for n, m in named:
            with torch.no_grad():
                m.weight.copy_(saved[n])

    return {'layer_names': names, 'similarities': sims}


# ---------------------------------------------------------------------------
# sensitivity_correlation  (Ancona 2018 / Bhatt 2020)
# ---------------------------------------------------------------------------

def sensitivity_correlation(
    model: nn.Module,
    x: torch.Tensor,
    R: torch.Tensor,
    *,
    subset_size: Union[int, float] = 0.2,
    n_samples: int = 100,
    baseline: Union[str, float, torch.Tensor] = 'zero',
    target: Optional[int] = None,
    score_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    r"""Pearson correlation, over ``n_samples`` random subsets ``S`` of
    positions, between ``sum_{i in S} R_i`` and ``f(x) - f(x without S)``.
    Integer ``subset_size`` is Sensitivity-N (Ancona et al. 2018), a
    fraction is Faithfulness Correlation (Bhatt et al. 2020). Returns
    ``{'correlation', 'n_samples', 'subset_size'}``.
    """
    x, R_bc, base, score_fn, target, logits_clean = _prepare_perturbation(
        model, x, R, baseline, target, score_fn)
    score_clean = _score(logits_clean, target, score_fn)

    flat_R = R_bc.reshape(-1)
    n_elem = flat_R.numel()
    if isinstance(subset_size, float):
        if not (0.0 < subset_size < 1.0):
            raise ValueError(f"fractional subset_size must be in (0,1); "
                              f"got {subset_size}")
        k = max(1, int(round(subset_size * n_elem)))
    else:
        k = int(subset_size)
        if not (0 < k < n_elem):
            raise ValueError(f"subset_size {k} out of range (0, {n_elem})")

    gen = torch.Generator(device='cpu')
    if seed is not None:
        gen.manual_seed(seed)

    sum_Rs = np.empty(n_samples, dtype=np.float64)
    drops  = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        idx = torch.randperm(n_elem, generator=gen)[:k]
        sum_Rs[i] = float(flat_R[idx].sum().item())
        mask = _bool_mask(idx, n_elem, x)
        x_pert = torch.where(mask, base, x)
        with torch.no_grad():
            logits = model(x_pert)
        drops[i] = score_clean - _score(logits, target, score_fn)

    # Pearson correlation.
    a = sum_Rs - sum_Rs.mean()
    b = drops  - drops.mean()
    denom = math.sqrt((a * a).sum() * (b * b).sum()) or 1e-12
    corr = float((a * b).sum() / denom)
    return {'correlation': corr, 'n_samples': n_samples, 'subset_size': k}
