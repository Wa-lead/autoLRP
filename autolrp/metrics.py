r"""Faithfulness of an attribution: :func:`conservation` (Bach et al.
2015), :func:`perturbation_curve` (Petsiuk et al. 2018), :func:`aopc`
(Samek et al. 2017) and :func:`faithfulness`, the three in one call.

Every function takes the explained tensor ``x``, the
:class:`~autolrp.LRPTensor` after ``.lrp()``, whose ``.relevance`` is the
attribution, and ``score``: the number that was explained as a function of
the input, the same expression that was seeded, e.g.
``lambda x: model(x)[0, pred]``. ``R=`` evaluates another attribution on the
same input instead of ``x.relevance``.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import numpy as np
import torch

__all__ = ['conservation', 'perturbation_curve', 'aopc', 'faithfulness']

Score = Callable[[torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# conservation
# ---------------------------------------------------------------------------

def conservation(x: torch.Tensor, seed: float = 1.0) -> float:
    r"""``x.relevance.sum() / seed``: the share of the seeded relevance that
    reached the input. ``.lrp()`` seeds ``+1``, so ``1.0`` means nothing was
    absorbed on the way; the deficit is what biases and constants took. A
    gate, not a score: a uniform map conserves too.
    """
    return float(_relevance(x, None).sum()) / seed


# ---------------------------------------------------------------------------
# perturbation curve, AOPC
# ---------------------------------------------------------------------------

def perturbation_curve(
    x: torch.Tensor,
    score: Score,
    *,
    R: Optional[torch.Tensor] = None,
    mode: str = 'deletion',
    n_steps: int = 20,
    baseline: Union[str, float, torch.Tensor] = 'zero',
) -> Dict[str, object]:
    r"""Deletion or insertion curve (Petsiuk et al. 2018). Positions are
    ranked by ``|R|``; step ``k`` perturbs the top ``k / n_steps`` of them.
    ``'deletion'`` starts from ``x`` and replaces them by ``baseline``,
    ``'insertion'`` starts from ``baseline`` and restores them from ``x``;
    ``score`` is read after each step. ``baseline`` is ``'zero'``, ``'mean'``
    (per channel for ``(B, C, ...)``), a float or a broadcastable tensor.
    Returns ``{'fractions', 'scores', 'auc', 'mode'}`` with ``n_steps + 1``
    points; ``scores[0]`` is the unperturbed start.
    """
    if mode not in ('deletion', 'insertion'):
        raise ValueError(f"mode must be 'deletion' or 'insertion'; got {mode!r}")
    x0 = _input(x)
    rel = _relevance(x, R).expand_as(x0)
    base = _baseline(x0, baseline)
    order = torch.argsort(rel.abs().reshape(-1), descending=True)
    n = order.numel()

    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    scores = np.empty_like(fractions)
    with torch.no_grad():
        for i, frac in enumerate(fractions):
            top = torch.zeros(n, dtype=torch.bool, device=x0.device)
            top[order[:int(round(frac * n))]] = True
            top = top.reshape(x0.shape)
            x_k = torch.where(top, base, x0) if mode == 'deletion' else torch.where(top, x0, base)
            scores[i] = _scalar(score(x_k))
    return {'fractions': fractions, 'scores': scores,
            'auc': float(((scores[1:] + scores[:-1]) / 2 * np.diff(fractions)).sum()),   # trapezoid rule
            'mode': mode}


def aopc(
    x: torch.Tensor,
    score: Score,
    *,
    R: Optional[torch.Tensor] = None,
    mode: str = 'deletion',
    n_steps: int = 20,
    baseline: Union[str, float, torch.Tensor] = 'zero',
) -> float:
    r"""Area over the perturbation curve (Samek et al. 2017), one number
    per curve, larger is better in both modes: for ``'deletion'`` the mean
    drop ``score(x) - score(x_k)`` as the most relevant positions are
    removed, for ``'insertion'`` the mean gain ``score(x_k) - score(base)``
    as they are restored. Same arguments as :func:`perturbation_curve`.
    """
    s = perturbation_curve(x, score, R=R, mode=mode, n_steps=n_steps,
                           baseline=baseline)['scores']
    drop = s[0] - s[1:]
    return float(drop.mean() if mode == 'deletion' else -drop.mean())


def faithfulness(
    x: torch.Tensor,
    score: Score,
    *,
    R: Optional[torch.Tensor] = None,
    n_steps: int = 20,
    baseline: Union[str, float, torch.Tensor] = 'zero',
) -> Dict[str, float]:
    r"""``{'conservation', 'aopc_deletion', 'aopc_insertion'}`` for one
    attribution: the rules are sound, the top of the ranking matters, and
    it is sufficient. ``conservation`` reads ``x.relevance`` regardless of
    ``R``.
    """
    return {
        'conservation': conservation(x),
        'aopc_deletion': aopc(x, score, R=R, mode='deletion', n_steps=n_steps, baseline=baseline),
        'aopc_insertion': aopc(x, score, R=R, mode='insertion', n_steps=n_steps, baseline=baseline),
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _input(x: torch.Tensor) -> torch.Tensor:
    r"""``x`` as a plain detached tensor: perturbed copies are scored with
    the native model, not re-wrapped."""
    return x.detach().as_subclass(torch.Tensor)


def _relevance(x: torch.Tensor, R: Optional[torch.Tensor]) -> torch.Tensor:
    if R is None:
        R = getattr(x, 'relevance', None)
        if R is None:
            raise ValueError("x has no .relevance: call .lrp() on the seeded scalar "
                             "first, or pass R=")
    return R.detach().as_subclass(torch.Tensor)


def _baseline(x: torch.Tensor, baseline) -> torch.Tensor:
    if isinstance(baseline, str):
        if baseline == 'zero':
            return torch.zeros_like(x)
        if baseline == 'mean':
            if x.ndim >= 3:                          # (B, C, ...): per-channel mean
                return x.mean(dim=tuple(range(2, x.ndim)), keepdim=True).expand_as(x)
            return torch.full_like(x, float(x.mean()))
        raise ValueError(f"baseline {baseline!r}; choices: 'zero', 'mean', a float, a tensor")
    if isinstance(baseline, (int, float)):
        return torch.full_like(x, float(baseline))
    if torch.is_tensor(baseline):
        return baseline.to(device=x.device, dtype=x.dtype).expand_as(x)
    raise TypeError(f"baseline must be 'zero'|'mean'|float|tensor; got {type(baseline).__name__}")


def _scalar(s) -> float:
    s = torch.as_tensor(s)
    if s.numel() != 1:
        raise ValueError(f"score must return one number; got shape {tuple(s.shape)}")
    return float(s)
