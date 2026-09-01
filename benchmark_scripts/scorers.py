r"""Faithfulness scorers — clean interfaces, raw output, no reductions.

Two metric families share one upstream attribution contract:

    attribute(model, inputs, target) -> relevance   # 1-D, one score per input unit

  * morf_lerf(...)  -> (morf_curve, lerf_curve)      # text + vision (perturbation)
  * tgs_tps(...)    -> (tgs_hit, tps_hit)            # EQA (localization)

Neither reduces anything. morf_lerf returns the raw curves; you compare them or
plot them. tgs_tps returns per-example booleans; you average yourself.

Every behavioural choice that historically caused divergence is an explicit
argument with a documented default:
  - morf_lerf: score_fn, baseline, rank, n_steps
  - tgs_tps:   membership ('index' vs 'string'), exclude_first
"""
from __future__ import annotations
from typing import Callable, List, Optional, Sequence, Tuple

import torch


# ===========================================================================
# Attribution contract (documentation only — each method implements this)
# ===========================================================================
# def attribute(model, inputs, target) -> torch.Tensor:
#     """Return a 1-D relevance tensor, one score per input unit (token for
#     text/EQA, flattened pixel for vision), aligned to model input positions.
#     No batching: inputs is a single example. No scoring logic here."""


# ===========================================================================
# Interface 1 — MoRF / LeRF  (text + vision)
# ===========================================================================
def morf_lerf(
    model,
    input_ids: torch.Tensor,                 # (1, seq) tokens, OR see vision note
    relevance: torch.Tensor,                 # (seq,) 1-D relevance
    target: int,
    *,
    score_fn: Callable[[object, int], float],
    baseline_value,                          # scalar id (text) / tensor (vision pixel fill)
    n_steps: int = 10,
    rank: str = "abs",                       # 'abs' | 'signed'
    apply_perturbation: Optional[Callable] = None,
    input_kind: str = "ids",
    device: Optional[torch.device] = None,
) -> Tuple[List[float], List[float]]:
    r"""Remove input units most-relevant-first (MoRF) and least-relevant-first
    (LeRF), returning the two raw score curves.

    Args:
        score_fn: (model_output, target) -> float. The single number recorded
            per step — e.g. target-class probability, or 1/0 correctness. YOU
            define what faithfulness is measured on; it is applied identically
            to MoRF and LeRF.
        baseline_value: what a removed unit is replaced with (pad token id for
            text; a fill tensor/blur for vision via apply_perturbation).
        n_steps: number of occlusion points (curve length = n_steps + 1, the
            +1 being the unperturbed point at step 0).
        rank: 'abs' ranks by |relevance| (magnitude of attribution); 'signed'
            ranks by raw value (most-positive first for MoRF).
        apply_perturbation: optional (input_ids, indices, baseline_value) ->
            perturbed_input override. Default replaces token ids in place. Pass
            a custom one for vision (fill pixels) or embedding-space masking.

    Returns:
        (morf_curve, lerf_curve): each length n_steps+1, raw score_fn values,
        step 0 = unperturbed. NO area, NO difference computed.
    """
    if rank not in ("abs", "signed"):
        raise ValueError(f"rank must be 'abs' or 'signed', got {rank!r}")
    if input_kind not in ("ids", "embeds"):
        raise ValueError(f"input_kind must be 'ids' or 'embeds', got {input_kind!r}")
    device = device or next(model.parameters()).device

    key = relevance.abs() if rank == "abs" else relevance
    order_morf = torch.argsort(key, descending=True)   # most relevant first
    order_lerf = torch.argsort(key, descending=False)  # least relevant first
    n_units = relevance.shape[0]

    def _default_perturb(ids, idx, base):
        out = ids.clone()
        out[0, idx] = base
        return out

    perturb = apply_perturbation or _default_perturb

    @torch.no_grad()
    def _score(t):
        out = model(inputs_embeds=t.to(device)) if input_kind == "embeds" else model(t.to(device))
        return float(score_fn(out, target))

    base_ids = input_ids.to(device)
    s0 = _score(base_ids)
    morf_curve = [s0]
    lerf_curve = [s0]

    for step in range(1, n_steps + 1):
        k = int(round(n_units * step / n_steps))
        if k == 0:
            morf_curve.append(s0); lerf_curve.append(s0); continue
        morf_curve.append(_score(perturb(base_ids, order_morf[:k], baseline_value)))
        lerf_curve.append(_score(perturb(base_ids, order_lerf[:k], baseline_value)))

    return morf_curve, lerf_curve


# Common score_fn builders (pick one, pass to morf_lerf) -------------------
def score_target_prob(output, target: int) -> float:
    r"""Softmax probability of `target`. For 2-D logits (B,C) uses [0,target];
    for 3-D (B,T,C) uses the last position [0,-1,target]."""
    logits = output.logits if hasattr(output, "logits") else output
    if logits.dim() == 3:
        logits = logits[0, -1]
    elif logits.dim() == 2:
        logits = logits[0]
    return float(torch.softmax(logits, dim=-1)[target])


def score_correct(output, target: int) -> float:
    r"""1.0 if argmax == target else 0.0 (dataset-accuracy style curves)."""
    logits = output.logits if hasattr(output, "logits") else output
    if logits.dim() == 3:
        logits = logits[0, -1]
    elif logits.dim() == 2:
        logits = logits[0]
    return 1.0 if int(logits.argmax()) == target else 0.0


# ===========================================================================
# Interface 2 — TGS / TPS  (EQA span localization)
# ===========================================================================
def tgs_tps(
    relevance: torch.Tensor,                 # (seq,) 1-D relevance
    *,
    gold_char_spans: Sequence[Tuple[int, int]],     # [(start_char, end_char), ...]
    predicted_token_span: Tuple[int, int],          # (start_idx, end_idx) inclusive
    token_offsets: Optional[torch.Tensor] = None,   # (seq, 2) char offsets, for 'string'/'index-gold'
    gold_answer_strings: Optional[Sequence[str]] = None,
    top_token_string: Optional[str] = None,         # decoded top-1 token, for 'string'
    membership: str = "index",                      # 'index' | 'string'
    exclude_first: bool = True,                      # drop BOS/CLS at position 0
    context_mask: Optional[torch.Tensor] = None,    # (seq,) bool, True for context tokens
) -> Tuple[bool, bool]:
    r"""Localize the single highest-relevance token and test span membership.

    TGS (Token-to-Gold-Span): is the top-1 token in a GOLD answer span?
    TPS (Token-to-Predicted-Span): is the top-1 token in the MODEL's predicted
        span (start <= idx <= end)?

    Args:
        membership: how TGS gold-membership is decided —
            'index'  : top-1 token's char-offset overlaps a gold char span
                       (requires token_offsets + gold_char_spans). Robust to
                       tokenization (WordPiece '##', byte-BPE) — RECOMMENDED.
            'string' : decoded top-1 token is a substring of a gold answer
                       string (requires top_token_string + gold_answer_strings).
                       This is the repo's method and the source of the BERT
                       WordPiece failure ('##'-continuations fail the substring
                       test even when correctly located).
        exclude_first: zero out position 0 before argmax (BOS/CLS is never the
            answer; including it pollutes the top-1).

    Returns:
        (tgs_hit, tps_hit) booleans for this one example.
    """
    if membership not in ("index", "string"):
        raise ValueError(f"membership must be 'index' or 'string', got {membership!r}")

    rel = relevance.clone()
    if exclude_first and rel.numel() > 0:
        rel[0] = float("-inf")
    # Restrict the top-1 search to the CONTEXT tokens (question + special tokens
    # masked out). This is the standard QA-eval step the AttnLRP authors use
    # (relevance.masked_fill(~mask, -inf) before argmax) — without it, methods
    # whose relevance peaks on question/SEP tokens (AttnLRP) are scored ~0 even
    # though they localize the answer correctly within the context.
    if context_mask is not None:
        cm = context_mask.to(rel.device).bool()
        rel = rel.masked_fill(~cm, float("-inf"))
    top_idx = int(rel.argmax())

    # TPS: index within predicted span (always index-based — unambiguous)
    p0, p1 = predicted_token_span
    tps_hit = (p0 <= top_idx <= p1)

    # TGS
    if membership == "index":
        if token_offsets is None:
            raise ValueError("membership='index' requires token_offsets")
        c0, c1 = int(token_offsets[top_idx][0]), int(token_offsets[top_idx][1])
        # overlap test against any gold char span
        tgs_hit = any(not (c1 <= g0 or g1 <= c0) for (g0, g1) in gold_char_spans)
    else:  # 'string'
        if top_token_string is None or gold_answer_strings is None:
            raise ValueError("membership='string' requires top_token_string + gold_answer_strings")
        tgs_hit = any(top_token_string in ans for ans in gold_answer_strings)

    return bool(tgs_hit), bool(tps_hit)
