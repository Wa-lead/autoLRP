r"""Attribution adapters — one per method, all implementing the contract:

    attribute(model, input_ids, target, **kw) -> relevance   # 1-D, per input unit

These are the ONLY method-specific code. Each returns a 1-D relevance aligned to
input positions and does NOTHING else (no scoring, no occlusion). The runner
calls one of these, then hands the relevance to scorers.morf_lerf / tgs_tps.

STATUS: UNTESTED SCAFFOLDING. The API calls (autoLRP .relevance, captum,
LXT monkey_patch) are transcribed from source and have NOT been executed.
Smoke-test each on 2 examples before trusting it. Per-model wiring
(embedding path, T5 decoder, the autoLRP requires-grad trap) is centralized in
model_setup.py.
"""
from __future__ import annotations
from typing import Optional
import torch


# ---------------------------------------------------------------------------
# autoLRP
# ---------------------------------------------------------------------------
def attribute_autolrp(model, input_ids, target, *, embed_layer, config,
                      decoder_input_ids=None, device=None, output_kind="logits"):
    r"""autoLRP relevance. Wraps the embedding output as an LRPTensor, runs the
    forward, seeds the target scalar, reads `.relevance`.

    output_kind: 'logits' (seq-cls / causal LM) or 'span' (EQA start+end).
    For 'span', `target` must be (start_idx, end_idx).
    """
    import autoLRP as autolrp
    device = device or next(model.parameters()).device
    x = autolrp.tensor(embed_layer(input_ids).detach())
    fwd = {"inputs_embeds": x}
    if decoder_input_ids is not None:
        fwd["decoder_input_ids"] = decoder_input_ids
    out = model(**fwd)

    if output_kind == "span":
        start, end = target
        scalar = out.start_logits[0, start] + out.end_logits[0, end]
    else:
        logits = out.logits if hasattr(out, "logits") else out
        if logits.dim() == 3:        # causal LM: last position
            scalar = logits[0, -1, target]
        else:                         # seq-cls
            scalar = logits[0, target]
    scalar.lrp(config=config)
    return x.relevance[0].sum(-1).detach().cpu()


# ---------------------------------------------------------------------------
# Integrated Gradients / GradientShap (captum, on embeddings)
# ---------------------------------------------------------------------------
def attribute_captum(model, input_ids, target, *, method, embed_layer,
                     attention_mask=None, decoder_input_ids=None, device=None,
                     output_kind="logits", n_steps=50, n_samples=50):
    r"""IG or GradShap on the embedding inputs. method in {'ig','gradshap'}.
    n_samples=50 matches the authors' captum scripts (was 20 → noisy GradShap)."""
    from captum.attr import IntegratedGradients, GradientShap
    device = device or next(model.parameters()).device

    def forward(emb):
        if emb.dtype != model.dtype:
            emb = emb.to(model.dtype)
        kw = {"inputs_embeds": emb}
        if attention_mask is not None:
            kw["attention_mask"] = attention_mask
        if decoder_input_ids is not None:
            kw["decoder_input_ids"] = decoder_input_ids.repeat(emb.shape[0], 1) if emb.shape[0] > 1 else decoder_input_ids
        out = model(**kw)
        if output_kind == "span":
            start, end = target
            bs = emb.shape[0]
            return out.start_logits[torch.arange(bs), start] + out.end_logits[torch.arange(bs), end]
        logits = out.logits if hasattr(out, "logits") else out
        if logits.dim() == 3:
            return logits[:, -1, target]
        return logits[:, target]

    e = embed_layer(input_ids).detach().float()
    e.requires_grad_()
    baseline = torch.zeros_like(e)
    if method == "ig":
        a = IntegratedGradients(forward).attribute(e, baselines=baseline,
                                                   n_steps=n_steps, internal_batch_size=4)
    elif method == "gradshap":
        gs = GradientShap(forward)
        total, rem, bs = None, n_samples, 5
        while rem > 0:
            cb = min(rem, bs)
            ab = gs.attribute(e, baselines=baseline, n_samples=cb, stdevs=0.0)
            total = ab * cb if total is None else total + ab * cb
            rem -= cb
        a = total / n_samples
    else:
        raise ValueError(method)
    return a.sum(-1)[0].detach().cpu()


# ---------------------------------------------------------------------------
# AttnLRP (LXT efficient: monkey_patch replaces backward with LRP rules, so
# embeds*grad after backward IS the redistributed relevance).
# ---------------------------------------------------------------------------
def attribute_attnlrp(model, input_ids, target, *, embed_layer,
                      attention_mask=None, decoder_input_ids=None, device=None,
                      output_kind="logits"):
    r"""AttnLRP via LXT. The model must already be monkey_patched by the caller
    (patch is global; do it once at load). Returns embeds * embeds.grad summed
    over hidden dim. `decoder_input_ids` is required for enc-dec models (T5).
    """
    device = device or next(model.parameters()).device
    e = embed_layer(input_ids).requires_grad_()
    e.retain_grad()
    kw = {"inputs_embeds": e}
    if attention_mask is not None:
        kw["attention_mask"] = attention_mask
    if decoder_input_ids is not None:
        kw["decoder_input_ids"] = decoder_input_ids
    out = model(**kw)
    if output_kind == "span":
        start, end = target
        scalar = out.start_logits[0, start] + out.end_logits[0, end]
    else:
        logits = out.logits if hasattr(out, "logits") else out
        # Seed the PASSED target (not argmax) so every method explains the same
        # token — required for a fair comparison (e.g. Wiki next-token).
        if logits.dim() == 3:
            scalar = logits[0, -1, target]
        else:
            scalar = logits[0, target]
    scalar.backward()
    return (e * e.grad).sum(-1)[0].detach().cpu()
