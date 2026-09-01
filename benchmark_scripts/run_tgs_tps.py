r"""TGS/TPS runner — EQA (bert/roberta/t5 on SQuADv2).

Thin: load model+data, for each answerable example predict the span, get
relevance from an adapter, call scorers.tgs_tps with membership='index'
(avoids the WordPiece substring trap), accumulate per-example booleans, report
the means. No reduction.

STATUS: UNTESTED end to end. Run smoke_test.py first.

Examples:
    python run_tgs_tps.py --task bert --method autolrp --n 1000
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorers import tgs_tps
import adapters as A
import model_setup as M


def _find_autolrp(repo):
    if repo:
        sys.path.insert(0, os.path.abspath(os.path.expanduser(repo)))


def build_attribute(method, task, model, emb, args, device):
    enc_dec = (task == "t5")
    if method == "autolrp":
        _find_autolrp(args.autolrp_repo)
        from _recipe import build_eqa_config
        cfg = build_eqa_config(args.gamma_linear)

        def attr(ids, am, tgt):
            dec = M.t5_decoder_ids(model, ids) if enc_dec else None
            return A.attribute_autolrp(model, ids, tgt, embed_layer=emb, config=cfg,
                                       decoder_input_ids=dec, device=device, output_kind="span")
        return attr
    if method in ("ig", "gradshap"):
        def attr(ids, am, tgt):
            dec = M.t5_decoder_ids(model, ids) if enc_dec else None
            return A.attribute_captum(model, ids, tgt, method=method, embed_layer=emb,
                                      attention_mask=am, decoder_input_ids=dec,
                                      device=device, output_kind="span")
        return attr
    if method == "attnlrp":
        def attr(ids, am, tgt):
            dec = M.t5_decoder_ids(model, ids) if enc_dec else None
            return A.attribute_attnlrp(model, ids, tgt, embed_layer=emb,
                                       attention_mask=am, decoder_input_ids=dec,
                                       device=device, output_kind="span")
        return attr
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["bert", "roberta", "t5"])
    ap.add_argument("--method", required=True, choices=["autolrp", "ig", "gradshap", "attnlrp"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--membership", default="index", choices=["index", "string"])
    ap.add_argument("--gamma-linear", type=float, default=None,
                    help="linear-layer LRP gamma. Default is per-task: 1.0 for "
                         "roberta/bert (gamma<=0.01 degenerates to epsilon and "
                         "under-localizes spans), 0.001 for t5.")
    ap.add_argument("--autolrp-repo", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    # EQA gamma (benchmark config, not a library change): roberta/bert need the
    # real gamma rule — autoLRP's gamma degenerates to plain epsilon at gamma<=0.01
    # (rules.py), which under-localizes the answer span. t5 is strongest at 0.001.
    if args.gamma_linear is None:
        args.gamma_linear = 0.001 if args.task == "t5" else 1.0

    from datasets import load_dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, emb, kind = M.load_model(
        args.task, for_autolrp=(args.method == "autolrp"),
        for_attnlrp=(args.method == "attnlrp"))
    attribute = build_attribute(args.method, args.task, model, emb, args, device)
    enc_dec = (args.task == "t5")
    ds = load_dataset("squad_v2")["validation"]
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))

    tgs_hits = tps_hits = total = 0
    skips = {"unanswerable": 0, "too_long": 0, "start_gt_end": 0}
    per_example = []

    for ex in tqdm(ds):
        answers = ex["answers"]["text"]
        starts = ex["answers"]["answer_start"]
        if not answers:
            skips["unanswerable"] += 1
            continue
        gold_spans = [(s, s + len(a)) for a, s in zip(answers, starts)]
        enc = tok(ex["question"], ex["context"], return_tensors="pt", return_offsets_mapping=True)
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        offsets = enc["offset_mapping"][0]
        if ids.shape[-1] > 512:
            skips["too_long"] += 1
            continue

        with torch.no_grad():
            dec = M.t5_decoder_ids(model, ids) if enc_dec else None
            out = model(ids, **({"decoder_input_ids": dec} if dec is not None else {}))
        # Find where the context segment begins so the predicted span is
        # restricted to it. Use the fast tokenizer's sequence_ids (0=question,
        # 1=context, None=special) — robust across BERT/RoBERTa AND T5. The old
        # SEP/offset fallback put T5's ctx_start past the context (last (0,0) gap
        # = trailing </s>), collapsing the predicted span to (0,0) and zeroing
        # TPS for every method; it also mis-set RoBERTa's double </s></s>.
        seq_ids = enc.sequence_ids(0)
        ctx_start = next((i for i, s in enumerate(seq_ids) if s == 1), None)
        if ctx_start is None:  # non-fast tokenizer: fall back to SEP / offsets
            sep_id = getattr(tok, "sep_token_id", None)
            if sep_id is not None and sep_id in ids[0].tolist():
                ctx_start = list(ids[0]).index(sep_id) + 1
            else:
                offs = offsets.tolist()
                zero_gaps = [i for i, (a, b) in enumerate(offs) if a == 0 and b == 0]
                ctx_start = (zero_gaps[-1] + 1) if zero_gaps else 1
        mask = torch.zeros_like(out.start_logits).bool()
        mask[:, ctx_start:] = True
        start = int(out.start_logits.masked_fill(~mask, -1e9).argmax())
        end = int(out.end_logits.masked_fill(~mask, -1e9).argmax())
        if start > end:
            skips["start_gt_end"] += 1
            continue

        # Context mask (True only for context tokens) — the top-1 attributed
        # token is searched within the context only, the standard QA-eval step
        # (relevance.masked_fill(~mask,-inf)).
        ctx_mask = torch.tensor([s == 1 for s in seq_ids])
        try:
            rel = attribute(ids, am, (start, end))
            rel_ctx = rel.clone().masked_fill(~ctx_mask.to(rel.device), float("-inf"))
            tgs, tps = tgs_tps(
                rel, gold_char_spans=gold_spans, predicted_token_span=(start, end),
                token_offsets=offsets, membership=args.membership,
                gold_answer_strings=answers, context_mask=ctx_mask,
                top_token_string=(tok.decode([int(ids[0][int(rel_ctx.argmax())])]).strip()
                                  if args.membership == "string" else None))
            tgs_hits += int(tgs); tps_hits += int(tps); total += 1
            per_example.append({"id": ex["id"], "tgs": bool(tgs), "tps": bool(tps),
                                "pred_span": [start, end], "top_idx": int(rel_ctx.argmax())})
        except Exception as e:
            print(f"  [skip] {type(e).__name__}: {e}")
        torch.cuda.empty_cache()

    n = max(total, 1)
    print(f"\n{args.task} {args.method}: TGS={tgs_hits/n:.4f}  TPS={tps_hits/n:.4f}  (n={total})")
    print(f"  skips: {skips}")
    out = args.out_json or f"results/{args.task}_{args.method}_tgstps.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"task": args.task, "method": args.method,
               "summary": {"tgs": tgs_hits/n, "tps": tps_hits/n, "n": total},
               "membership": args.membership, "skips": skips,
               "per_example": per_example}, open(out, "w"))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
