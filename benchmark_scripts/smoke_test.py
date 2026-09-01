r"""Smoke test — run each adapter on 2 examples, check relevance is finite,
1-D, right length, non-zero. Catches the mechanical breakages (LRPTensor not
propagating, NaN trap, wrong embedding index, LXT-unsupported model, shape
mismatch) before any full run.

    python smoke_test.py --task imdb --method autolrp
"""
from __future__ import annotations
import argparse, os, sys, traceback
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_setup as M


def check(rel, n_expected, name):
    if rel is None: return f"{name}: relevance is None"
    if rel.ndim != 1: return f"{name}: ndim={rel.ndim} (want 1), shape={tuple(rel.shape)}"
    if rel.shape[0] != n_expected: return f"{name}: len={rel.shape[0]} want {n_expected}"
    if not torch.isfinite(rel).all(): return f"{name}: non-finite ({int((~torch.isfinite(rel)).sum())} bad)"
    if float(rel.abs().sum()) == 0: return f"{name}: all zeros (no signal)"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--autolrp-repo", default=None)
    args = ap.parse_args()
    if args.autolrp_repo:
        sys.path.insert(0, os.path.abspath(os.path.expanduser(args.autolrp_repo)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_eqa = args.task in ("bert", "roberta", "t5")

    try:
        if is_eqa:
            from run_tgs_tps import build_attribute
            model, tok, emb, kind = M.load_model(
                args.task, for_autolrp=(args.method=="autolrp"), for_attnlrp=(args.method=="attnlrp"))
            attribute = build_attribute(args.method, args.task, model, emb, args, device)
            enc = tok("What color is the sky?", "The sky is blue.", return_tensors="pt", return_offsets_mapping=True)
            ids = enc["input_ids"].to(device); am = enc["attention_mask"].to(device)
            dec = M.t5_decoder_ids(model, ids) if args.task=="t5" else None
            out = model(ids, **({"decoder_input_ids": dec} if dec is not None else {}))
            start, end = int(out.start_logits[0].argmax()), int(out.end_logits[0].argmax())
            if end < start: end = start
            for trial in range(2):
                rel = attribute(ids, am, (start, end))
                err = check(rel, ids.shape[-1], f"{args.task}/{args.method} trial{trial}")
                print("  PASS" if err is None else f"  FAIL: {err}")
                if err: break
        else:
            from run_morf_lerf import build_attribute
            model, tok, emb, kind = M.load_model(
                args.task, for_autolrp=(args.method=="autolrp"), for_attnlrp=(args.method=="attnlrp"))
            attribute = build_attribute(args.method, args.task, model, tok, emb, kind, args, device)
            ids = tok("A wonderful film." if args.task=="imdb" else "The capital of France is",
                      return_tensors="pt")["input_ids"].to(device)
            if args.task == "wiki":
                pos = ids.shape[-1]-1; ids_attr = ids[:, :pos]; tgt = int(ids[0,pos])
            else:
                ids_attr = ids; tgt = int(model(ids).logits.argmax())
            for trial in range(2):
                rel = attribute(ids_attr, tgt)
                err = check(rel, ids_attr.shape[-1], f"{args.task}/{args.method} trial{trial}")
                print("  PASS" if err is None else f"  FAIL: {err}")
                if err: break
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
