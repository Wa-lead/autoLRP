r"""MoRF/LeRF runner — text (imdb/wiki) and vision (vgg/vit).

Thin: load model+data, loop examples, call an adapter for relevance, call
scorers.morf_lerf, save the RAW curves. No reduction.

STATUS: UNTESTED end to end. Run smoke_test.py first.

Examples:
    python run_morf_lerf.py --task imdb --method autolrp --n 100
    python run_morf_lerf.py --task vit  --method autolrp --n 100
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorers import morf_lerf, score_target_prob, score_correct
import adapters as A
import model_setup as M


def _find_autolrp(repo):
    if repo:
        sys.path.insert(0, os.path.abspath(os.path.expanduser(repo)))


class _LogitWrap(torch.nn.Module):
    r"""Wrap an HF classifier so forward returns the logits tensor (not a
    ModelOutput) — zennit's Gradient attributor needs a tensor output to seed."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).logits


def _zennit_cnn_attr(model, img, pred, device):
    r"""Zennit CNN-LRP: EpsilonGammaBox composite (ZBox input rule + gamma on
    conv + epsilon elsewhere), |attr|.sum convention. Used by `--method zennit`
    and as the AttnLRP(Zennit) CNN column for VGG (lxt-AttnLRP is
    transformer-only). ImageNet-1k one-hot seed on the predicted class."""
    from zennit.attribution import Gradient as ZGradient
    from zennit.composites import EpsilonGammaBox
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    low, high = (0.0 - mean) / std, (1.0 - mean) / std
    onehot = torch.zeros(1, 1000, device=device); onehot[0, pred] = 1.0
    with ZGradient(model=model, composite=EpsilonGammaBox(low=low, high=high)) as attr:
        _, attribution = attr(img, onehot)
    return attribution.abs().detach()


def build_attribute(method, task, model, tok, emb, kind, args, device):
    r"""Return a closure attribute(input_ids, target) -> relevance for this method."""
    if method == "autolrp":
        _find_autolrp(args.autolrp_repo)
        from autoLRP import LRPConfig
        from _recipe import make_rule
        # Build the text (LLaMA) config from CLI knobs. Defaults are the faithful
        # config: gamma=1.0 linears + softmax=jacobian (the imdb/wiki fix) +
        # BmmBackward epsilon (the old bilinear='full'); the ablation can still
        # flip layernorm/residual-split/bilinear/softmax.
        rule = make_rule(gamma_linear=args.gamma_linear, bilinear=args.bilinear)
        try:
            rule['AddBackward'] = ('fixed', {'p': float(args.residual_split)})
        except ValueError:
            rule['AddBackward'] = args.residual_split  # 'proportional' | 'equal'
        cfg = LRPConfig(
            rule=rule,
            softmax=args.softmax, layernorm=args.layernorm,
            activation=args.activation)
        ok = "logits"
        return lambda ids, tgt: A.attribute_autolrp(
            model, ids, tgt, embed_layer=emb, config=cfg, device=device, output_kind=ok)
    if method in ("ig", "gradshap"):
        return lambda ids, tgt: A.attribute_captum(
            model, ids, tgt, method=method, embed_layer=emb,
            attention_mask=torch.ones_like(ids), device=device, output_kind="logits")
    if method == "attnlrp":
        return lambda ids, tgt: A.attribute_attnlrp(
            model, ids, tgt, embed_layer=emb, attention_mask=torch.ones_like(ids),
            device=device, output_kind="logits")
    raise ValueError(method)


def iter_text(task, tok, n, max_length=512):
    from datasets import load_dataset
    if task == "imdb":
        ds = load_dataset("stanfordnlp/imdb")["test"].shuffle(seed=42)
        for ex in ds:
            ids = tok(ex["text"], return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]
            yield ids, ex["label"]
    else:  # wiki
        from datasets.utils.info_utils import VerificationMode
        files = [f"20231101.en/train-0000{i}-of-00041.parquet" for i in range(7)]
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", data_files=files,
                          verification_mode=VerificationMode.NO_CHECKS)["train"].shuffle(seed=42)
        for ex in ds:
            ids = tok(ex["text"], return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]
            if ids.shape[-1] < 2:
                continue
            yield ids, None  # target set per-example below (next token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["imdb", "wiki", "vgg", "vit"])
    ap.add_argument("--method", required=True, choices=["autolrp", "ig", "gradshap", "attnlrp", "zennit"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-steps", type=int, default=10)        # used only as a fallback
    ap.add_argument("--max-steps", type=int, default=256)     # cap per-token text granularity
    ap.add_argument("--rank", default="signed", choices=["abs", "signed"])
    ap.add_argument("--score", default="prob", choices=["prob", "correct"])
    ap.add_argument("--gamma-linear", type=float, default=1.0)
    # autoLRP rule-knob overrides for ablation. softmax defaults to 'jacobian' (the
    # AttnLRP DeepTaylor softmax): on LLaMA text the old 'passthrough' default treated
    # softmax as identity and under-attributed attention (probe-localized to the
    # attention block; residual/RMSNorm/MLP all agree).
    # jacobian only affects imdb/wiki (vit/vgg use build_vit_config / build_vgg_config).
    ap.add_argument("--softmax", default="jacobian", choices=["passthrough", "jacobian", "detach"])
    ap.add_argument("--layernorm", default="passthrough", choices=["passthrough", "yx", "detach_std"])
    ap.add_argument("--activation", default="passthrough", choices=["passthrough", "yx"])
    ap.add_argument("--bilinear", default="full", choices=["full", "cplrp", "uniform"])
    ap.add_argument("--residual-split", default="proportional")
    # ViT-only ablation overrides (default None = use build_vit_config's paper values:
    # softmax=passthrough, bilinear=full). Lets us test whether the LLaMA attention
    # fix (jacobian softmax) also closes the vit gap.
    ap.add_argument("--vit-softmax", default=None, choices=["passthrough", "jacobian", "detach"])
    ap.add_argument("--vit-bilinear", default=None, choices=["full", "cplrp", "uniform"])
    ap.add_argument("--autolrp-repo", default=None)
    ap.add_argument("--imagenet-dir", default=None)
    ap.add_argument("--attnlrp-attrs", default=None,
                    help="path to a precomputed AttnLRP attribution tensor (.pt) for ViT, "
                         "generated by running lxt's OWN ViT example "
                         "(LRP-eXplains-Transformers), shape (N,3,H,W) or (N,1,1,H,W). "
                         "Required for --task vit --method attnlrp; the paper itself loads "
                         "such a file (ViT.ipynb cell 44) rather than computing it in-process.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_fn = score_target_prob if args.score == "prob" else score_correct

    if args.task in ("imdb", "wiki"):
        model, tok, emb, kind = M.load_model(
            args.task, for_autolrp=(args.method == "autolrp"),
            for_attnlrp=(args.method == "attnlrp"))
        attribute = build_attribute(args.method, args.task, model, tok, emb, kind, args, device)
        baseline = tok.pad_token_id if tok.pad_token_id is not None else 0

        records = []
        seen = 0
        for ids, label in tqdm(iter_text(args.task, tok, args.n), total=args.n):
            if seen >= args.n:
                break
            ids = ids.to(device)
            if args.task == "wiki":
                pos = ids.shape[-1] - 1
                ctx = ids[:, :pos]
                target = int(ids[0, pos].item())
                attr_ids = ctx
            else:
                with torch.no_grad():
                    target = int(model(ids).logits.argmax())
                if target != label:
                    continue
                attr_ids = ids
            try:
                rel = attribute(attr_ids, target)
                if not torch.isfinite(rel).all():
                    continue
                score_ids = attr_ids
                # one token removed per step (capped at --max-steps) — 10 fixed steps
                # was too coarse for a saturating softmax-prob, collapsing all methods.
                n_steps = min(attr_ids.shape[-1], args.max_steps)
                morf, lerf = morf_lerf(model, score_ids, rel, target,
                                       score_fn=score_fn, baseline_value=baseline,
                                       n_steps=n_steps, rank=args.rank, input_kind='ids', device=device)
                records.append({"morf": morf, "lerf": lerf, "target": target})
                seen += 1
            except Exception as e:
                print(f"  [skip] {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

        out = args.out_json or f"results/{args.task}_{args.method}.json"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump({"task": args.task, "method": args.method, "n": len(records),
                   "rank": args.rank, "score": args.score, "n_steps": args.n_steps,
                   "records": records}, open(out, "w"))
        print(f"\nsaved {len(records)} raw morf/lerf curves -> {out}")
        return

    # ---- vision (vgg / vit): pixel-occlusion MoRF/LeRF ----
    model, _, _, kind = M.load_model(
        args.task, for_autolrp=(args.method == "autolrp"),
        for_attnlrp=(args.method == "attnlrp"))
    from torchvision import transforms as T
    if args.task == "vit":
        from torchvision import datasets
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)
        get_logits = lambda o: o.logits
    else:  # vgg
        if not args.imagenet_dir:
            raise SystemExit("vgg needs --imagenet-dir (ImageFolder val set)")
        import torchvision
        tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        ds = torchvision.datasets.ImageFolder(args.imagenet_dir, transform=tf)
        get_logits = lambda o: o

    if args.method == "autolrp":
        _find_autolrp(args.autolrp_repo)
        from _recipe import build_vit_config, build_vgg_config
        if args.task == "vit":
            # vit fix: default softmax=jacobian (passthrough under-attributes the
            # attention block: +0.3700 -> +0.4127 ABPC at n=100). --vit-softmax
            # passthrough recovers the old baseline.
            vcfg = build_vit_config(softmax=(args.vit_softmax or "jacobian"),
                                    bilinear=(args.vit_bilinear or "full"))
            print(f"[vit] softmax={vcfg.softmax} bmm={vcfg.rule['BmmBackward']}")
        else:
            vcfg = build_vgg_config()

    attnlrp_attrs = None
    if args.method == "attnlrp" and args.task == "vit":
        if not args.attnlrp_attrs:
            raise SystemExit("--task vit --method attnlrp requires --attnlrp-attrs PATH "
                             "(precomputed by lxt's own ViT example).")
        attnlrp_attrs = torch.load(args.attnlrp_attrs)
        if attnlrp_attrs.shape[0] < min(args.n, 1):
            raise SystemExit(f"--attnlrp-attrs has {attnlrp_attrs.shape[0]} rows, need >= n")

    H = W = 224

    # Both vision tasks use PATCH-occlusion (16x16 -> 196 units): relevance is
    # max-pooled into the patch grid and whole patches removed in relevance order.
    # Unified across ViT and VGG (was pixel-occlusion at 10 steps for VGG, far too
    # coarse) so the metric granularity is consistent and tractable (vs 50k pixels).
    if args.task in ("vit", "vgg"):
        PATCH = 16
        npx = H // PATCH                       # 14 patches per side
        def vperturb(image, patch_idx, base):
            o = image.clone()
            for pid in patch_idx.tolist():
                r, c = divmod(pid, npx)
                o[0, :, r*PATCH:(r+1)*PATCH, c*PATCH:(c+1)*PATCH] = base
            return o
        def patch_relevance(relmap):
            # per-patch importance = max relevance in the patch (channels summed),
            # matching util.run_morf_lerf_occlusion_patches (MaxPool2d over patch).
            pooled = torch.nn.functional.max_pool2d(relmap.sum(1), kernel_size=PATCH, stride=PATCH)
            return pooled.flatten()
        n_units_for_steps = npx * npx
    else:  # vgg: pixel occlusion
        def vperturb(image, idx, base):
            o = image.clone(); f = o.view(1, 3, H * W); f[0, :, idx] = base
            return f.view(1, 3, H, W)
        def patch_relevance(relmap):
            return relmap.abs().sum(1).flatten()
        n_units_for_steps = None  # use args.n_steps as given

    def vscore(o, t):
        lg = get_logits(o)
        return float(torch.softmax(lg[0], dim=-1)[t])

    records = []
    for i in tqdm(range(min(args.n, len(ds))), total=min(args.n, len(ds))):
        img, _ = ds[i]
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(get_logits(model(img)).argmax(-1))
        try:
            if args.method == "autolrp":
                import autoLRP as autolrp
                x = autolrp.tensor(img)
                o = get_logits(model(x))
                o[0, pred].lrp(config=vcfg)
                relmap = x.relevance.detach()
            elif args.method in ("ig", "gradshap"):
                from captum.attr import IntegratedGradients, GradientShap
                xb = img.clone().requires_grad_()
                fwd = lambda im: get_logits(model(im))
                if args.method == "ig":
                    relmap = IntegratedGradients(fwd).attribute(xb, target=pred, n_steps=50, internal_batch_size=4).detach()
                else:
                    relmap = GradientShap(fwd).attribute(xb, baselines=torch.zeros_like(xb), target=pred, n_samples=20, stdevs=0.0).detach()
            elif args.method == "attnlrp":
                if args.task == "vgg":
                    # The paper's VGG column is "AttnLRP(Zennit)": lxt-AttnLRP is
                    # transformer-only, so the CNN-LRP stand-in is Zennit's
                    # EpsilonGammaBox — identical to the standalone `--method zennit`.
                    relmap = _zennit_cnn_attr(model, img, pred, device)
                else:  # vit: AttnLRP attributions are PRECOMPUTED by lxt's own
                    # ViT example and loaded from --attnlrp-attrs, exactly as the
                    # paper does (ViT.ipynb cell 44 torch.load(...attrs.pt)). We do
                    # NOT compute ViT-AttnLRP in-process — lxt doesn't support HF
                    # ViTForImageClassification, and a hand-built patch_map would be
                    # a reconstruction, not "using lxt as intended". The tensor is
                    # indexed per-example (row i aligns with dataset index i).
                    if attnlrp_attrs is None:
                        raise SystemExit(
                            "--task vit --method attnlrp requires --attnlrp-attrs PATH "
                            "(a .pt tensor produced by lxt's own ViT example). See the "
                            "--attnlrp-attrs help.")
                    a = attnlrp_attrs[i].to(device)
                    while a.dim() > 3 and a.shape[0] == 1:   # (1,1,H,W)->(1,H,W) etc.
                        a = a.squeeze(0)
                    if a.dim() == 2:                          # (H,W) -> (1,H,W)
                        a = a.unsqueeze(0)
                    relmap = a.unsqueeze(0).detach()         # -> (1,C,H,W)
            elif args.method == "zennit":
                # Zennit CNN-LRP baseline (the paper's standalone VGG "Zennit" column).
                if args.task != "vgg":
                    raise SystemExit("--method zennit is the CNN-LRP baseline (vgg only); "
                                     "transformers use attnlrp / autolrp.")
                relmap = _zennit_cnn_attr(model, img, pred, device)
            else:
                raise ValueError(f"unhandled vision method {args.method!r}")
            rel_units = patch_relevance(relmap)
            n_steps = n_units_for_steps if n_units_for_steps is not None else args.n_steps
            morf, lerf = morf_lerf(model, img, rel_units, pred, score_fn=vscore,
                                   baseline_value=0.0, n_steps=n_steps, rank=args.rank,
                                   apply_perturbation=vperturb, input_kind="ids", device=device)
            records.append({"morf": morf, "lerf": lerf, "target": pred})
        except Exception as e:
            print(f"  [skip] {type(e).__name__}: {e}")
        torch.cuda.empty_cache()

    out = args.out_json or f"results/{args.task}_{args.method}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"task": args.task, "method": args.method, "n": len(records),
               "rank": args.rank, "n_steps": args.n_steps, "records": records}, open(out, "w"))
    print(f"\nsaved {len(records)} raw vision morf/lerf curves -> {out}")


if __name__ == "__main__":
    main()
