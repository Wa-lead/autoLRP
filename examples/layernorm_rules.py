r"""The two LayerNorm rules under the faithfulness metrics.

``identity`` is the default and is what the statistic_operand analyzer
does on a LayerNorm written out in primitive ops: mean, std and the
affine weight are detached, relevance passes straight through, the bias
takes its share. ``detach_std`` detaches the std only: the centering
``x - mean(x)`` is propagated as a sub, so the mean's share returns over
``x`` (LXT's rule). Both conserve; they differ in where relevance lands.
This script asks whether that difference shows in ``autolrp.metrics`` on
pretrained models with real inputs: ViT-B/16 (torchvision, ImageNet
weights) on the showcase images and BERT SST-2 on a few sentences. A
random map is the floor for the AOPC numbers.

Run: python examples/layernorm_rules.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                            # autolrp without pip install -e
sys.path.insert(0, str(ROOT / 'examples' / 'showcase'))  # _common

import torch

import autolrp
from autolrp import LRPConfig, BASE, CPLRP, on, LAYERNORM_NODES, metrics

RULES = ('identity', 'detach_std')


def _ranks(v):
    r = torch.empty_like(v)
    r[v.argsort()] = torch.arange(v.numel(), dtype=v.dtype)
    return r


def _agreement(a, b, top=None):
    r"""Pearson and Spearman correlation of two maps; with ``top``, also the
    overlap of their top-``top`` sets by ``|R|``."""
    a, b = a.flatten().float(), b.flatten().float()
    out = {'pearson': torch.corrcoef(torch.stack([a, b]))[0, 1].item(),
           'spearman': torch.corrcoef(torch.stack([_ranks(a), _ranks(b)]))[0, 1].item()}
    if top is not None:
        k = max(1, int(a.numel() * top))
        top_a = set(a.abs().topk(k).indices.tolist())
        top_b = set(b.abs().topk(k).indices.tolist())
        out['overlap'] = len(top_a & top_b) / k
    return out


def _print_agreement(prefix, agree):
    mean = {k: sum(a[k] for a in agree) / len(agree) for k in agree[0]}
    line = f"{prefix}: Pearson {mean['pearson']:.3f}, Spearman {mean['spearman']:.3f}"
    if 'overlap' in mean:
        line += f", top-10% overlap {mean['overlap']:.2f}"
    print(line)


def _print_row(name, f):
    print(f"    {name:11s} conservation {f['conservation']:6.3f}   "
          f"AOPC deletion {f['aopc_deletion']:+.4f}   insertion {f['aopc_insertion']:+.4f}")


def _print_mean(rows):
    print("  mean over inputs")
    for name, fs in rows.items():
        mean = {k: sum(f[k] for f in fs) / len(fs) for k in fs[0]}
        _print_row(name, mean)


def _random_row(x, score, R, n_steps, baseline='zero'):
    return {'conservation': float('nan'),
            'aopc_deletion': metrics.aopc(x, score, R=R, mode='deletion', n_steps=n_steps, baseline=baseline),
            'aopc_insertion': metrics.aopc(x, score, R=R, mode='insertion', n_steps=n_steps, baseline=baseline)}


def vit_experiment(n_steps=10):
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    from _common import load_showcase_images, imagenet_classes

    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval()
    gamma = ('gamma', {'gamma': 0.25})
    rule = {**BASE, 'AddmmBackward': gamma, 'MmBackward': gamma, 'ConvolutionBackward': gamma}
    images = load_showcase_images()
    names = imagenet_classes()
    print(f"ViT-B/16 (ImageNet weights), {len(images)} images: gamma 0.25 on the linear family, "
          f"CPLRP attention; pixels ranked by |R|, baseline zero (the mean colour), {n_steps} steps")
    rows = {r: [] for r in RULES + ('random',)}
    agree = []
    for img, label in images:
        with torch.no_grad():
            pred = int(model(img).argmax(-1))
        score = lambda t, p=pred: torch.softmax(model(t)[0], -1)[p]
        print(f"  {label} (predicted {names[pred]!r})")
        maps = {}
        for ln in RULES:
            x = autolrp.tensor(img.clone())
            model(x)[0, pred].lrp(config=LRPConfig(rule={**rule, **CPLRP, **on(LAYERNORM_NODES, ln)}))
            maps[ln] = x.relevance.detach().clone()
            f = metrics.faithfulness(x, score, n_steps=n_steps)
            rows[ln].append(f)
            _print_row(ln, f)
        torch.manual_seed(0)
        f = _random_row(x, score, torch.randn_like(img), n_steps)
        rows['random'].append(f)
        _print_row('random', f)
        agree.append(_agreement(maps['identity'], maps['detach_std'], top=0.1))
        _print_agreement("    identity vs detach_std", agree[-1:])
    _print_mean(rows)
    _print_agreement("  mean agreement", agree)


def bert_experiment():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    name = 'textattack/bert-base-uncased-SST-2'
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()

    class BertClf(torch.nn.Module):
        r"""Embeddings to sentiment logits, as in the showcase."""
        def __init__(self, m):
            super().__init__()
            self.encoder, self.pooler, self.classifier = m.bert.encoder, m.bert.pooler, m.classifier

        def forward(self, x):
            return self.classifier(self.pooler(self.encoder(x).last_hidden_state))

    wrapper = BertClf(model).eval()
    sentences = [
        'The movie was absolutely fantastic and I loved every minute of it',
        'This was the worst film I have ever seen in my life',
        'The acting was great but the plot was terrible',
        'Not bad at all, surprisingly good',
    ]
    print(f"\nBERT SST-2 (post-LN), {len(sentences)} sentences: BASE (epsilon everywhere); whole tokens "
          f"ranked by their summed relevance, one token per step, a removed token becomes [MASK]")
    rows = {r: [] for r in RULES + ('random',)}
    agree = []
    for text in sentences:
        ids = tok(text, return_tensors='pt').input_ids
        emb = model.bert.embeddings(ids).detach()
        base = model.bert.embeddings(torch.full_like(ids, tok.mask_token_id)).detach()
        with torch.no_grad():
            logits = wrapper(emb)
            pred = int(logits.argmax(-1))
        score = lambda t, p=pred: torch.softmax(wrapper(t), -1)[0, p]
        n_tokens = ids.shape[1]
        label = ['negative', 'positive'][pred]
        print(f"  {text!r} -> {label} ({torch.softmax(logits, -1)[0, pred]:.0%})")
        maps = {}
        for ln in RULES:
            x = autolrp.tensor(emb.clone())
            wrapper(x)[0, pred].lrp(config=LRPConfig(rule={**BASE, **on(LAYERNORM_NODES, ln)}))
            per_token = x.relevance.detach().sum(-1, keepdim=True)     # (1, T, 1)
            maps[ln] = per_token.flatten().clone()
            f = metrics.faithfulness(x, score, R=per_token.expand_as(emb), n_steps=n_tokens,
                                     baseline=base)
            rows[ln].append(f)
            _print_row(ln, f)
        torch.manual_seed(0)
        f = _random_row(x, score, torch.randn(1, n_tokens, 1).expand_as(emb), n_tokens, base)
        rows['random'].append(f)
        _print_row('random', f)
        agree.append(_agreement(maps['identity'], maps['detach_std']))
        _print_agreement("    identity vs detach_std, per token", agree[-1:])
    _print_mean(rows)
    _print_agreement("  mean agreement", agree)


if __name__ == '__main__':
    t0 = time.time()
    vit_experiment()
    print(f"  ({time.time() - t0:.0f} s)")
    t0 = time.time()
    bert_experiment()
    print(f"  ({time.time() - t0:.0f} s)")
