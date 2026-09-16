r"""A residual-add detector written against autolrp's extension API, and the
experiment that decides whether the library needs one.

    python examples/residual_add.py     # the experiment
    pytest examples/residual_add.py     # the detector test alone

The detector is one analyzer. It tags an ``Add``/``Sub`` whose two operands
descend from a shared tensor: ``a + F(a)`` (identity skip) or ``G(a) + F(a)``
(projection skip). Adds with a constant or a parameter on one side are not
merges and stay untagged. The fact value is the slot of the skip operand,
the shorter path. A config then addresses only those nodes::

    LRPConfig(rule={**BASE, 'residual_add': 'equal'})

so a ResNet remedy (Otsuki et al. 2024) reaches the skip connections and
nothing else.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root: autolrp (or `pip install -e .`)

import torch
import torch.nn as nn

import autolrp
from autolrp import BASE, LRPConfig, metrics, CPLRP
from autolrp.backward import analysis
from autolrp.backward.analysis import register_analyzer
from autolrp.backward.graph import is_input, is_leaf, node_facts, parents, topo_order


# ---------------------------------------------------------------------------
# The extension: one analyzer
# ---------------------------------------------------------------------------

def _subgraph(node):
    return {id(n): n for n in topo_order([node])}


def _weight_leaf(node):
    return is_leaf(node) and not is_input(node)


@register_analyzer('residual_add')
def residual_add(node):
    """The skip slot of an add/sub that merges two paths from one tensor,
    ``None`` for any other node."""
    if not any(k in node.name() for k in ('AddBackward', 'SubBackward')):
        return None
    p = parents(node)                                    # real producers, aliases collapsed
    if len(p) < 2 or p[0] is None or p[1] is None:       # a constant operand: not a merge
        return None
    sub = [_subgraph(p[0]), _subgraph(p[1])]
    shared = [m for i, m in sub[0].items() if i in sub[1] and not _weight_leaf(m)]
    if not shared:                                       # only weights in common: not a merge
        return None
    if id(p[0]) in sub[1]:
        return 0                                         # a + F(a)
    if id(p[1]) in sub[0]:
        return 1                                         # F(a) + a
    return 0 if len(sub[0]) <= len(sub[1]) else 1        # G(a) + F(a): shortcut = shorter path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_facts(scalar):
    """``[(node name, residual_add fact)]`` for the add/sub nodes under ``scalar``, walk order."""
    plan = autolrp.walk(scalar)
    analysis.run(plan)
    return [(n.name(), node_facts(n).get('residual_add'))
            for n, _ in plan if 'AddBackward' in n.name() or 'SubBackward' in n.name()]


CONFIGS = {
    'proportional everywhere': LRPConfig(),
    'equal everywhere':        LRPConfig(rule={**BASE, 'AddBackward': 'equal'}),
    'equal at residual adds':  LRPConfig(rule={**BASE, 'residual_add': 'equal'}),
}


# ---------------------------------------------------------------------------
# Test: the detector tags exactly the merges
# ---------------------------------------------------------------------------

class _FourAdds(nn.Module):
    """Identity skip, projection skip, a parameter add, a constant add."""
    def __init__(self):
        super().__init__()
        self.f1, self.f2, self.proj = (nn.Linear(8, 8, bias=False) for _ in range(3))
        self.pos = nn.Parameter(torch.randn(8))
        self.register_buffer('mask', torch.zeros(8))

    def forward(self, x):
        h = x + self.f1(x).relu()                # identity skip: slot 0 is the skip
        h = self.f2(h).relu() + self.proj(h)     # projection skip: slot 1 is the shorter path
        h = h + self.pos                         # a parameter: not a merge
        h = h + self.mask                        # a constant: not a merge
        return h


def test_detector_tags_only_skip_adds():
    torch.manual_seed(0)
    m = _FourAdds().eval()
    x = autolrp.tensor(torch.randn(1, 8))
    facts = [f for _, f in add_facts(m(x)[0, 0])]     # walk order: last add first
    assert facts == [None, None, 1, 0], facts


# ---------------------------------------------------------------------------
# Experiment: does the distinction change conservation or faithfulness?
# ---------------------------------------------------------------------------

def _row(name, x, score, n_steps):
    f = metrics.faithfulness(x, score, n_steps=n_steps, baseline='mean')
    print(f"  {name:26s} conservation {f['conservation']:6.3f}   "
          f"AOPC deletion {f['aopc_deletion']:+.4f}   insertion {f['aopc_insertion']:+.4f}")


def resnet18_experiment():
    from PIL import Image
    from torchvision import transforms
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    pre = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = pre(Image.open(Path(__file__).resolve().parent.parent / 'data' / 'cat0.jpg').convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        target = int(model(img).argmax(-1))
    score = lambda t: torch.softmax(model(t)[0], -1)[target]

    facts = add_facts(model(autolrp.tensor(img.clone()))[0, target])
    print(f"ResNet-18 (ImageNet weights, real image): {len(facts)} add nodes, "
          f"{sum(f is not None for _, f in facts)} tagged residual "
          f"(skip slot {sorted({f for _, f in facts if f is not None})})")
    for name, cfg in CONFIGS.items():
        x = autolrp.tensor(img.clone())
        model(x)[0, target].lrp(config=LRPConfig(rule={**cfg.rule, 'ConvolutionBackward': 'zplus'}))
        _row(name, x, score, n_steps=20)


def transformer_experiment():
    # Bias-free on purpose: every other rule is then exactly conservative, so
    # only the add rule moves the number. (With biases the split z / (z + b)
    # amplifies on random weights and buries the effect.)
    torch.manual_seed(0)
    enc = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(16, 4, 32, dropout=0.0, batch_first=True, bias=False),
        num_layers=4, enable_nested_tensor=False).eval()
    mask = torch.triu(torch.full((8, 8), float('-inf')), 1)      # causal: a constant added to the scores
    data = torch.randn(1, 8, 16)
    score = lambda t: enc(t, mask=mask)[0, -1, 0]

    facts = add_facts(score(autolrp.tensor(data.clone())))
    tagged = sum(f is not None for _, f in facts)
    print(f"\n4-layer bias-free TransformerEncoder with a causal mask (random weights): "
          f"{len(facts)} add nodes, {tagged} tagged residual; the other {len(facts) - tagged} "
          f"add the mask (a constant slot)")
    for attn_label, frag in [("epsilon attention (BASE): relevance flows through the scores", {}),
                             ("CPLRP: attention weights detached, the scores get none", CPLRP)]:
        print(f"  {attn_label}")
        for name, cfg in CONFIGS.items():
            x = autolrp.tensor(data.clone())
            score(x).lrp(config=LRPConfig(rule={**cfg.rule, **frag}))
            print(f"    {name:26s} conservation {metrics.conservation(x):6.3f}")
    print("  (random weights: only conservation is meaningful here)")


if __name__ == '__main__':
    test_detector_tags_only_skip_adds(); print("detector test: ok\n")
    resnet18_experiment()
    transformer_experiment()
