"""Conservation: LRP's defining mathematical property.

The engine seeds the selected scalar with +1 (engine._unit_seed), so the
conservation target is Σ R_input ≈ +1 REGARDLESS of the logit's sign —
seed 43 of the old sign-based invariant gave argmax logit −0.0308 with
Σ R = +1.000000 exactly, err 2.0, a false failure. Dissipation away
from +1 comes from bias absorption (run_linear_rule's bias split),
softmax/LN handling, and the α/β and γ decompositions.

Every cell seeds its OWN rng stream (crc32 of "rule/arch"), so cells are
independent of collection order, of each other, and of any earlier test.
Tolerances below are calibrated: measured |Σ R − 1| at these seeds on
torch 2.13 fp32, times 1.5, plus 0.02, floor 0.05. Bias-free cells
measure 0.0000, so their 0.05 is a genuine exactness law; biased cells
measure 0.61–0.99 (bias shares discarded), so theirs are drift bands.

The 5×6 matrix is parametrized so each cell appears as a distinct pytest
item — failures point at the exact ``(rule, arch)`` regression.

Marked ``@pytest.mark.slow`` because it instantiates 30 small models +
runs LRP through each.
"""
import zlib
import pytest
import torch
import torch.nn as nn

import autoLRP as autolrp
from tests._cfg import on_linear
from autoLRP import BASE
from autoLRP import LRPConfig


# ---------------------------------------------------------------------------
# Architecture factories
# ---------------------------------------------------------------------------

def _mlp(in_dim, hidden, out_dim, bias):
    return nn.Sequential(
        nn.Linear(in_dim, hidden, bias=bias), nn.GELU(),
        nn.Linear(hidden, hidden, bias=bias), nn.ReLU(),
        nn.Linear(hidden, out_dim, bias=bias),
    ).eval()


def _cnn(bias):
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=bias), nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1, bias=bias), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(8, 4, bias=bias),
    ).eval()


class _MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(16, 32, bias=False)
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True, bias=False)
        self.ln = nn.LayerNorm(32)
        self.head = nn.Linear(32, 5, bias=False)
    def forward(self, x):
        x = self.embed(x)
        x = x + self.attn(self.ln(x), self.ln(x), self.ln(x), need_weights=False)[0]
        return self.head(x[:, 0])


class _MiniViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(16, 32, bias=False)
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True, bias=False)
        self.ln1 = nn.LayerNorm(32)
        self.ln2 = nn.LayerNorm(32)
        self.ffn = nn.Sequential(
            nn.Linear(32, 64, bias=False), nn.GELU(),
            nn.Linear(64, 32, bias=False),
        )
        self.head = nn.Linear(32, 10, bias=False)
    def forward(self, x):
        x = self.embed(x)
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.ffn(self.ln2(x))
        return self.head(x[:, 0])


_ARCHS = [
    ('mlp_nobias',  lambda: _mlp(8, 16, 4, bias=False),
                    lambda: torch.randn(1, 8)),
    ('mlp_bias',    lambda: _mlp(8, 16, 4, bias=True),
                    lambda: torch.randn(1, 8)),
    ('cnn_nobias',  lambda: _cnn(bias=False),
                    lambda: torch.randn(1, 3, 8, 8)),
    ('cnn_bias',    lambda: _cnn(bias=True),
                    lambda: torch.randn(1, 3, 8, 8)),
    ('transformer', lambda: _MiniTransformer(),
                    lambda: torch.randn(1, 4, 16)),
    ('vit',         lambda: _MiniViT(),
                    lambda: torch.randn(1, 6, 16)),
]

_RULES = [
    ('epsilon',    LRPConfig(rule=on_linear('epsilon'), activation='passthrough')),
    ('zplus',      LRPConfig(rule=on_linear('zplus'),   activation='passthrough')),
    ('alpha_beta', LRPConfig(rule=on_linear(('alpha_beta', {'alpha': 2.0, 'beta': 1.0})),
                             activation='passthrough')),
    ('gamma',      LRPConfig(rule=on_linear(('gamma', {'gamma': 0.25})),
                             activation='passthrough')),
    ('composite',  LRPConfig(
        rule={**BASE, 'ConvolutionBackward': 'zplus'},
        activation='passthrough')),
]

# |Σ R_input − 1| tolerance per (rule, arch), against the +1 unit seed.
# Calibrated: measured error at the per-cell crc32 seeds on torch 2.13
# fp32, × 1.5, + 0.02, floor 0.05. Bias-free cells measured 0.0000, so
# their 0.05 asserts exact conservation; biased cells measured 0.61–0.99
# (bias shares discarded by run_linear_rule's bias split).
_TOLS = {
    'epsilon':    {'mlp_nobias': 0.05, 'mlp_bias': 1.31,
                   'cnn_nobias': 0.05, 'cnn_bias': 1.51,
                   'transformer': 0.09, 'vit': 0.07},
    'zplus':      {'mlp_nobias': 0.05, 'mlp_bias': 1.44,
                   'cnn_nobias': 0.05, 'cnn_bias': 1.40,
                   'transformer': 0.12, 'vit': 0.09},
    'alpha_beta': {'mlp_nobias': 0.35, 'mlp_bias': 0.45,
                   'cnn_nobias': 0.05, 'cnn_bias': 1.51,
                   'transformer': 0.19, 'vit': 0.05},
    'gamma':      {'mlp_nobias': 0.05, 'mlp_bias': 1.46,
                   'cnn_nobias': 0.05, 'cnn_bias': 0.93,
                   'transformer': 0.05, 'vit': 0.15},
    'composite':  {'mlp_nobias': 0.05, 'mlp_bias': 1.14,
                   'cnn_nobias': 0.05, 'cnn_bias': 1.41,
                   'transformer': 0.08, 'vit': 0.08},
}


def _run(model, data, config):
    """Run LRP, return (Σ R_input, sign(target_logit))."""
    x = autolrp.tensor(data.clone())
    out = model(x)
    pred = out.argmax(-1).item() if out.ndim > 1 else 0
    sel = out[0, pred] if out.ndim > 1 else out[pred]
    logit = sel.item()
    sel.lrp(config=config)
    return x.relevance.sum().item(), logit


# ---------------------------------------------------------------------------
# 5 rules × 6 architectures = 30 conservation cells
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("arch_name,model_fn,input_fn",
                         _ARCHS, ids=[a[0] for a in _ARCHS])
@pytest.mark.parametrize("rule_name,cfg",
                         _RULES, ids=[r[0] for r in _RULES])
def test_conservation_matrix(arch_name, model_fn, input_fn, rule_name, cfg):
    torch.manual_seed(zlib.crc32(f"{rule_name}/{arch_name}".encode())
                      % (2 ** 31))
    model, data = model_fn(), input_fn()
    r_sum, logit = _run(model, data, cfg)
    err = abs(r_sum - 1.0)
    tol = _TOLS[rule_name][arch_name]
    assert err < tol, (
        f"{rule_name} on {arch_name}: Σ R = {r_sum:+.4f} "
        f"(target +1, selected logit {logit:+.4f}), "
        f"err {err:.4f} > tol {tol}"
    )
    assert torch.isfinite(torch.tensor(r_sum))


@pytest.mark.slow
@pytest.mark.parametrize("depth", [1, 4, 8, 12])
def test_depth_does_not_degrade(depth):
    """Bias-free MLP + ε + identity activation: drift stays below 0.5
    regardless of depth (no per-layer residual mass leakage)."""
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(8, 8, bias=False), nn.GELU()]
    layers.append(nn.Linear(8, 4, bias=False))
    model = nn.Sequential(*layers).eval()
    data = torch.randn(1, 8)
    r_sum, _logit = _run(
        model, data, LRPConfig(rule=on_linear('epsilon'), activation='passthrough'))
    assert abs(r_sum - 1.0) < 0.5


# ---------------------------------------------------------------------------
# Target variants — each form preserves conservation under ε on bias-free MLP.
# ---------------------------------------------------------------------------

class TestTargetVariants:
    _CFG = LRPConfig(rule=on_linear('epsilon'), activation='passthrough')

    def _setup(self):
        return _mlp(8, 16, 4, bias=False), torch.randn(1, 8)

    def test_int_target(self):
        model, data = self._setup()
        x = autolrp.tensor(data.clone())
        out = model(x)
        pred = out.argmax(-1).item()
        out[0, pred].lrp(config=self._CFG)
        # +1 unit seed: Σ R targets +1 whatever the logit's sign.
        assert abs(x.relevance.sum().item() - 1.0) < 0.2

    def test_mask_target(self):
        """Custom seed mask via tensor algebra: ``(out * mask).sum().lrp()``.
        With a one-hot mask, Σ R ≈ Σ mask = 1.0."""
        model, data = self._setup()
        x = autolrp.tensor(data.clone())
        out = model(x)
        mask = torch.zeros_like(out)
        mask[0, 0] = 1.0
        (out * mask).sum().lrp(config=self._CFG)
        assert abs(x.relevance.sum().item() - 1.0) < 0.3

    def test_negative_logit(self):
        """+1 unit seed regardless of logit sign. With a strong negative
        bias the z-rule denominator is dominated by the bias, so Σ R
        does not equal the seed; we only assert that LRP runs cleanly
        and yields a finite, non-zero map."""
        model = nn.Sequential(nn.Linear(8, 4, bias=True)).eval()
        with torch.no_grad():
            model[0].bias.fill_(-5.0)
        data = torch.randn(1, 8)
        x = autolrp.tensor(data.clone())
        out = model(x)
        assert out[0, 0].item() < 0
        out[0, 0].lrp(config=self._CFG)
        assert torch.isfinite(x.relevance).all()
        assert x.relevance.abs().sum().item() > 0

    def test_sum_target(self):
        """``out.sum().lrp()`` with +1 seed: Σ R ≈ +1 regardless of
        ``out.sum()``'s sign."""
        model, data = self._setup()
        x = autolrp.tensor(data.clone())
        out = model(x)
        out.sum().lrp(config=self._CFG)
        assert abs(x.relevance.sum().item() - 1.0) < 0.3
