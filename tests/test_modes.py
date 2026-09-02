"""End-to-end mode-flag behavior.

Flipping a config (``LRPConfig.composite()``, ``activation='yx'``,
``softmax='detach'/'jacobian'``) must produce an observably different
heatmap than the default. This is end-to-end coverage of the
strategy/installer wiring; the underlying math is in ``test_rules.py``.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import autolrp
from tests._cfg import on_linear
from autolrp import BASE
from autolrp import LRPConfig


# ---------------------------------------------------------------------------
# Composite rule routing
# ---------------------------------------------------------------------------

def test_composite_differs_from_epsilon_on_cnn():
    """``LRPConfig.composite()`` puts zplus on Conv → different from plain epsilon."""
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(8, 4),
    ).eval()
    data = torch.randn(1, 3, 8, 8)

    def run(cfg):
        x = autolrp.tensor(data.clone())
        model(x)[0, 0].lrp(config=cfg)
        return x.relevance.clone()

    r_eps = run(LRPConfig(rule=on_linear('epsilon')))
    r_cmp = run(LRPConfig.composite())
    assert not torch.allclose(r_eps, r_cmp, atol=1e-5)


# ---------------------------------------------------------------------------
# Activation y/x rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("act_cls", [nn.GELU, nn.SiLU, nn.ELU])
def test_activation_yx_differs_from_passthrough(act_cls):
    """``activation='yx'`` produces a different heatmap than ``'passthrough'``."""
    lin = nn.Linear(8, 8, bias=False).eval()
    act = act_cls()
    data = torch.randn(1, 8)

    def run(mode):
        x = autolrp.tensor(data.clone())
        act(lin(x)).sum().lrp(config=LRPConfig(activation=mode))
        return x.relevance.clone()

    r_pt, r_yx = run('passthrough'), run('yx')
    assert torch.isfinite(r_pt).all() and torch.isfinite(r_yx).all()
    assert not torch.allclose(r_pt, r_yx, atol=1e-5)


def test_relu_yx_zeros_dead_preactivations():
    """Under ``activation='yx'``, positions where ReLU's preactivation ≤ 0
    receive no relevance (verified through a near-identity linear)."""
    lin = nn.Linear(16, 16, bias=False).eval()
    with torch.no_grad():
        lin.weight.copy_(torch.eye(16) * 2.0)

    data = torch.randn(1, 16)
    x = autolrp.tensor(data.clone())
    F.relu(lin(x)).sum().lrp(config=LRPConfig(activation='yx'))

    dead = (data[0] <= 0)
    assert x.relevance[0][dead].abs().max().item() < 1e-4


# ---------------------------------------------------------------------------
# Softmax mode dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode_a,mode_b", [
    ('passthrough', 'detach'),
    ('passthrough', 'jacobian'),
    ('detach',      'jacobian'),
])
def test_softmax_modes_distinct(mode_a, mode_b):
    """The three softmax modes produce mutually different heatmaps."""
    lin1 = nn.Linear(8, 4, bias=False).eval()
    lin2 = nn.Linear(4, 2, bias=False).eval()
    data = torch.randn(1, 8)

    def run(mode):
        x = autolrp.tensor(data.clone())
        out = lin2(torch.softmax(lin1(x), dim=-1))
        out[0, 0].lrp(config=LRPConfig(softmax=mode))
        return x.relevance.clone()

    assert not torch.allclose(run(mode_a), run(mode_b), atol=1e-6)


def test_softmax_jacobian_zeros_dead_positions():
    """Forced -inf positions → softmax ≈ 0 → R ≈ 0 there."""
    data = torch.randn(1, 8)
    data[0, 0] = data[0, 1] = -1e30
    x = autolrp.tensor(data.clone())
    torch.softmax(x, dim=-1).sum().lrp(config=LRPConfig(softmax='jacobian'))
    assert abs(x.relevance[0, 0].item()) < 1e-6
    assert abs(x.relevance[0, 1].item()) < 1e-6
