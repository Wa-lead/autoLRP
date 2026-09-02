"""Cross-library parity: autoLRP vs zennit, autoLRP vs captum.

Goal: same model + same input + same rule should produce numerically
matching attributions across implementations. Where exact agreement
holds (bias-free MLP under ε / z⁺), we assert it tightly. Where
implementation conventions diverge (CNN with average pooling; captum's
seed-by-logit-value vs autoLRP's signed-unit seed), we assert the
weaker invariant of cosine-direction agreement.

Skipped when ``zennit`` / ``captum`` aren't installed — establishes
the ``pytest.importorskip`` precedent for optional-dep tests.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import autolrp
from tests._cfg import on_linear
from autolrp import BASE
from autolrp import LRPConfig

zennit = pytest.importorskip(
    'zennit', reason='zennit required for parity tests')


# ---------------------------------------------------------------------------
# Models — bias-free for cleanest cross-library comparability.
# ---------------------------------------------------------------------------

def _mlp():
    return nn.Sequential(
        nn.Linear(8, 16, bias=False), nn.ReLU(),
        nn.Linear(16, 4, bias=False),
    ).eval()


def _cnn():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(8, 4, bias=False),
    ).eval()


def _cosine(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


# ---------------------------------------------------------------------------
# autoLRP runner — slice the target logit so the +1 seed at out[0, k] is
# routed back through SelectBackward as a one-hot mask (matching zennit's
# ``grad_outputs=target_oh`` convention).
# ---------------------------------------------------------------------------

def _autolrp_R(model, x_data, target_idx, rule):
    x = autolrp.tensor(x_data.clone())
    out = model(x)
    out[0, target_idx].lrp(
        config=LRPConfig(rule=on_linear(rule), activation='passthrough'),
    )
    return x.relevance[0].detach().clone()


# ---------------------------------------------------------------------------
# zennit runner — LayerMapComposite with explicit Linear/Convolution rules.
# ---------------------------------------------------------------------------

def _zennit_R(model, x_data, target_idx, rule_cls, rule_kwargs=None):
    """Zennit attribution with Linear/Convolution ε rules + ReLU Pass.

    The ``Pass`` rule on ``nn.ReLU`` matches autoLRP's
    ``install_passthrough`` for ReluBackward. Without it, zennit falls
    back to autograd-native ReLU backward (mask by ``input > 0``)
    while autoLRP passes R unchanged — and the two libraries diverge
    when an averaging op (AvgPool) sits between ReLU and the next ε
    site, because the uniform broadcast spreads R across positions
    where the autograd-native mask would have zeroed it.
    """
    from zennit.composites import LayerMapComposite
    from zennit.types import Linear, Convolution
    from zennit.rules import Pass
    from zennit.attribution import Gradient

    rule_kwargs = rule_kwargs or {}
    composite = LayerMapComposite([
        (Linear,      rule_cls(**rule_kwargs)),
        (Convolution, rule_cls(**rule_kwargs)),
        (nn.ReLU,     Pass()),
    ])
    attribution = Gradient(model=model, composite=composite)

    out = model(x_data)
    target_oh = torch.zeros_like(out.detach())
    target_oh[0, target_idx] = 1.0
    _, R = attribution(x_data.clone(), target_oh)
    return R[0].detach().clone()


# ===========================================================================
# Bit-exact parity (bias-free MLP)
# ---------------------------------------------------------------------------
# autoLRP's epsilon and z⁺ kernels match zennit's to float-precision when
# the architecture is bias-free, the activation is passthrough, and the
# eps stabilizer is the same (1e-11). Any drift here is a real regression.
# ===========================================================================

class TestZennitParityBitExact:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_mlp_epsilon(self, seed):
        from zennit.rules import Epsilon
        torch.manual_seed(seed)
        model = _mlp()
        x_data = torch.randn(1, 8)
        R_auto = _autolrp_R(model, x_data, target_idx=0, rule='epsilon')
        R_zen  = _zennit_R(model,  x_data, target_idx=0,
                           rule_cls=Epsilon, rule_kwargs={'epsilon': 1e-11})
        torch.testing.assert_close(R_auto, R_zen, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_mlp_zplus(self, seed):
        from zennit.rules import ZPlus
        torch.manual_seed(seed)
        model = _mlp()
        x_data = torch.randn(1, 8)
        R_auto = _autolrp_R(model, x_data, target_idx=0, rule='zplus')
        R_zen  = _zennit_R(model,  x_data, target_idx=0, rule_cls=ZPlus)
        torch.testing.assert_close(R_auto, R_zen, atol=1e-5, rtol=1e-5)


# ===========================================================================
# CNN bit-exact parity — also achievable once ReLU convention is matched.
# ---------------------------------------------------------------------------
# Empirically (see commit history): the AvgPool-between-Conv-and-Linear case
# disagrees by cos≈0.7 if zennit's composite has no rule for nn.ReLU
# (zennit falls back to autograd-native ReLU mask). Adding ``Pass()`` for
# nn.ReLU — which is what _zennit_R does — recovers bit-exact agreement.
# ===========================================================================

class TestZennitParityBitExactCNN:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_cnn_epsilon(self, seed):
        from zennit.rules import Epsilon
        torch.manual_seed(seed)
        model = _cnn()
        x_data = torch.randn(1, 3, 8, 8)
        R_auto = _autolrp_R(model, x_data, target_idx=0, rule='epsilon')
        R_zen  = _zennit_R(model,  x_data, target_idx=0,
                           rule_cls=Epsilon, rule_kwargs={'epsilon': 1e-11})
        torch.testing.assert_close(R_auto, R_zen, atol=1e-5, rtol=1e-5)


# ===========================================================================
# captum parity — direction match with scale difference
# ---------------------------------------------------------------------------
# captum's LRP seeds the output at the target's logit value (scalar = output
# value); autoLRP with a one-hot mask seeds at +1. Result: same heatmap
# direction (cosine = 1.0), captum's magnitudes scaled by the logit.
# ===========================================================================

@pytest.fixture(scope='module')
def captum_lrp():
    captum = pytest.importorskip(
        'captum', reason='captum required for captum-parity tests')
    from captum.attr import LRP
    return LRP


class TestCaptumParity:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_mlp_epsilon_direction(self, seed, captum_lrp):
        torch.manual_seed(seed)
        model = _mlp()
        x_data = torch.randn(1, 8, requires_grad=True)
        R_auto = _autolrp_R(model, x_data, target_idx=0, rule='epsilon')
        R_cap  = captum_lrp(model).attribute(
            x_data, target=0,
        )[0].detach().clone()
        # captum scales R by the *signed* target logit (R_cap = logit · grad_LRP).
        # autoLRP with a one-hot mask seeds at +1 unconditionally. So R_cap
        # = ±k · R_auto depending on sign(logit) — direction agreement is
        # |cosine| ≈ 1, not cosine ≈ 1.
        assert abs(_cosine(R_auto, R_cap)) > 0.99
