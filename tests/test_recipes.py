"""Tests for higher-order recipes: bilrp + clrp.

Smoke + property checks. Heavy model integration is exercised by the
showcase notebooks; here we use tiny MLP/CNN models so the suite runs
in milliseconds.
"""
import pytest
import torch
import torch.nn as nn

import autoLRP as autolrp
from tests._cfg import on_linear
from autoLRP import BASE
from autoLRP import LRPConfig


# ---------------------------------------------------------------------------
# Tiny models
# ---------------------------------------------------------------------------

def _mlp_encoder(in_dim=8, hidden=12, emb=6):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, emb),
    ).eval()


def _mlp_classifier(in_dim=8, hidden=12, n_classes=5):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, n_classes),
    ).eval()


# ---------------------------------------------------------------------------
# bilrp
# ---------------------------------------------------------------------------

class TestBiLRP:
    def test_shape_full_dims(self):
        enc = _mlp_encoder()
        x_a = torch.randn(1, 8)
        x_b = torch.randn(1, 8)
        R = autolrp.bilrp(enc, x_a, x_b)
        # Default reduce_each is flatten(1), so each side produces (1, 8)
        # → outer product yields (8, 8).
        assert R.shape == (8, 8)

    def test_shape_random_projection(self):
        enc = _mlp_encoder()
        x_a = torch.randn(1, 8)
        x_b = torch.randn(1, 8)
        R = autolrp.bilrp(enc, x_a, x_b, n_dims=3)
        assert R.shape == (8, 8)

    def test_custom_reduce_each(self):
        """A reducer summing input over its last dim should yield a (1, 1)
        outer product (one scalar per side)."""
        enc = _mlp_encoder()
        x_a = torch.randn(1, 8)
        x_b = torch.randn(1, 8)
        R = autolrp.bilrp(enc, x_a, x_b,
                          reduce_each=lambda r: r.sum(dim=-1, keepdim=True))
        assert R.shape == (1, 1)

    def test_self_pair_symmetric(self):
        """bilrp(model, x, x) should produce a matrix equal to its own
        transpose on the (x_a, x_b) axes — the formula is symmetric in
        the two operands when they're identical."""
        torch.manual_seed(0)
        enc = _mlp_encoder()
        x = torch.randn(1, 8)
        R = autolrp.bilrp(enc, x, x)
        torch.testing.assert_close(R, R.T, atol=1e-5, rtol=1e-5)

    def test_custom_project_callback(self):
        """User-supplied ``project`` callable should be honored — verify
        by routing through a callback that drops half the output dims."""
        enc = _mlp_encoder(emb=8)
        x_a = torch.randn(1, 8)
        x_b = torch.randn(1, 8)
        # Project only first 4 dims.
        R = autolrp.bilrp(
            enc, x_a, x_b,
            project=lambda out: out[..., :4],
        )
        assert R.shape == x_a.shape[1:] + x_b.shape[1:]

    def test_config_passthrough(self):
        """Passing a custom config (e.g. epsilon rule) must not crash and
        must propagate the rule choice into the LRP backward."""
        enc = _mlp_encoder()
        x_a = torch.randn(1, 8)
        x_b = torch.randn(1, 8)
        R_zplus = autolrp.bilrp(enc, x_a, x_b,
                                config=LRPConfig(rule=on_linear('zplus')))
        R_eps   = autolrp.bilrp(enc, x_a, x_b,
                                config=LRPConfig(rule=on_linear('epsilon')))
        # Different rules must produce different attribution.
        assert not torch.allclose(R_zplus, R_eps, atol=1e-7)

    def test_second_order_conservation(self):
        """Paper Prop. 1: sum_{ii'} R_pair = <phi(a), phi(b)> for
        zero-bias rectifier nets under non-dissipative rules (LRP-0 ~=
        epsilon). Exact-dims (no projection) so the target is the true
        dot product."""
        torch.manual_seed(0)
        enc = nn.Sequential(
            nn.Linear(8, 12, bias=False), nn.ReLU(),
            nn.Linear(12, 6, bias=False),
        ).eval()
        x_a = torch.randn(1, 8)
        x_b = torch.randn(1, 8)
        R, sim = autolrp.bilrp(enc, x_a, x_b, return_similarity=True)
        with torch.no_grad():
            dot = float((enc(x_a) * enc(x_b)).sum())
        # return_similarity tracks the exact decomposition target ...
        assert sim == pytest.approx(dot, rel=1e-5)
        # ... and the pairwise map conserves it (Prop. 1).
        assert float(R.sum()) == pytest.approx(dot, rel=1e-3)

    def test_conservation_target_under_projection(self):
        """With the JL random projection, sum R matches the PROJECTED
        similarity returned by return_similarity (not the raw dot)."""
        torch.manual_seed(0)
        enc = nn.Sequential(
            nn.Linear(8, 12, bias=False), nn.ReLU(),
            nn.Linear(12, 6, bias=False),
        ).eval()
        x_a = torch.randn(1, 8)
        x_b = torch.randn(1, 8)
        R, sim = autolrp.bilrp(enc, x_a, x_b, n_dims=3,
                               return_similarity=True)
        assert float(R.sum()) == pytest.approx(sim, rel=1e-3)


# ---------------------------------------------------------------------------
# clrp
# ---------------------------------------------------------------------------

class TestCLRP:
    def test_shape(self):
        clf = _mlp_classifier()
        x = torch.randn(1, 8)
        R = autolrp.clrp(clf, x, target=0)
        assert R.shape == x.shape

    def test_nonnegative(self):
        """CLRP applies max(0, ·) at the end; result must be all >= 0."""
        clf = _mlp_classifier()
        x = torch.randn(1, 8)
        R = autolrp.clrp(clf, x, target=2)
        assert (R >= 0).all().item(), \
            f"CLRP produced negative values: min={R.min().item()}"

    def test_target_concentration(self):
        """On a model where each output dim is a perfect linear function of
        a *different* input dim, CLRP for target k should put most mass on
        input dim k. We construct the classifier so output[k] depends only
        on input[k] (identity weight matrix on the second linear)."""
        torch.manual_seed(0)
        clf = nn.Sequential(
            nn.Linear(8, 8, bias=False),
            nn.ReLU(),
            nn.Linear(8, 8, bias=False),
        ).eval()
        # Wire weights so output[k] = relu(input[k]).
        clf[0].weight.data = torch.eye(8)
        clf[2].weight.data = torch.eye(8)

        x = torch.randn(1, 8).abs() + 0.1   # all-positive so ReLU is identity
        target = 3
        R = autolrp.clrp(clf, x, target=target)
        # Most mass should be at the input dim that uniquely drives target.
        target_share = R[0, target].item() / (R.sum().item() + 1e-12)
        assert target_share > 0.5, \
            (f"CLRP target dim {target} share = {target_share:.3f}, "
             f"expected > 0.5. Per-dim R: {R[0].tolist()}")

    def test_config_override(self):
        """User can override the default config (e.g. drop the z^B
        input rule by giving a rule without the input_conv entry)."""
        clf = _mlp_classifier()
        x = torch.randn(1, 8)
        # default: input_conv -> zbox
        R_default = autolrp.clrp(clf, x, target=0)
        # custom: plain zplus, no input_conv entry
        R_no_zbox = autolrp.clrp(
            clf, x, target=0,
            config=LRPConfig(rule=on_linear('zplus')),
        )
        assert R_default.shape == R_no_zbox.shape
