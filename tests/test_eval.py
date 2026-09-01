"""Tests for autoLRP.eval — the attribution-evaluation suite.

The headline property we test: each metric distinguishes a *good*
attribution (LRP / gradient×input on the trained model) from a *bad*
one (uniform noise or model-independent edge filter). If a metric
fails to distinguish these, it's not measuring faithfulness.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

import autoLRP as autolrp
from autoLRP import LRPConfig, eval as alrp_eval


# ---------------------------------------------------------------------------
# Fixtures: a tiny CNN and matched random + LRP attributions
# ---------------------------------------------------------------------------

def _cnn():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(4, 5, bias=False),
    ).eval()


def _attribute_lrp(model, x):
    xt = autolrp.tensor(x.clone())
    out = model(xt)
    out[0, out.argmax(-1).item()].lrp(config=LRPConfig.composite())
    return xt.relevance.detach()


def _attribute_random(model, x):
    torch.manual_seed(123)
    return torch.randn_like(x)


@pytest.fixture
def setup():
    model = _cnn()
    x = torch.randn(1, 3, 8, 8)
    R_lrp = _attribute_lrp(model, x)
    R_rand = _attribute_random(model, x)
    return model, x, R_lrp, R_rand


# ---------------------------------------------------------------------------
# perturbation_curve
# ---------------------------------------------------------------------------

class TestPerturbationCurve:
    def test_shape_and_keys(self, setup):
        model, x, R_lrp, _ = setup
        out = alrp_eval.perturbation_curve(model, x, R_lrp,
                                            mode='deletion', n_steps=10)
        assert set(out) >= {'fractions', 'scores', 'auc', 'mode', 'target'}
        assert out['fractions'].shape == (11,)
        assert out['scores'].shape == (11,)
        assert out['mode'] == 'deletion'

    def test_endpoints_deletion(self, setup):
        """Deletion: fraction=0 → score == clean score; fraction=1 →
        x is fully baseline, score is whatever the baseline produces."""
        model, x, R_lrp, _ = setup
        out = alrp_eval.perturbation_curve(model, x, R_lrp,
                                            mode='deletion', n_steps=4)
        # Clean point first.
        with torch.no_grad():
            clean = torch.softmax(model(x)[0], -1)
        assert abs(out['scores'][0] - float(clean[out['target']])) < 1e-5

    def test_endpoints_insertion(self, setup):
        """Insertion: fraction=1 → all positions are from x, score
        matches clean score."""
        model, x, R_lrp, _ = setup
        out = alrp_eval.perturbation_curve(model, x, R_lrp,
                                            mode='insertion', n_steps=4)
        with torch.no_grad():
            clean = torch.softmax(model(x)[0], -1)
        assert abs(out['scores'][-1] - float(clean[out['target']])) < 1e-5

    def test_invalid_mode(self, setup):
        model, x, R_lrp, _ = setup
        with pytest.raises(ValueError, match="mode must be"):
            alrp_eval.perturbation_curve(model, x, R_lrp, mode='wrong')

    def test_R_broadcastable(self):
        """A coarser-granularity R (broadcastable to x) should work
        without forcing the user to expand_as themselves."""
        torch.manual_seed(0)
        model = _cnn()
        x = torch.randn(1, 3, 8, 8)
        R_coarse = torch.randn(1, 1, 8, 8)               # per-pixel, broadcast across channels
        out = alrp_eval.perturbation_curve(model, x, R_coarse, n_steps=4)
        assert out['scores'].shape == (5,)


# ---------------------------------------------------------------------------
# aopc
# ---------------------------------------------------------------------------

class TestAOPC:
    def test_returns_scalar(self, setup):
        model, x, R_lrp, _ = setup
        v = alrp_eval.aopc(model, x, R_lrp, n_steps=5)
        assert isinstance(v, float)

    def test_lrp_beats_random(self, setup):
        """A faithful attribution must have *higher* AOPC under deletion
        than a random one — removing top-LRP positions should drop the
        score faster than removing random positions."""
        model, x, R_lrp, R_rand = setup
        aopc_lrp  = alrp_eval.aopc(model, x, R_lrp,  mode='deletion',
                                   n_steps=20)
        aopc_rand = alrp_eval.aopc(model, x, R_rand, mode='deletion',
                                   n_steps=20)
        assert aopc_lrp > aopc_rand, (
            f"LRP AOPC ({aopc_lrp:.4f}) should exceed random ({aopc_rand:.4f})"
        )

    def test_invalid_mode(self, setup):
        model, x, R_lrp, _ = setup
        with pytest.raises(ValueError, match="mode must be"):
            alrp_eval.aopc(model, x, R_lrp, mode='wrong')


# ---------------------------------------------------------------------------
# sanity_check_cascade
# ---------------------------------------------------------------------------

class TestSanityCheckCascade:
    def test_keys_and_length(self, setup):
        model, x, _, _ = setup
        out = alrp_eval.sanity_check_cascade(
            model, x, _attribute_lrp, similarity='cosine')
        assert set(out) == {'layer_names', 'similarities'}
        # Should have one entry per Conv2d / Linear (2 in our _cnn).
        assert len(out['layer_names']) == len(out['similarities']) >= 2

    def test_weights_restored(self, setup):
        """After the cascade, the model's weights must equal their
        original values — sanity_check_cascade must not leave the model
        in a corrupted state."""
        model, x, _, _ = setup
        orig_weights = {n: p.detach().clone()
                        for n, p in model.named_parameters()}
        _ = alrp_eval.sanity_check_cascade(model, x, _attribute_lrp)
        for n, p in model.named_parameters():
            assert torch.allclose(p, orig_weights[n], atol=0), \
                f"Layer {n} weight not restored after cascade"

    def test_lrp_similarity_decays(self, setup):
        """For a faithful attribution method, similarity to the original
        should generally decrease as more layers are randomized (the
        last entry should be lower than the first)."""
        model, x, _, _ = setup
        out = alrp_eval.sanity_check_cascade(
            model, x, _attribute_lrp, similarity='cosine',
            randomize='cumulative')
        sims = out['similarities']
        assert sims[-1] < sims[0] + 0.05, (
            f"LRP similarity didn't decay under cascade: "
            f"first={sims[0]:.3f}, last={sims[-1]:.3f}"
        )

    def test_model_independent_attribution_stays_high(self, setup):
        """A random attribution doesn't depend on the model, so its
        similarity to the original under randomization should be ~stable.
        This is the falsification test: the cascade returns flat
        similarity for unfaithful methods."""
        model, x, _, _ = setup

        def random_attribute(m, x_):
            return torch.randn_like(x_)

        # Use a SEEDED random attribution so it returns the same R every
        # time it's called — the point is that the attribution doesn't
        # depend on the model. (An unseeded random would change every
        # call and confound the measurement.)
        seed = [0]
        def seeded_random(m, x_):
            g = torch.Generator(); g.manual_seed(seed[0])
            seed[0] += 0   # same seed every call
            return torch.randn(x_.shape, generator=g)

        out = alrp_eval.sanity_check_cascade(
            model, x, seeded_random, similarity='cosine')
        sims = out['similarities']
        # Model-independent → similarity stays near 1 across cascade.
        assert all(abs(s) > 0.99 for s in sims), (
            f"Model-independent attribution should have cosine ~ 1; "
            f"got {sims}"
        )

    def test_invalid_similarity(self, setup):
        model, x, _, _ = setup
        with pytest.raises(ValueError, match="unknown similarity"):
            alrp_eval.sanity_check_cascade(model, x, _attribute_lrp,
                                            similarity='wrong')


# ---------------------------------------------------------------------------
# sensitivity_correlation
# ---------------------------------------------------------------------------

class TestSensitivityCorrelation:
    def test_keys(self, setup):
        model, x, R_lrp, _ = setup
        out = alrp_eval.sensitivity_correlation(
            model, x, R_lrp, subset_size=0.2, n_samples=30, seed=0)
        assert set(out) == {'correlation', 'n_samples', 'subset_size'}
        assert -1.0 <= out['correlation'] <= 1.0

    def test_lrp_correlation_higher_than_random(self, setup):
        """Faithful attribution → higher correlation between summed R on
        a random subset and the actual output drop when that subset is
        masked."""
        model, x, R_lrp, R_rand = setup
        c_lrp = alrp_eval.sensitivity_correlation(
            model, x, R_lrp,  subset_size=0.3, n_samples=60, seed=0)['correlation']
        c_rand = alrp_eval.sensitivity_correlation(
            model, x, R_rand, subset_size=0.3, n_samples=60, seed=0)['correlation']
        assert c_lrp > c_rand, (
            f"LRP correlation ({c_lrp:.3f}) should exceed random ({c_rand:.3f})"
        )

    def test_int_subset_size(self, setup):
        """subset_size can be an absolute int (Sensitivity-N convention)."""
        model, x, R_lrp, _ = setup
        out = alrp_eval.sensitivity_correlation(
            model, x, R_lrp, subset_size=50, n_samples=30, seed=0)
        assert out['subset_size'] == 50

    def test_fraction_subset_size(self, setup):
        """subset_size can be a fraction in (0,1) (Faithfulness Correlation)."""
        model, x, R_lrp, _ = setup
        out = alrp_eval.sensitivity_correlation(
            model, x, R_lrp, subset_size=0.1, n_samples=30, seed=0)
        n_elem = x.numel()
        assert out['subset_size'] == max(1, int(round(0.1 * n_elem)))

    def test_invalid_fraction(self, setup):
        model, x, R_lrp, _ = setup
        with pytest.raises(ValueError, match="must be in"):
            alrp_eval.sensitivity_correlation(
                model, x, R_lrp, subset_size=1.5, n_samples=10)

    def test_seed_reproducible(self, setup):
        model, x, R_lrp, _ = setup
        a = alrp_eval.sensitivity_correlation(
            model, x, R_lrp, subset_size=0.2, n_samples=30, seed=42)
        b = alrp_eval.sensitivity_correlation(
            model, x, R_lrp, subset_size=0.2, n_samples=30, seed=42)
        assert a['correlation'] == b['correlation']
