"""autolrp.metrics: conservation is exact where the theory says so, the
perturbation curves start and end where they must, and a ranking that is
provably optimal beats a random one."""
import pytest
import torch
import torch.nn as nn

import autolrp
from autolrp import LRPConfig, metrics


def _explained(model, data):
    x = autolrp.tensor(data.clone())
    model(x)[0, 0].lrp()
    return x, (lambda t: model(t)[0, 0])


class TestConservation:
    def test_bias_free_is_exact(self):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(8, 16, bias=False), nn.ReLU(), nn.Linear(16, 4, bias=False)).eval()
        x, _ = _explained(m, torch.randn(1, 8))
        assert abs(metrics.conservation(x) - 1.0) < 1e-5

    def test_bias_absorbs(self):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
        x, _ = _explained(m, torch.randn(1, 8))
        assert metrics.conservation(x) != pytest.approx(1.0, abs=1e-3)

    def test_needs_relevance(self):
        with pytest.raises(ValueError, match="no .relevance"):
            metrics.conservation(autolrp.tensor(torch.randn(1, 8)))


class TestCurves:
    @pytest.fixture
    def setup(self):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Tanh(), nn.Linear(8, 3, bias=False)).eval()
        x, score = _explained(m, torch.randn(1, 8))
        return x, score

    def test_shape_and_endpoints(self, setup):
        x, score = setup
        score = lambda t, _s=score: _s(t).detach()
        base = torch.zeros_like(x.detach())
        d = metrics.perturbation_curve(x, score, mode='deletion', n_steps=5)
        i = metrics.perturbation_curve(x, score, mode='insertion', n_steps=5)
        assert len(d['scores']) == 6 and len(d['fractions']) == 6
        assert d['scores'][0] == pytest.approx(float(score(x.detach())), abs=1e-6)
        assert d['scores'][-1] == pytest.approx(float(score(base)), abs=1e-6)
        assert i['scores'][0] == pytest.approx(float(score(base)), abs=1e-6)
        assert i['scores'][-1] == pytest.approx(float(score(x.detach())), abs=1e-6)

    def test_invalid_mode(self, setup):
        x, score = setup
        with pytest.raises(ValueError, match="mode"):
            metrics.perturbation_curve(x, score, mode='wrong')

    def test_R_override_equals_relevance_path(self, setup):
        x, score = setup
        a = metrics.aopc(x, score, n_steps=4)
        b = metrics.aopc(x, score, R=x.relevance, n_steps=4)
        assert a == pytest.approx(b)

    @pytest.mark.parametrize("baseline", ['zero', 'mean', 0.5, torch.full((1, 8), 0.25)])
    def test_baselines(self, setup, baseline):
        x, score = setup
        assert isinstance(metrics.aopc(x, score, baseline=baseline, n_steps=3), float)

    def test_score_must_be_scalar(self, setup):
        x, _ = setup
        with pytest.raises(ValueError, match="one number"):
            metrics.perturbation_curve(x, lambda t: t, n_steps=2)


class TestAOPC:
    def test_optimal_ranking_beats_random(self):
        # Positive linear model: R = x * w exactly, so removing the largest
        # contributions first is the steepest possible drop at every step.
        torch.manual_seed(0)
        lin = nn.Linear(16, 1, bias=False)
        with torch.no_grad():
            lin.weight.abs_()
        m = lin.eval()
        data = torch.rand(1, 16) + 0.1
        x, score = _explained(m, data)
        best = metrics.aopc(x, score, mode='deletion', n_steps=8)
        rand = metrics.aopc(x, score, R=torch.rand(1, 16), mode='deletion', n_steps=8)
        assert best >= rand - 1e-7

    def test_insertion_is_positive_for_optimal_ranking(self):
        torch.manual_seed(1)
        lin = nn.Linear(16, 1, bias=False)
        with torch.no_grad():
            lin.weight.abs_()
        x, score = _explained(lin.eval(), torch.rand(1, 16) + 0.1)
        assert metrics.aopc(x, score, mode='insertion', n_steps=8) > 0


def test_faithfulness_keys():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 8, bias=False), nn.ReLU(), nn.Linear(8, 2, bias=False)).eval()
    x, score = _explained(m, torch.randn(1, 8))
    out = metrics.faithfulness(x, score, n_steps=4)
    assert set(out) == {'conservation', 'aopc_deletion', 'aopc_insertion'}
    assert out['conservation'] == pytest.approx(1.0, abs=1e-5)
