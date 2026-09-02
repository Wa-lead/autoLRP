"""Silent-degradation is banned: matched-but-uninstallable ops warn."""
import warnings

import torch
import torch.nn as nn

import autolrp as A
from tests._cfg import on_linear
from autolrp import BASE
from autolrp import LRPConfig
from autolrp.backward import install as INST


class TestFrozenParamTrap:
    def test_frozen_params_give_the_same_relevance(self):
        """Frozen weights (requires_grad=False) used to leave the input
        unsaved on the linear nodes; the intercept now makes every
        weight live, so the rule runs and the relevance is identical."""
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 3)).eval()
        X = torch.randn(1, 8)
        x = A.tensor(X.clone())
        m(x)[0, 0].lrp(config=LRPConfig(rule=on_linear('epsilon')))
        r_trainable = x.relevance.clone()
        for p in m.parameters():
            p.requires_grad_(False)
        INST._MISSING_STATE_WARNED.clear()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x = A.tensor(X.clone())
            m(x)[0, 0].lrp(config=LRPConfig(rule=on_linear('epsilon')))
        assert not [str(m_.message) for m_ in w]
        assert torch.equal(x.relevance, r_trainable)

    def test_healthy_model_does_not_warn(self):
        m = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 3)).eval()
        INST._MISSING_STATE_WARNED.clear()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x = A.tensor(torch.randn(1, 8))
            m(x)[0, 0].lrp(config=LRPConfig(rule=on_linear('epsilon')))
        assert not any('degrades to plain gradient' in str(m_.message)
                       for m_ in w)
