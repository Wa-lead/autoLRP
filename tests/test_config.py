"""LRPConfig: exact keys, checked values, printable BASE, presets.

Every case here was first run by hand; the numbers in comments are
what the library printed.
"""
import pytest
import torch
import torch.nn as nn

import autolrp
from autolrp import LRPConfig, BASE, register_analyzer, ANALYZERS
from autolrp.backward.rules import FAMILIES
from tests._cfg import on_linear


class TestBase:
    def test_one_entry_per_family_plus_the_statistic_entry(self):
        assert set(BASE) == set(FAMILIES) | {'statistic_operand'}
        assert BASE['statistic_operand'] == ('detach', {'by': 'statistic_operand'})

    def test_default_config_is_base_and_a_copy(self):
        cfg = LRPConfig()
        assert cfg.rule == BASE
        assert cfg.rule is not BASE

    def test_softmax_defaults_to_passthrough(self):
        assert LRPConfig().softmax == 'passthrough'


class TestKeys:
    def test_bare_string_is_rejected(self):
        with pytest.raises(TypeError, match="start from autolrp.BASE"):
            LRPConfig(rule='epsilon')

    def test_bare_tuple_is_rejected(self):
        with pytest.raises(TypeError):
            LRPConfig(rule=('gamma', {'gamma': 0.5}))

    def test_default_key_is_rejected_with_a_hint(self):
        with pytest.raises(ValueError, match="There is no 'default'"):
            LRPConfig(rule={**BASE, 'default': 'epsilon'})

    def test_alias_or_free_key_is_rejected(self):
        for key in ['linear', 'conv', 'Linear', 'Addmm', 'attention', 'ward']:
            with pytest.raises(ValueError, match="unknown rule key"):
                LRPConfig(rule={**BASE, key: 'zplus'})

    def test_key_with_version_digit_is_rejected(self):
        with pytest.raises(ValueError, match="version digit"):
            LRPConfig(rule={**BASE, 'MulBackward0': 'proportional'})

    def test_every_family_name_is_a_key(self):
        for name in FAMILIES:
            LRPConfig(rule={**BASE, name: next(iter(FAMILIES[name]))})

    def test_registered_fact_is_a_key(self):
        assert 'weights_operand' in ANALYZERS
        LRPConfig(rule={**BASE, 'weights_operand': 'uniform'})

    def test_a_fact_name_may_end_in_digits(self):
        register_analyzer('group_00')(lambda nodes: {})
        try:
            LRPConfig(rule={**BASE, 'group_00': 'zplus'})
        finally:
            ANALYZERS.pop('group_00')

    def test_unregistered_fact_is_rejected(self):
        with pytest.raises(ValueError, match="unknown rule key"):
            LRPConfig(rule={**BASE, 'my_fact': 'epsilon'})


class TestValues:
    def test_rule_the_family_cannot_run_is_rejected(self):
        with pytest.raises(ValueError, match="not a choice here"):
            LRPConfig(rule={**BASE, 'MulBackward': 'zbox'})

    def test_unknown_name_is_rejected(self):
        with pytest.raises(ValueError, match="not a choice here"):
            LRPConfig(rule={**BASE, 'AddmmBackward': 'not_a_rule'})

    def test_detach_needs_by(self):
        with pytest.raises(ValueError, match="needs by="):
            LRPConfig(rule={**BASE, 'BmmBackward': 'detach'})

    def test_by_must_be_a_registered_fact(self):
        with pytest.raises(ValueError, match="not a registered fact"):
            LRPConfig(rule={**BASE, 'BmmBackward': ('detach', {'by': 'nope'})})

    def test_detach_on_a_family_without_sides_is_rejected(self):
        with pytest.raises(ValueError, match="no side to detach"):
            LRPConfig(rule={**BASE, 'AddmmBackward':
                            ('detach', {'by': 'weights_operand'})})

    def test_tuple_with_kwargs_and_callable_are_accepted(self):
        LRPConfig(rule={**BASE, 'ConvolutionBackward': ('gamma', {'gamma': 0.25})})
        LRPConfig(rule={**BASE, 'ConvolutionBackward': lambda *a, **k: None})

    def test_unary_fields_take_exact_node_names_or_a_callable(self):
        LRPConfig(softmax={'SoftmaxBackward': 'jacobian',
                           'LogSoftmaxBackward': 'passthrough'})
        LRPConfig(activation=lambda node, cfg: None)
        with pytest.raises(ValueError, match="unknown softmax key"):
            LRPConfig(softmax={'softmax': 'jacobian'})
        with pytest.raises(ValueError, match="not a choice here"):
            LRPConfig(activation='nope')

    def test_frozen(self):
        cfg = LRPConfig()
        with pytest.raises(Exception):
            cfg.eps = 1.0


class TestPresets:
    def test_composite(self):
        cfg = LRPConfig.composite()
        assert cfg.rule == {**BASE, 'ConvolutionBackward': 'zplus'}

    def test_attnlrp(self):
        cfg = LRPConfig.attnlrp(gamma=0.25)
        assert cfg.rule['ConvolutionBackward'] == ('gamma', {'gamma': 0.25})
        assert cfg.rule['BmmBackward'] == 'epsilon'
        assert cfg.softmax == 'jacobian' and cfg.activation == 'yx'

    def test_bilrp_is_base(self):
        assert LRPConfig.bilrp().rule == BASE

    def test_cplrp_writes_two_explicit_entries(self):
        cfg = LRPConfig.cplrp()
        assert cfg.rule['weights_operand'] == ('detach', {'by': 'weights_operand'})
        assert cfg.rule['BmmBackward'] == 'epsilon'
        assert cfg.softmax == 'passthrough'
        assert cfg.rule == LRPConfig(attn='cplrp').rule

    def test_epsilon_alpha2_beta1(self):
        cfg = LRPConfig.epsilon_alpha2_beta1()
        ab = ('alpha_beta', {'alpha': 2.0, 'beta': 1.0})
        for k in ('AddmmBackward', 'MmBackward', 'ConvolutionBackward'):
            assert cfg.rule[k] == ab
        assert cfg.rule['BmmBackward'] == 'epsilon'

    def test_attn_replaces_the_base_entry(self):
        cfg = LRPConfig(rule={**BASE, 'ConvolutionBackward': 'zplus'}, attn='uniform')
        assert cfg.rule['BmmBackward'] == 'uniform'
        assert cfg.rule['ConvolutionBackward'] == 'zplus'

    def test_attn_conflicts_with_a_user_entry(self):
        with pytest.raises(ValueError, match="conflicts with attn"):
            LRPConfig(rule={**BASE, 'BmmBackward': 'uniform'}, attn='cplrp')

    def test_attn_accepts_the_same_entry(self):
        LRPConfig(rule={**BASE, 'BmmBackward': 'uniform'}, attn='uniform')

    def test_unknown_preset_and_softmax_conflict_raise(self):
        with pytest.raises(ValueError, match="Unknown attn preset"):
            LRPConfig(attn='nope')
        with pytest.raises(ValueError, match="conflicts with attn"):
            LRPConfig(attn='attnlrp', softmax='passthrough')


class TestPartialDict:
    def test_missing_family_errors_at_install_naming_the_key(self):
        m = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 1))
        x = autolrp.tensor(torch.randn(1, 6))
        with pytest.raises(ValueError, match="no entry for node 'AddmmBackward0'"):
            m(x).sum().lrp(config=LRPConfig(rule={'BmmBackward': 'uniform'}))


class TestPerOpDispatch:
    def test_conv_entry_changes_only_conv(self):
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(4, 2, bias=False),
        ).eval()
        x_data = torch.randn(1, 3, 8, 8)
        x_a = autolrp.tensor(x_data.clone())
        model(x_a)[0, 0].lrp(config=LRPConfig(
            rule={**BASE, 'ConvolutionBackward': ('gamma', {'gamma': 0.5})},
            activation='passthrough'))
        x_b = autolrp.tensor(x_data.clone())
        model(x_b)[0, 0].lrp(config=LRPConfig(activation='passthrough'))
        assert not torch.allclose(x_a.relevance, x_b.relevance, atol=1e-6)

    def test_rule_kwargs_propagate(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 4, bias=False)).eval()
        x_data = torch.randn(1, 8)
        from autolrp.backward import rules as _rules
        _rules._GAMMA_DEGENERATION_WARNED.clear()
        x1 = autolrp.tensor(x_data.clone())
        with pytest.warns(UserWarning, match="runs the epsilon rule instead"):
            model(x1)[0, 0].lrp(config=LRPConfig(
                rule=on_linear(('gamma', {'gamma': 0.0})), activation='passthrough'))
        x2 = autolrp.tensor(x_data.clone())
        model(x2)[0, 0].lrp(config=LRPConfig(
            rule=on_linear(('gamma', {'gamma': 0.5})), activation='passthrough'))
        assert not torch.allclose(x1.relevance, x2.relevance, atol=1e-6)

    def test_input_conv_fact_routes_zbox(self):
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.ReLU(),
            nn.Conv2d(4, 4, 3, padding=1, bias=False),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(4, 2, bias=False),
        ).eval()
        x_data = torch.randn(1, 3, 8, 8)
        x_plain = autolrp.tensor(x_data.clone())
        model(x_plain)[0, 0].lrp(config=LRPConfig(activation='passthrough'))
        x_zbox = autolrp.tensor(x_data.clone())
        model(x_zbox)[0, 0].lrp(config=LRPConfig(
            rule={**BASE, 'input_conv': ('zbox', {'low': 0.0, 'high': 1.0})},
            activation='passthrough'))
        assert not torch.allclose(x_zbox.relevance, x_plain.relevance, atol=1e-6)


class TestResidualAdd:
    def _residual_model(self):
        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.f = nn.Linear(8, 8, bias=False)

            def forward(self, x):
                return x + self.f(x)
        torch.manual_seed(0)
        return nn.Sequential(_Block(), nn.Linear(8, 4, bias=False)).eval()

    def test_three_add_rules_give_three_maps(self):
        torch.manual_seed(1)
        model = self._residual_model()
        x_data = torch.randn(1, 8)
        Rs = {}
        for label, spec in (('proportional', 'proportional'), ('equal', 'equal'),
                            ('0.2', ('fixed', {'p': 0.2}))):
            x = autolrp.tensor(x_data.clone())
            model(x)[0, 0].lrp(config=LRPConfig(rule={**BASE, 'AddBackward': spec}))
            Rs[label] = x.relevance.clone()
        for a, b in [('proportional', 'equal'), ('proportional', '0.2'), ('equal', '0.2')]:
            assert not torch.allclose(Rs[a], Rs[b], atol=1e-6)

    def test_equal_split_is_symmetric(self):
        torch.manual_seed(2)
        xa = autolrp.tensor(torch.randn(1, 4))
        xb = autolrp.tensor(torch.randn(1, 4))
        (xa + xb).sum().lrp(config=LRPConfig(rule={**BASE, 'AddBackward': 'equal'}))
        share_a = xa.relevance.abs().sum().item()
        share_b = xb.relevance.abs().sum().item()
        assert abs(share_a / (share_a + share_b) - 0.5) < 0.05
