"""LRPConfig: exact keys, checked values, printable BASE, presets.

Every case here was first run by hand; the numbers in comments are
what the library printed.
"""
import pytest
import torch
import torch.nn as nn

import autolrp
from autolrp import (LRPConfig, BASE, on, CPLRP, ATTNLRP, UNIFORM, ELEMENTWISE_NODES,
                     SOFTMAX_NODES, register_analyzer, ANALYZERS)
from autolrp.backward.rules import RULES_FOR
from tests._cfg import on_linear


class TestBase:
    def test_one_entry_per_node_name_plus_the_statistic_entry(self):
        assert set(BASE) == set(RULES_FOR) | {'statistic_operand'}
        assert BASE['statistic_operand'] == ('detach', {'by': 'statistic_operand'})

    def test_every_entry_is_its_tables_default(self):
        assert all(BASE[name] == table.default for name, table in RULES_FOR.items())

    def test_default_config_is_base_and_a_copy(self):
        cfg = LRPConfig()
        assert cfg.rule == BASE
        assert cfg.rule is not BASE

    def test_one_operand_defaults(self):
        assert BASE['SoftmaxBackward'] == 'passthrough'
        assert BASE['NativeLayerNormBackward'] == 'identity'
        assert BASE['ReluBackward'] == 'passthrough'
        assert BASE['MeanBackward'] == 'proportional'


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

    def test_every_node_name_is_a_key(self):
        for name in RULES_FOR:
            LRPConfig(rule={**BASE, name: next(iter(RULES_FOR[name]))})

    def test_registered_fact_is_a_key(self):
        assert 'attention_weights' in ANALYZERS
        LRPConfig(rule={**BASE, 'attention_weights': 'gradient_input'})

    def test_a_fact_name_may_end_in_digits(self):
        register_analyzer('group_00')(lambda node: None)
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

    def test_detach_on_a_table_without_sides_is_rejected(self):
        with pytest.raises(ValueError, match="no side to detach"):
            LRPConfig(rule={**BASE, 'SoftmaxBackward':
                            ('detach', {'by': 'attention_weights'})})

    def test_attribute_is_checked(self):
        LRPConfig(rule={**BASE, 'BmmBackward': ('zplus', {'attribute': 'rhs'})})
        with pytest.raises(ValueError, match="attribute must be"):
            LRPConfig(rule={**BASE, 'BmmBackward': ('zplus', {'attribute': 'left'})})

    def test_tuple_with_kwargs_and_callable_are_accepted(self):
        LRPConfig(rule={**BASE, 'ConvolutionBackward': ('gamma', {'gamma': 0.25})})
        LRPConfig(rule={**BASE, 'ConvolutionBackward': lambda *a, **k: None})

    def test_one_operand_nodes_are_ordinary_entries(self):
        frag = on(SOFTMAX_NODES, 'jacobian')
        assert frag == {'LogSoftmaxBackward': 'jacobian', 'SoftmaxBackward': 'jacobian'}
        cfg = LRPConfig(rule={**BASE, **frag})
        assert cfg.rule['SoftmaxBackward'] == 'jacobian'
        assert cfg.rule['MulBackward'] == BASE['MulBackward']
        LRPConfig(rule={**BASE, 'ReluBackward': lambda x, y, R_out, eps, **kw: R_out})
        with pytest.raises(ValueError, match="unknown rule key"):
            LRPConfig(rule={**BASE, 'softmax': 'jacobian'})
        with pytest.raises(ValueError, match="not a choice here"):
            LRPConfig(rule={**BASE, 'ReluBackward': 'nope'})

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
        assert cfg.rule['SoftmaxBackward'] == 'jacobian'
        assert all(cfg.rule[n] == 'yx' for n in ELEMENTWISE_NODES)

    def test_bilrp_is_base(self):
        assert LRPConfig.bilrp().rule == BASE

    def test_cplrp_writes_the_bilinear_entry(self):
        cfg = LRPConfig.cplrp()
        assert cfg.rule['bilinear'] == ('detach', {'by': 'attention_weights'})
        assert cfg.rule['BmmBackward'] == 'epsilon'
        assert cfg.rule['SoftmaxBackward'] == 'passthrough'
        assert cfg.rule == LRPConfig(rule={**BASE, **CPLRP}).rule

    def test_epsilon_alpha2_beta1(self):
        cfg = LRPConfig.epsilon_alpha2_beta1()
        ab = ('alpha_beta', {'alpha': 2.0, 'beta': 1.0})
        for k in ('AddmmBackward', 'MmBackward', 'ConvolutionBackward'):
            assert cfg.rule[k] == ab
        assert cfg.rule['BmmBackward'] == 'epsilon'

    def test_a_fragment_overrides_only_its_keys(self):
        cfg = LRPConfig(rule={**BASE, 'ConvolutionBackward': 'zplus', **UNIFORM})
        changed = {k for k in cfg.rule if cfg.rule[k] != BASE.get(k)}
        assert changed == {'ConvolutionBackward', 'bilinear'}
        assert cfg.rule['bilinear'] == 'gradient_input'

    def test_the_last_entry_for_a_key_wins(self):
        assert LRPConfig(rule={**BASE, **UNIFORM, **CPLRP}).rule['bilinear'] == CPLRP['bilinear']
        assert LRPConfig(rule={**BASE, **CPLRP, **UNIFORM}).rule['bilinear'] == 'gradient_input'

    def test_attnlrp_fragment_touches_attention_only(self):
        merged = {**BASE, **ATTNLRP}
        assert {k for k in merged if merged[k] != BASE[k]} == set(SOFTMAX_NODES)


class TestPartialDict:
    def test_missing_family_errors_at_install_naming_the_key(self):
        m = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 1))
        x = autolrp.tensor(torch.randn(1, 6))
        with pytest.raises(ValueError, match="no entry for node 'SumBackward'. Add 'SumBackward'"):
            m(x).sum().lrp(config=LRPConfig(rule={'BmmBackward': 'epsilon'}))     # reductions need an entry too


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
            rule={**BASE, 'ConvolutionBackward': ('gamma', {'gamma': 0.5})}))
        x_b = autolrp.tensor(x_data.clone())
        model(x_b)[0, 0].lrp(config=LRPConfig())
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
                rule=on_linear(('gamma', {'gamma': 0.0}))))
        x2 = autolrp.tensor(x_data.clone())
        model(x2)[0, 0].lrp(config=LRPConfig(
            rule=on_linear(('gamma', {'gamma': 0.5}))))
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
        model(x_plain)[0, 0].lrp(config=LRPConfig())
        x_zbox = autolrp.tensor(x_data.clone())
        model(x_zbox)[0, 0].lrp(config=LRPConfig(
            rule={**BASE, 'input_conv': ('zbox', {'low': 0.0, 'high': 1.0})}))
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
