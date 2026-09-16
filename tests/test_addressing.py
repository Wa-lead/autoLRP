"""Addressing: which entry reaches which node, and what runs there.

- A key is a node name without its version digit or a fact name; a fact
  entry wins over the name entry on the nodes that carry the fact.
- ``attribute`` is positional: ``('epsilon', {'attribute': 'rhs'})``
  attributes the operand written on the right of that op, always. The one
  virtual name ``'detach'`` takes ``by=<fact>`` and attributes the operand
  the fact does not name, resolved per node from the fact's value.
- Which operands of a product are attributed is decided by which reach
  the wrapped input (the facts ``bilinear`` and ``weight_operand``), not
  by the node's name.
- The fused attention node resolves each of its two products through a
  stand-in and gives the same relevance as the decomposed graph.
"""
import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import autolrp
from autolrp import (LRPConfig, BASE, CPLRP, ATTNLRP, UNIFORM, on, SOFTMAX_NODES,
                     set_decompose_attention, node_facts, walk, resolve)
from autolrp.backward import analysis
from autolrp.backward.install import _Named
from autolrp.backward.rules import PRODUCT_RULES, MUL_RULES


def _bmm_attn_model(transposed):
    """softmax(QK^T)@V with the AV product spelled either way."""
    torch.manual_seed(3)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(6, 6, bias=False)
            self.k = nn.Linear(6, 6, bias=False)
            self.v = nn.Linear(6, 6, bias=False)
            self.transposed = transposed

        def forward(self, x):
            q, k, v = self.q(x), self.k(x), self.v(x)
            A = torch.softmax(torch.bmm(q, k.transpose(-2, -1)) / 6 ** 0.5, dim=-1)
            if self.transposed:
                y = torch.bmm(v.transpose(-2, -1), A.transpose(-2, -1)).transpose(-2, -1)
            else:
                y = torch.bmm(A, v)
            return y.sum()
    return M().double().eval()


def _run(model, x_data, config):
    x = autolrp.tensor(x_data.clone())
    model(x).lrp(config=config)
    return x.relevance.clone()


X = torch.randn(1, 5, 6, dtype=torch.float64)


class TestVirtualDetach:
    def test_spelling_invariance_via_by(self):
        sd = _bmm_attn_model(False).state_dict()
        cfg = LRPConfig(rule={**BASE, **CPLRP}, eps=1e-12)
        ms = _bmm_attn_model(False); ms.load_state_dict(sd)
        mt = _bmm_attn_model(True); mt.load_state_dict(sd)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            Rs = _run(ms, X, cfg)
            Rt = _run(mt, X, cfg)
        assert torch.allclose(Rs, Rt, atol=1e-12)

    def test_concrete_names_are_positional(self):
        sd = _bmm_attn_model(False).state_dict()
        cfg = LRPConfig(rule={**BASE, 'BmmBackward': ('epsilon', {'attribute': 'rhs'})}, eps=1e-12)
        ms = _bmm_attn_model(False); ms.load_state_dict(sd)
        mt = _bmm_attn_model(True); mt.load_state_dict(sd)
        assert not torch.allclose(_run(ms, X, cfg), _run(mt, X, cfg), atol=1e-9)

    def test_virtual_equals_resolved_concrete_on_the_straight_spelling(self):
        sd = _bmm_attn_model(False).state_dict()
        m1 = _bmm_attn_model(False); m1.load_state_dict(sd)
        m2 = _bmm_attn_model(False); m2.load_state_dict(sd)
        Rv = _run(m1, X, LRPConfig(rule={**BASE, **CPLRP}, eps=1e-12))
        Rc = _run(m2, X, LRPConfig(rule={**BASE, 'BmmBackward': ('epsilon', {'attribute': 'rhs'})}, eps=1e-12))
        assert torch.allclose(Rv, Rc, atol=1e-12)

    def test_absent_fact_runs_the_family_fallback_quietly(self):
        """A 'detach' keyed by the node name reaches the score bmm,
        which carries no attention_weights fact; the table default
        (epsilon, both operands) runs there, silently, and the result
        equals the CPLRP fragment keyed by the bilinear fact."""
        sd = _bmm_attn_model(False).state_dict()
        m1 = _bmm_attn_model(False); m1.load_state_dict(sd)
        m2 = _bmm_attn_model(False); m2.load_state_dict(sd)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            R1 = _run(m1, X, LRPConfig(rule={**BASE, 'BmmBackward':
                                             ('detach', {'by': 'attention_weights'})}, eps=1e-12))
        R2 = _run(m2, X, LRPConfig(rule={**BASE, **CPLRP}, eps=1e-12))
        assert torch.allclose(R1, R2, atol=1e-12)

    def test_present_fact_that_is_not_a_side_is_an_error(self):
        m = nn.Conv2d(3, 4, 3)
        x = autolrp.tensor(torch.randn(1, 3, 6, 6))
        with pytest.raises(ValueError, match="not a side"):
            m(x).sum().lrp(config=LRPConfig(rule={
                **BASE, 'input_conv': ('detach', {'by': 'input_conv'}),
                'ConvolutionBackward': 'zplus'}))


class TestStatisticEntry:
    def test_rmsnorm_transparent_under_base(self):
        torch.manual_seed(0)

        class RMS(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(6, 6, bias=False)
                self.l2 = nn.Linear(6, 3, bias=False)

            def forward(self, t):
                h = self.l1(t)
                h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
                return self.l2(h)
        m = RMS().double().eval()
        x = autolrp.tensor(torch.randn(1, 6, dtype=torch.float64))
        m(x)[0, 0].lrp(config=LRPConfig(eps=1e-12))
        assert abs(x.relevance.sum().item() - 1.0) < 1e-8

    def test_overriding_the_entry_changes_the_map(self):
        torch.manual_seed(0)

        class RMS(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(6, 6, bias=False)

            def forward(self, t):
                h = self.l1(t)
                return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)).sum()
        m = RMS().double().eval()
        xd = torch.randn(1, 6, dtype=torch.float64)
        maps = {}
        for label, rule in (('detach', dict(BASE)),
                            ('prop', {**BASE, 'statistic_operand': 'proportional'})):
            x = autolrp.tensor(xd.clone())
            m(x).lrp(config=LRPConfig(rule=rule, eps=1e-12))
            maps[label] = x.relevance.clone()
        assert not torch.allclose(maps['detach'], maps['prop'], atol=1e-9)

    def test_fact_entry_outranks_the_family_entry(self):
        """rule={**BASE, 'MulBackward': 'proportional'} must not switch
        off the statistic transparency: the fact entry wins on tagged
        nodes, the family entry runs on the rest."""
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(6, 6, bias=False), nn.LayerNorm(6), nn.Linear(6, 3, bias=False)).double().eval()
        xd = torch.randn(1, 6, dtype=torch.float64)
        x1 = autolrp.tensor(xd.clone()); m(x1)[0, 0].lrp(config=LRPConfig(eps=1e-12))
        x2 = autolrp.tensor(xd.clone()); m(x2)[0, 0].lrp(config=LRPConfig(
            rule={**BASE, 'MulBackward': 'proportional', 'DivBackward': 'proportional'}, eps=1e-12))
        assert torch.allclose(x1.relevance, x2.relevance, atol=1e-12)


class TestResolve:
    def _node(self, out, name):
        plan = walk(out)
        analysis.run(plan)
        return next(n for n, _ in plan if name in n.name())

    def test_returns_the_function_and_kwargs(self):
        x = autolrp.tensor(torch.randn(2, 4)); y = autolrp.tensor(torch.randn(2, 4))
        out = (x * y).sum()
        node = self._node(out, 'MulBackward')
        fn, kw = resolve({**BASE, 'MulBackward': 'proportional'}, node)
        assert fn is MUL_RULES['proportional'] and kw == {}

    def test_detach_resolves_to_the_table_rule_on_the_other_side(self):
        m = _bmm_attn_model(False)
        x = autolrp.tensor(X.clone()); out = m(x)
        cfg = LRPConfig(rule={**BASE, **CPLRP})
        plan = walk(out); analysis.run(plan)
        tagged = [n for n, _ in plan if node_facts(n).get('attention_weights') == 0]
        assert len(tagged) == 1
        fn, kw = resolve(cfg.rule, tagged[0])
        assert fn is PRODUCT_RULES['epsilon'] and kw == {'attribute': 'rhs'}

    def test_wrong_family_names_entry_node_and_choices(self):
        x = autolrp.tensor(torch.randn(2, 4)); y = autolrp.tensor(torch.randn(2, 4))
        node = self._node((x * y).sum(), 'MulBackward')
        with pytest.raises(ValueError, match=r"entry 'MulBackward'='zbox'.*MulBackward0.*'proportional'"):
            resolve({**BASE, 'MulBackward': 'zbox'}, node)

    def test_no_entry_names_the_node_and_the_key_to_add(self):
        x = autolrp.tensor(torch.randn(2, 4)); y = autolrp.tensor(torch.randn(2, 4))
        node = self._node((x * y).sum(), 'MulBackward')
        with pytest.raises(ValueError, match="no entry for node 'MulBackward0'. Add 'MulBackward'"):
            resolve({'AddBackward': 'equal'}, node)

    def test_two_fact_entries_on_one_node_is_an_error(self):
        from autolrp import register_analyzer, ANALYZERS
        register_analyzer('also_weights')(
            lambda node: True if 'BmmBackward' in node.name() else None)
        try:
            m = _bmm_attn_model(False)
            x = autolrp.tensor(X.clone())
            with pytest.raises(ValueError, match="two entries address"):
                m(x).lrp(config=LRPConfig(rule={**BASE, 'attention_weights': 'gradient_input',
                                                'also_weights': 'epsilon'}))
        finally:
            ANALYZERS.pop('also_weights')


class TestProductFamilyByLiveness:
    """One operand from the input: a linear layer; two: bilinear."""

    def test_constant_weight_on_either_side_and_frozen_params(self):
        torch.manual_seed(0)
        W = torch.randn(3, 4, dtype=torch.float64)
        x = autolrp.tensor(torch.randn(2, 3, dtype=torch.float64)); (x @ W).sum().lrp()
        assert abs(x.relevance.sum().item() - 1.0) < 1e-9
        x = autolrp.tensor(torch.randn(2, 3, dtype=torch.float64)); (W.T @ x.T).sum().lrp()
        assert abs(x.relevance.sum().item() - 1.0) < 1e-9
        x = autolrp.tensor(torch.randn(1, 2, 3, dtype=torch.float64))
        torch.einsum('bij,jk->bik', x, W).sum().lrp()
        assert abs(x.relevance.sum().item() - 1.0) < 1e-9

    def test_frozen_equals_trainable_on_a_transformer_layer(self):
        torch.manual_seed(0)
        blk = nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True).double().eval()
        Xd = torch.randn(1, 5, 8, dtype=torch.float64)
        x1 = autolrp.tensor(Xd.clone()); blk(x1).sum().lrp(config=LRPConfig(rule={**BASE, **ATTNLRP}))
        for p in blk.parameters():
            p.requires_grad_(False)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            x2 = autolrp.tensor(Xd.clone()); blk(x2).sum().lrp(config=LRPConfig(rule={**BASE, **ATTNLRP}))
        assert torch.equal(x1.relevance, x2.relevance)

    def test_bmm_with_a_constant_weight_is_not_bilinear(self):
        torch.manual_seed(0)
        W = torch.randn(1, 3, 4, dtype=torch.float64)
        x = autolrp.tensor(torch.randn(1, 2, 3, dtype=torch.float64))
        torch.bmm(x, W).sum().lrp(config=LRPConfig(rule={**BASE, **UNIFORM}))   # the bilinear entry does not reach it
        assert abs(x.relevance.sum().item() - 1.0) < 1e-9

    def test_mm_with_both_operands_from_the_input_is_bilinear(self):
        torch.manual_seed(0)
        data = torch.randn(2, 3, dtype=torch.float64)
        x = autolrp.tensor(data.clone())
        (x @ x.T).sum().lrp(config=LRPConfig(rule={**BASE, **UNIFORM, 'MmBackward': 'zplus'}))
        r_uniform = x.relevance.clone()
        x = autolrp.tensor(data.clone())
        (x @ x.T).sum().lrp(config=LRPConfig(rule={**BASE, 'MmBackward': 'zplus'}))
        assert not torch.allclose(r_uniform, x.relevance)

    def test_a_plain_requires_grad_tensor_is_a_weight(self):
        x = autolrp.tensor(torch.randn(2, 3, dtype=torch.float64))
        y = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        (x @ y).sum().lrp()
        assert abs(x.relevance.sum().item() - 1.0) < 1e-9


class TestConstantOperands:
    @pytest.mark.parametrize('label, f', [
        ('-x', lambda x: -x), ('x*0.25', lambda x: x * 0.25), ('x/4', lambda x: x / 4),
        ('x*c', lambda x: x * torch.tensor([2.0, 2.0, 2.0])),
        ('2/x', lambda x: torch.tensor([2.0, 2.0, 2.0]) / x),
        ('1-x', lambda x: 1.0 - x), ('x-1', lambda x: x - 1.0),
        ('cumsum', lambda x: x.cumsum(-1)), ('norm', lambda x: x.norm()),
        ('sum dtype', lambda x: x.sum(dtype=torch.float64)),
    ])
    def test_all_relevance_reaches_the_live_operand(self, label, f):
        x = autolrp.tensor(torch.tensor([1.0, -2.0, 3.0]))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            f(x).sum().lrp()
        assert abs(x.relevance.sum().item() - 1.0) < 1e-5, label

    def test_a_constant_added_is_a_bias_and_keeps_its_share(self):
        x = autolrp.tensor(torch.tensor([1.0, -2.0, 3.0]))
        (x + torch.tensor([2.0, 2.0, 2.0])).sum().lrp()
        assert abs(x.relevance.sum().item() - 0.5) < 1e-5

    def test_add_with_alpha_goes_through_our_add(self):
        x = autolrp.tensor(torch.tensor([1.0, 2.0])); y = autolrp.tensor(torch.tensor([3.0, 4.0]))
        torch.add(x, y, alpha=2.0).sum().lrp()
        assert abs(x.relevance.sum().item() + y.relevance.sum().item() - 1.0) < 1e-6


class TestFusedStandIns:
    def test_stand_ins_answer_as_the_decomposed_nodes(self):
        real = _Named(object.__new__(object), 'X')  # facts of a bare object: none
        av = _Named(real, 'BmmBackward0', {'bilinear': True, 'attention_weights': 0})
        assert av.name() == 'BmmBackward0'
        assert node_facts(av) == {'bilinear': True, 'attention_weights': 0}
        assert node_facts(_Named(real, 'BmmBackward0')) == {}

    @pytest.mark.parametrize('cfg', [
        LRPConfig(rule={**BASE, **CPLRP}), LRPConfig(rule={**BASE, **ATTNLRP}), LRPConfig(rule={**BASE, **UNIFORM}),
        LRPConfig(rule={**BASE, 'attention_weights': 'gradient_input',
                        **on(SOFTMAX_NODES, 'jacobian')}),
    ])
    def test_fused_equals_decomposed_per_product(self, cfg):
        torch.manual_seed(0)
        d = 8
        Q = torch.randn(1, 4, 5, d, dtype=torch.float64)
        K = torch.randn(1, 2, 5, d, dtype=torch.float64)
        V = torch.randn(1, 2, 5, d, dtype=torch.float64)

        def run(dec):
            set_decompose_attention(dec)
            q = autolrp.tensor(Q.clone()); k = autolrp.tensor(K.clone()); v = autolrp.tensor(V.clone())
            F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).sum().lrp(config=cfg)
            return q.relevance.clone(), k.relevance.clone(), v.relevance.clone()
        try:
            dec, fus = run(True), run(False)
        finally:
            set_decompose_attention(True)
        for a, b in zip(dec, fus):
            assert torch.allclose(a, b, atol=1e-11)

    def test_score_side_relevance_is_not_scaled_by_the_constant(self):
        """Epsilon on both products, softmax passthrough: V gets half,
        Q plus K get the other half; the 1/sqrt(d) scale passes
        relevance through unchanged."""
        torch.manual_seed(0)
        d = 8
        q = autolrp.tensor(torch.randn(1, 2, 5, d, dtype=torch.float64))
        k = autolrp.tensor(torch.randn(1, 2, 5, d, dtype=torch.float64))
        v = autolrp.tensor(torch.randn(1, 2, 5, d, dtype=torch.float64))
        F.scaled_dot_product_attention(q, k, v).sum().lrp(
            config=LRPConfig(rule={**BASE, 'BmmBackward': 'epsilon'}))
        assert abs(v.relevance.sum().item() - 0.5) < 1e-6
        assert abs(q.relevance.sum().item() + k.relevance.sum().item() - 0.5) < 1e-6
