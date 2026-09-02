"""explain(): which entry and which rule ran at every node."""
import warnings

import torch
import torch.nn as nn

import autolrp
from autolrp import LRPConfig, BASE, explain, explain_summary, register_analyzer, ANALYZERS


def _layer():
    torch.manual_seed(0)
    return nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True).eval()


class TestExplain:
    def test_rows_name_the_entry_and_the_function(self):
        blk = _layer()
        x = autolrp.tensor(torch.randn(1, 5, 8))
        rows = explain(blk(x)[0, -1, 0], LRPConfig(attn='cplrp'))
        by = {}
        for node, key, what in rows:
            by.setdefault((node, key, what), 0)
            by[(node, key, what)] += 1
        assert by[('AddmmBackward0', 'AddmmBackward', 'epsilon')] == 4
        assert by[('BmmBackward0', 'weights_operand', 'detach_lhs_bmm')] == 1
        assert by[('BmmBackward0', 'BmmBackward', 'epsilon_bmm')] == 1
        assert by[('SoftmaxBackward', None, 'softmax=passthrough')] == 1
        assert by[('DivBackward0', None, 'passthrough (constant operand)')] == 1
        assert 'AddmmBackward0' in explain_summary(rows)

    def test_explain_leaves_no_trace(self):
        blk = _layer()
        X = torch.randn(1, 5, 8)
        x1 = autolrp.tensor(X.clone()); out = blk(x1)
        explain(out[0, -1, 0], LRPConfig(attn='attnlrp'))
        out[0, -1, 0].lrp(config=LRPConfig(attn='attnlrp'))
        x2 = autolrp.tensor(X.clone()); blk(x2)[0, -1, 0].lrp(config=LRPConfig(attn='attnlrp'))
        assert torch.equal(x1.relevance, x2.relevance)


class TestAnalyzerValues:
    def test_a_bare_value_is_the_fact_value(self):
        register_analyzer('every_linear')(
            lambda nodes: {n: True for n in nodes if 'AddmmBackward' in n.name()})
        try:
            m = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 1))
            x = autolrp.tensor(torch.randn(1, 6))
            rows = explain(m(x).sum(), LRPConfig(rule={**BASE, 'every_linear': 'zplus'}))
            assert [r for r in rows if r[0] == 'AddmmBackward0'] == \
                [('AddmmBackward0', 'every_linear', 'zplus')] * 2
        finally:
            ANALYZERS.pop('every_linear')


class TestUnmatchedWarning:
    def test_only_nodes_on_the_input_path_warn(self):
        """An embedding lookup of ids reaches no wrapped input; its
        unmatched node must not warn. An unmatched op on the input path
        still does."""
        emb = nn.Embedding(10, 4)
        x = autolrp.tensor(torch.randn(1, 3, 4))
        pos = emb(torch.arange(3))[None]
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            (x + pos).sum().lrp()
        x = autolrp.tensor(torch.randn(1, 3, 4))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            torch.special.erf(x).sum().lrp()
        assert any('no installer matched' in str(m.message) for m in w)
