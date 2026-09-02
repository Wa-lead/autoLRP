"""Strategy / plan / installer dispatch.

Scope:
- Graph coverage: every grad_fn produced by common op forwards resolves
  to an installer (no NO_MATCH on the supported op set).
- ``match_installer`` semantics: substring collisions resolve in
  most-specific-first order (LeakyReLU vs ReLU, LogSoftmax vs Softmax,
  Bmm vs Mm).
- ``is_shape_node`` recognizes the shape-routing grad_fn set exactly.
- Plan determinism: same forward graph → same plan.
- ``merge`` composition: overrides win, base insertion order preserved.
- Custom strategy override: user installers fire instead of the default.
"""
import pytest
import torch
import torch.nn as nn

import autolrp
from autolrp import LRPConfig, EXPLICIT_STRATEGY, INSTALLERS, merge, register_installer, installer
from autolrp.backward.strategies import match_installer, is_shape_node
from autolrp.backward.install import install_passthrough, install_noop


# ---------------------------------------------------------------------------
# Graph coverage — every grad_fn we walk on the supported op set must
# resolve to a non-None installer.
# ---------------------------------------------------------------------------

class TestGraphCoverage:
    def _walk(self, model, data):
        x = autolrp.tensor(data)
        out = model(x)
        sel = out[0, 0] if out.ndim > 1 else out[0]
        return autolrp.walk(sel, strategy=EXPLICIT_STRATEGY, config=LRPConfig())

    def _assert_no_unhandled(self, plan):
        missing = [(n, i) for n, i in plan
                   if i is None and 'AccumulateGrad' not in n.name()]
        assert not missing, f"no-installer nodes: {[n.name() for n, _ in missing]}"

    def test_mlp(self):
        model = nn.Sequential(
            nn.Linear(8, 16), nn.GELU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4),
        ).eval()
        self._assert_no_unhandled(self._walk(model, torch.randn(1, 8)))

    def test_cnn(self):
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(16, 4),
        ).eval()
        self._assert_no_unhandled(self._walk(model, torch.randn(1, 3, 8, 8)))

    def test_attention(self):
        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.mha = nn.MultiheadAttention(16, 4, batch_first=True, bias=False)
                self.ln = nn.LayerNorm(16)
                self.head = nn.Linear(16, 4, bias=False)
            def forward(self, x):
                y = self.ln(x)
                z, _ = self.mha(y, y, y, need_weights=False)
                return self.head((x + z)[:, 0])

        self._assert_no_unhandled(self._walk(Attn().eval(), torch.randn(1, 6, 16)))

    @pytest.mark.parametrize("act_cls", [
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid,
        nn.Hardtanh, nn.Hardswish, nn.Hardsigmoid, nn.ELU, nn.SELU,
        nn.CELU, nn.Softplus, nn.Softsign, nn.LogSigmoid, nn.Mish,
    ])
    def test_every_activation_has_installer(self, act_cls):
        model = nn.Sequential(nn.Linear(4, 4), act_cls(), nn.Linear(4, 2)).eval()
        plan = self._walk(model, torch.randn(1, 4))
        needle = act_cls.__name__.lower().replace('_', '')
        for node, installer in plan:
            if needle in node.name().lower():
                assert installer is not None, (
                    f"{act_cls.__name__}: no installer for {node.name()}"
                )


# ---------------------------------------------------------------------------
# match_installer — first-match-wins on substring collisions.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grad_fn_name,expected_pattern", [
    # Substring collisions: more-specific must match first.
    ('LeakyReluBackward0',  'LeakyReluBackward'),
    ('LogSoftmaxBackward0', 'LogSoftmaxBackward'),
    ('BmmBackward0',        'BmmBackward'),
])
def test_match_installer_substring_collision(grad_fn_name, expected_pattern):
    pattern, _ = match_installer(grad_fn_name)
    assert pattern == expected_pattern


def test_match_installer_unknown_returns_none():
    pattern, installer = match_installer('TotallyMadeUpBackward')
    assert pattern is None
    assert installer is None


# ---------------------------------------------------------------------------
# is_shape_node
# ---------------------------------------------------------------------------

def test_is_shape_node_classification():
    shape = ['ViewBackward0', 'ReshapeBackward0', 'TransposeBackward0',
             'PermuteBackward0', 'SqueezeBackward0', 'UnsqueezeBackward0',
             'ExpandBackward0', 'CatBackward0', 'StackBackward0',
             'SplitBackward0', 'NarrowBackward0', 'SliceBackward0',
             'IndexBackward0', 'SelectBackward0', 'AliasBackward0',
             'CloneBackward0', 'TBackward0', 'AsStridedBackward0']
    compute = ['AddmmBackward0', 'ConvolutionBackward0', 'BmmBackward0',
               'ReluBackward0', 'GeluBackward0', 'SoftmaxBackward0',
               'AddBackward0', 'MulBackward0']
    misrouted = ([n for n in shape if not is_shape_node(n)]
                 + [n for n in compute if is_shape_node(n)])
    assert not misrouted, f"shape/compute misclassified: {misrouted}"


# ---------------------------------------------------------------------------
# Plan determinism
# ---------------------------------------------------------------------------

def test_same_forward_same_plan():
    """Walking the same graph twice produces the same (node, installer) sequence."""
    model = nn.Sequential(
        nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4),
    ).eval()
    data = torch.randn(1, 8)

    def build_plan():
        x = autolrp.tensor(data.clone())
        out = model(x)
        return autolrp.walk(out[0, 0], strategy=EXPLICIT_STRATEGY,
                            config=LRPConfig())

    p1, p2 = build_plan(), build_plan()
    assert [n.name() for n, _ in p1] == [n.name() for n, _ in p2]
    assert ([i.__name__ if i else None for _, i in p1]
            == [i.__name__ if i else None for _, i in p2])


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

class TestMerge:
    def test_base_order_preserved(self):
        """LeakyRelu must come before Relu after merge — substring-collision protection."""
        base = {
            'LeakyReluBackward': install_passthrough,
            'ReluBackward':      install_passthrough,
        }
        merged = merge(base, {'ReluBackward': install_noop})
        keys = list(merged.keys())
        assert keys.index('LeakyReluBackward') < keys.index('ReluBackward')
        assert merged['ReluBackward'] is install_noop


# ---------------------------------------------------------------------------
# Custom-strategy override actually fires
# ---------------------------------------------------------------------------

def test_custom_installer_is_called():
    """A user-supplied installer in a merged strategy fires at runtime."""
    called = {'n': 0}

    def my_install(node, config):
        def _hook(gi, go, _c=called):
            _c['n'] += 1
            return (go[0],) + gi[1:]
        return node.register_hook(_hook)

    custom = merge(EXPLICIT_STRATEGY, {'ReluBackward': my_install})

    model = nn.Sequential(
        nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4),
    ).eval()
    x = autolrp.tensor(torch.randn(1, 8))
    out = model(x)
    autolrp.graph_lrp(out[0, 0], strategy=custom, config=LRPConfig())

    assert called['n'] >= 1, "custom ReLU installer never fired"


# ---------------------------------------------------------------------------
# register_installer + @installer decorator (public extension API)
# ---------------------------------------------------------------------------

class TestRegisterInstaller:
    def test_register_installer_global(self):
        """Calling ``register_installer('Pat', fn)`` makes the global
        registry pick ``fn`` for any node whose ``grad_fn.name()``
        contains ``'Pat'``. We register under a unique key so we don't
        clobber a real installer."""
        called = {'n': 0}
        def my_install(node, config):
            called['n'] += 1
            return None
        unique_key = 'ZZ_FakeBackward_For_Test'
        try:
            register_installer(unique_key, my_install)
            assert INSTALLERS[unique_key] is my_install
            # match_installer treats the key as a substring pattern.
            _, fn = match_installer(unique_key + '0', INSTALLERS)
            assert fn is my_install
        finally:
            INSTALLERS.pop(unique_key, None)

    def test_decorator_equivalent(self):
        """``@installer('Pat')`` produces the same registry effect as
        ``register_installer('Pat', fn)``."""
        unique_key = 'ZZ_DecoratorBackward_For_Test'
        try:
            @installer(unique_key)
            def my_install(node, config):
                return None
            assert INSTALLERS[unique_key] is my_install
        finally:
            INSTALLERS.pop(unique_key, None)
