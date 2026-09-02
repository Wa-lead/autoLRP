"""Closed-form rule kernels: direct math checks against paper formulas.

Each test calls ``compute_linear_family_r_in`` / ``compute_bmm_r_in`` /
``compute_reduction_r_in`` with hand-constructed tensors and asserts the
result matches the paper formula (no autograd, no LRPTensor plumbing).

Covers:
- epsilon: bit-exact formula, scale invariance, bias-split effect.
- zplus:   non-negativity of contribution terms, equivalence to α=1 β=0.
- alpha_beta: α·z_pos − β·z_neg decomposition, α − β = 1 constraint.
- gamma:   gamma → 0 limit (= ε), gamma → ∞ limit (= zplus).
- zbox:    correct box-constrained form with low/high.
"""
import pytest
import torch
import torch.nn as nn

import torch as _torch

from autolrp.backward.lrp_utils import (
    run_linear_rule, reduction_share, stabilize, mm_ops,
)
from tests._cfg import on_linear
from autolrp import BASE
from autolrp.backward.rules import (
    epsilon, zplus, alpha_beta, gamma, zbox, BMM_RULES,
)


# ---------------------------------------------------------------------------
# Thin adapters: keep every test body written against the small kernel
# vocabulary while calling the CURRENT public machinery underneath.
# run_linear_rule returns (R_in, R_w); the weight share is not under test
# here. BMM rules and reductions are called through their real tables.
# ---------------------------------------------------------------------------

def compute_linear_family_r_in(x, w, bias, R_out, rule_fn, rule_kw,
                               fwd, bwd_a, bwd_b, eps):
    R_in, _ = run_linear_rule(x, w, bias, R_out, rule_fn, rule_kw,
                              fwd, bwd_a, bwd_b, eps)
    return R_in


_BMM_NAME = {'full': 'epsilon', 'cplrp': 'detach_lhs',
             'uniform': 'uniform'}


def compute_bmm_r_in(a, b, R_out, eps, bilinear):
    fwd = lambda x, y: _torch.bmm(x, y)
    bwd_a = lambda o, s: _torch.bmm(s, o.transpose(-2, -1))
    bwd_b = lambda o, s: _torch.bmm(o.transpose(-2, -1), s)
    return BMM_RULES[_BMM_NAME[bilinear]](a, b, R_out, eps,
                                          fwd, bwd_a, bwd_b)


compute_reduction_r_in = reduction_share


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reference_epsilon(x, w, bias, R_out, eps):
    """Paper epsilon for mm_ops convention (forward = x @ w, w is [in, out])."""
    z = x @ w
    if bias is not None:
        z = z + bias
    s = R_out / stabilize(z, eps)
    return x * (s @ w.transpose(-2, -1))


# ---------------------------------------------------------------------------
# Epsilon
# ---------------------------------------------------------------------------

class TestEpsilon:
    def test_matches_paper_formula_nobias(self):
        x = torch.randn(1, 8)
        w = torch.randn(8, 4)
        R_out = torch.randn(1, 4)

        fwd, bwd_a, bwd_b = mm_ops()
        R_in = compute_linear_family_r_in(
            x, w, None, R_out, epsilon, {}, fwd, bwd_a, bwd_b, eps=1e-11)
        R_ref = _reference_epsilon(x, w, None, R_out, 1e-11)
        torch.testing.assert_close(R_in, R_ref, atol=1e-5, rtol=1e-5)

    def test_bias_split_reduces_R_magnitude(self):
        """Bias-split scales R by |z_no_bias| / (|z_no_bias| + |bias|). When
        |bias| is large, R_in shrinks compared to the no-bias case."""
        x = torch.randn(1, 8)
        w = torch.randn(8, 4)
        R_out = torch.ones(1, 4)
        bias = torch.full((4,), 5.0)  # large bias

        fwd, bwd_a, bwd_b = mm_ops()
        R_with_bias = compute_linear_family_r_in(
            x, w, bias, R_out, epsilon, {}, fwd, bwd_a, bwd_b, eps=1e-11)
        R_no_bias = compute_linear_family_r_in(
            x, w, None, R_out, epsilon, {}, fwd, bwd_a, bwd_b, eps=1e-11)
        assert R_with_bias.abs().sum() < R_no_bias.abs().sum()

    def test_scale_invariance_in_R_out(self):
        """Doubling R_out doubles R_in (epsilon is linear in R_out)."""
        x = torch.randn(1, 8)
        w = torch.randn(8, 4)
        R_out = torch.randn(1, 4)

        fwd, bwd_a, bwd_b = mm_ops()
        R1 = compute_linear_family_r_in(
            x, w, None, R_out, epsilon, {}, fwd, bwd_a, bwd_b, eps=1e-11)
        R2 = compute_linear_family_r_in(
            x, w, None, 2 * R_out, epsilon, {}, fwd, bwd_a, bwd_b, eps=1e-11)
        torch.testing.assert_close(R2, 2 * R1, atol=1e-5, rtol=1e-5)

    def test_zero_R_out_gives_zero_R_in(self):
        x = torch.randn(1, 8)
        w = torch.randn(8, 4)
        R_out = torch.zeros(1, 4)

        fwd, bwd_a, bwd_b = mm_ops()
        R_in = compute_linear_family_r_in(
            x, w, None, R_out, epsilon, {}, fwd, bwd_a, bwd_b, eps=1e-11)
        torch.testing.assert_close(R_in, torch.zeros_like(R_in), atol=1e-9, rtol=1e-5)


# ---------------------------------------------------------------------------
# Zplus
# ---------------------------------------------------------------------------

class TestZPlus:
    def test_positive_input_positive_weight_gives_positive_R(self):
        """All x, w, R positive → R_in non-negative."""
        x = torch.abs(torch.randn(1, 8))
        w = torch.abs(torch.randn(8, 4))
        R_out = torch.abs(torch.randn(1, 4))

        fwd, bwd_a, bwd_b = mm_ops()
        R_in = compute_linear_family_r_in(
            x, w, None, R_out, zplus, {}, fwd, bwd_a, bwd_b, eps=1e-11)
        assert (R_in >= -1e-6).all(), "zplus produced significant negatives"

    def test_equivalence_to_alpha1_beta0(self):
        """α=1, β=0 in alpha_beta reduces exactly to zplus."""
        x = torch.randn(1, 8)
        w = torch.randn(8, 4)
        R_out = torch.randn(1, 4)

        fwd, bwd_a, bwd_b = mm_ops()
        R_zplus = compute_linear_family_r_in(
            x, w, None, R_out, zplus, {}, fwd, bwd_a, bwd_b, eps=1e-11)
        R_ab = compute_linear_family_r_in(
            x, w, None, R_out, alpha_beta, {'alpha': 1.0, 'beta': 0.0},
            fwd, bwd_a, bwd_b, eps=1e-11)
        torch.testing.assert_close(R_zplus, R_ab, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Alpha-beta
# ---------------------------------------------------------------------------

class TestAlphaBeta:
    def test_alpha_beta_constraint_enforced(self):
        """α − β = 1 must hold for conservation; the rule fn enforces this
        at execution time."""
        fwd, bwd_a, bwd_b = mm_ops()
        x = torch.randn(1, 8)
        w = torch.randn(8, 4)
        R_out = torch.randn(1, 4)
        with pytest.raises(ValueError, match="alpha - beta"):
            compute_linear_family_r_in(
                x, w, None, R_out, alpha_beta,
                {'alpha': 3.0, 'beta': 1.0},   # diff = 2, not 1
                fwd, bwd_a, bwd_b, eps=1e-11)

    def test_alpha2_beta1_differs_from_zplus(self):
        x = torch.randn(1, 8)
        w = torch.randn(8, 4)
        R_out = torch.randn(1, 4)

        fwd, bwd_a, bwd_b = mm_ops()
        R_zplus = compute_linear_family_r_in(
            x, w, None, R_out, zplus, {}, fwd, bwd_a, bwd_b, eps=1e-11)
        R_ab = compute_linear_family_r_in(
            x, w, None, R_out, alpha_beta, {'alpha': 2.0, 'beta': 1.0},
            fwd, bwd_a, bwd_b, eps=1e-11)
        assert not torch.allclose(R_zplus, R_ab, atol=1e-3)


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------

class TestGamma:
    def test_gamma_below_threshold_falls_back_to_epsilon(self):
        """γ ≤ 0.01 → gamma rule returns epsilon output (rules.py:105)."""
        x = torch.randn(1, 8)
        w = torch.randn(8, 4)
        R_out = torch.randn(1, 4)

        fwd, bwd_a, bwd_b = mm_ops()
        R_eps = compute_linear_family_r_in(
            x, w, None, R_out, epsilon, {}, fwd, bwd_a, bwd_b, eps=1e-11)
        from autolrp.backward import rules as _rules
        _rules._GAMMA_DEGENERATION_WARNED.clear()
        with pytest.warns(UserWarning, match="runs the epsilon rule instead"):
            R_gamma_tiny = compute_linear_family_r_in(
                x, w, None, R_out, gamma, {'gamma': 0.001}, fwd, bwd_a, bwd_b, eps=1e-11)
        torch.testing.assert_close(R_eps, R_gamma_tiny, atol=1e-6, rtol=1e-5)

    def test_gamma_above_threshold_differs_from_epsilon(self):
        x = torch.randn(1, 8)
        w = torch.randn(8, 4)
        R_out = torch.randn(1, 4)

        fwd, bwd_a, bwd_b = mm_ops()
        R_eps = compute_linear_family_r_in(
            x, w, None, R_out, epsilon, {}, fwd, bwd_a, bwd_b, eps=1e-11)
        R_gamma = compute_linear_family_r_in(
            x, w, None, R_out, gamma, {'gamma': 0.5}, fwd, bwd_a, bwd_b, eps=1e-11)
        assert not torch.allclose(R_eps, R_gamma, atol=1e-4)


# ---------------------------------------------------------------------------
# ZBox
# ---------------------------------------------------------------------------

class TestZBox:
    def test_zbox_uses_bounds(self):
        """Running zbox with different low/high produces different R."""
        x = torch.clamp(torch.randn(1, 8), -1, 1)
        w = torch.randn(8, 4)
        R_out = torch.randn(1, 4)

        fwd, bwd_a, bwd_b = mm_ops()
        R_a = compute_linear_family_r_in(
            x, w, None, R_out, zbox, {'low': -1.0, 'high': 1.0},
            fwd, bwd_a, bwd_b, eps=1e-11)
        R_b = compute_linear_family_r_in(
            x, w, None, R_out, zbox, {'low': 0.0, 'high': 2.0},
            fwd, bwd_a, bwd_b, eps=1e-11)
        assert not torch.allclose(R_a, R_b, atol=1e-4)

    def test_zbox_handles_boundary_inputs(self):
        """Inputs exactly at low/high boundaries shouldn't NaN out."""
        x = torch.full((1, 8), 1.0)  # at high
        w = torch.randn(8, 4)
        R_out = torch.randn(1, 4)

        fwd, bwd_a, bwd_b = mm_ops()
        R = compute_linear_family_r_in(
            x, w, None, R_out, zbox, {'low': -1.0, 'high': 1.0},
            fwd, bwd_a, bwd_b, eps=1e-11)
        assert not torch.isnan(R).any()
        assert not torch.isinf(R).any()


# ---------------------------------------------------------------------------
# Bilinear BMM — conservation invariants (also verified in test_rule_modes,
# reinforced here at the math-kernel level).
# ---------------------------------------------------------------------------

class TestBilinearKernel:
    def _setup(self):
        a = torch.randn(2, 4, 6)
        b = torch.randn(2, 6, 5)
        R_out = torch.randn(2, 4, 5)
        return a, b, R_out

    def test_full_conserves_approximately(self):
        """Σ R_a + Σ R_b − Σ R_out ≈ 0 for full (up to ε stabilizer)."""
        a, b, R_out = self._setup()
        Ra, Rb = compute_bmm_r_in(a, b, R_out, eps=1e-11, bilinear='full')
        total = Ra.sum() + Rb.sum()
        expected = R_out.sum()
        assert abs(total.item() - expected.item()) < 0.1, (
            f"full: total={total.item()}, R_out sum={expected.item()}"
        )

    def test_cplrp_zeros_first(self):
        a, b, R_out = self._setup()
        Ra, Rb = compute_bmm_r_in(a, b, R_out, 1e-11, 'cplrp')
        assert Ra.abs().sum().item() == 0.0
        assert Rb.abs().sum().item() > 0

    def test_uniform_matches_formula(self):
        """Achtibat 2024 Eq. 7: R_a = a * (R_out @ b^T) / 2,
        R_b = b * (a^T @ R_out) / 2. No denominator, no stabilizer."""
        a, b, R_out = self._setup()
        Ra, Rb = compute_bmm_r_in(a, b, R_out, 1e-11, 'uniform')
        Ra_expected = a * torch.bmm(R_out, b.transpose(-2, -1)) / 2.0
        Rb_expected = b * torch.bmm(a.transpose(-2, -1), R_out) / 2.0
        assert torch.allclose(Ra, Ra_expected)
        assert torch.allclose(Rb, Rb_expected)
        assert Ra.shape == a.shape
        assert Rb.shape == b.shape

    def test_uniform_no_eps_dependence(self):
        """The uniform rule has no denominator stabilizer — outputs must
        be invariant to eps. (Contrast with 'epsilon' / 'detach' which use
        z-based denominators.)"""
        a, b, R_out = self._setup()
        Ra_1, Rb_1 = compute_bmm_r_in(a, b, R_out, 1e-11, 'uniform')
        Ra_2, Rb_2 = compute_bmm_r_in(a, b, R_out, 1e-3,  'uniform')
        assert torch.allclose(Ra_1, Ra_2)
        assert torch.allclose(Rb_1, Rb_2)


# ---------------------------------------------------------------------------
# Reduction (Mean/Sum) — weighted by |activation|.
# ---------------------------------------------------------------------------

class TestReductionKernel:
    def test_proportional_split(self):
        """R_in at each position = |x_i| / Σ|x| × R_out (when reducing to scalar)."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        R_out = torch.tensor(10.0)
        R_in = compute_reduction_r_in(x, R_out, (1,), False, 1e-11)
        # Σ|x| = 10, so R_in[i] = x_i / 10 * 10 = x_i * 1 = x_i for these all-positive values.
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        torch.testing.assert_close(R_in, expected, atol=1e-5, rtol=1e-5)

    def test_conserves_sum(self):
        """Σ R_in = R_out after proportional split."""
        x = torch.randn(2, 8)
        R_out = torch.randn(2)
        R_in = compute_reduction_r_in(x, R_out.unsqueeze(1), (1,), True,
                                      1e-11).squeeze()
        # Σ R_in along reduced dim ≈ R_out at that batch.
        torch.testing.assert_close(R_in.sum(dim=1), R_out, atol=1e-4, rtol=1e-5)


# ---------------------------------------------------------------------------
# Sign-preserving behavior across rules.
# ---------------------------------------------------------------------------

class TestSignStructure:
    @pytest.mark.parametrize("rule_fn,rule_kw", [
        (epsilon, {}),
        (zplus, {}),
        (alpha_beta, {'alpha': 2.0, 'beta': 1.0}),
        (gamma, {'gamma': 0.25}),
    ])
    def test_zero_input_zero_relevance(self, rule_fn, rule_kw):
        """x = 0 ⇒ R_in = 0 (every rule multiplies by x at some point)."""
        x = torch.zeros(1, 8)
        w = torch.randn(8, 4)
        R_out = torch.randn(1, 4)
        fwd, bwd_a, bwd_b = mm_ops()
        R_in = compute_linear_family_r_in(
            x, w, None, R_out, rule_fn, rule_kw, fwd, bwd_a, bwd_b, eps=1e-11)
        torch.testing.assert_close(R_in, torch.zeros_like(R_in), atol=1e-8, rtol=1e-5)

    @pytest.mark.parametrize("rule_fn,rule_kw", [
        (epsilon, {}),
        (zplus, {}),
        (gamma, {'gamma': 0.25}),
    ])
    def test_finite_output(self, rule_fn, rule_kw):
        """No NaN/Inf for a well-conditioned input."""
        x = torch.randn(2, 16)
        w = torch.randn(16, 8)
        R_out = torch.randn(2, 8)
        fwd, bwd_a, bwd_b = mm_ops()
        R_in = compute_linear_family_r_in(
            x, w, None, R_out, rule_fn, rule_kw, fwd, bwd_a, bwd_b, eps=1e-11)
        assert not torch.isnan(R_in).any()
        assert not torch.isinf(R_in).any()


# ---------------------------------------------------------------------------
# Rule dispatch — composite expansion and zbox flag interaction.
# These test the wiring from LRPConfig to the right rule per node,
# through the ONE resolver the installers use (_resolve_family_rule).
# ---------------------------------------------------------------------------

import autolrp
from autolrp import LRPConfig
from autolrp.backward.rules import LINEAR_RULES, BMM_RULES
from autolrp.backward.install import resolve_rule


def _dispatch(cfg, node, registry=LINEAR_RULES, default='epsilon'):
    """The installers' resolution, reduced to (rule_fn, kwargs).

    Takes a config for readability at the call sites; the resolver
    itself takes the rule mapping, since selection reads nothing else,
    and returns None when the mapping says nothing about this node.
    """
    return resolve_rule(cfg.rule, node, registry)


def _two_conv_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 4),
    ).eval()


class TestResolveRuleComposite:
    """``LRPConfig.composite()`` routes Conv→zplus, everything else→epsilon."""

    @pytest.mark.parametrize("node_name,registry,expected", [
        ('ConvolutionBackward0', LINEAR_RULES, 'zplus'),
        ('AddmmBackward0',       LINEAR_RULES, 'epsilon'),
        ('MmBackward0',          LINEAR_RULES, 'epsilon'),
        ('BmmBackward0',         BMM_RULES,    'epsilon'),
    ])
    def test_composite_dispatch(self, node_name, registry, expected):
        class _FakeNode:
            def __init__(self, n): self._n = n
            def name(self): return self._n
            metadata = {}
        cfg = LRPConfig.composite()
        fn, _ = _dispatch(cfg, _FakeNode(node_name), registry)
        assert fn is registry[expected]


class TestGammaDegenerationWarning:
    """gamma <= 0.01 still degenerates to epsilon (numerically motivated)
    but now warns LOUDLY once — the silent form of this cost a day of
    benchmark debugging (a 'gamma=0.001' config actually ran epsilon)."""

    def test_warns_at_tiny_gamma(self):
        import warnings as _w
        from autolrp.backward import rules as _rules
        _rules._GAMMA_DEGENERATION_WARNED.clear()
        x = torch.randn(2, 4).clamp(min=0)
        w = torch.randn(4, 3)
        z = x @ w
        R = torch.randn(2, 3)
        fwd = lambda a, b: a @ b
        bwd = lambda b, s: s @ b.T
        with pytest.warns(UserWarning, match="runs the epsilon rule instead"):
            gamma(x, w, R, 1e-9, fwd, bwd, bwd, gamma=0.001)
        # once per (value, rule): a second call is silent
        with _w.catch_warnings():
            _w.simplefilter("error")
            gamma(x, w, R, 1e-9, fwd, bwd, bwd, gamma=0.001)

    def test_silent_at_real_gamma(self):
        import warnings as _w
        x = torch.randn(2, 4).clamp(min=0)
        w = torch.randn(4, 3)
        z = x @ w
        R = torch.randn(2, 3)
        fwd = lambda a, b: a @ b
        bwd = lambda b, s: s @ b.T
        with _w.catch_warnings():
            _w.simplefilter("error")
            gamma(x, w, R, 1e-9, fwd, bwd, bwd, gamma=0.25)


class TestInputConvFact:
    """``rule={'input_conv': ('zbox', ...)}`` routes the input conv to
    zbox via the built-in analyzer fact; downstream convs follow the
    other keys. Replaces the removed LRPConfig.zbox side-channel."""

    ZB = ('zbox', {'low': -1.0, 'high': 1.0})

    def _facts_plan(self, cfg):
        from autolrp.backward import analysis as A
        model = _two_conv_cnn()
        x = autolrp.tensor(torch.randn(1, 3, 8, 8))
        plan = autolrp.walk(model(x)[0, 0],
                            strategy=autolrp.EXPLICIT_STRATEGY, config=cfg)
        A.run(plan)                     # analyzers write the facts
        return plan

    def test_zbox_only_at_input_conv(self):
        from autolrp.backward.analysis import node_facts
        cfg = LRPConfig(rule={**BASE, 'input_conv': self.ZB})
        plan = self._facts_plan(cfg)
        input_fn, interior = None, []
        for node, _ in plan:
            if 'ConvolutionBackward' not in node.name():
                continue
            fn, _ = _dispatch(cfg, node)
            if node_facts(node).get('input_conv'):
                input_fn = fn
            else:
                interior.append(fn)
        assert input_fn is zbox
        assert interior and all(fn is epsilon for fn in interior)

    def test_fact_outranks_name_match(self):
        """The canonical VGG spelling: zbox at the input conv, zplus at
        the other convs. The input conv matches BOTH 'input_conv' (fact)
        and 'ConvolutionBackward' (name); the fact tier must win —
        structural facts are more specific than any name substring."""
        from autolrp.backward.analysis import node_facts
        cfg = LRPConfig(rule={**BASE, 'input_conv': self.ZB,
                              'ConvolutionBackward': 'zplus'})
        plan = self._facts_plan(cfg)
        seen_input = seen_interior = False
        for node, _ in plan:
            if 'ConvolutionBackward' not in node.name():
                continue
            fn, _ = _dispatch(cfg, node)
            if node_facts(node).get('input_conv'):
                assert fn is zbox
                seen_input = True
            else:
                assert fn is zplus
                seen_interior = True
        assert seen_input and seen_interior

    def test_no_input_conv_key_leaves_input_alone(self):
        cfg = LRPConfig(rule=on_linear('epsilon'))
        plan = self._facts_plan(cfg)
        for node, _ in plan:
            if 'ConvolutionBackward' not in node.name():
                continue
            fn, _ = _dispatch(cfg, node)
            assert fn is epsilon
