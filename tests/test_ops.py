"""Per-operation LRP integration + autograd.Function wrapper backward parity.

Two concerns covered here:

1. Op coverage: every supported PyTorch op flows finite, input-shaped R
   through ``autolrp.tensor → op → .lrp()``. One parametrized table per
   op category. Math correctness lives in ``test_rules.py``; this is
   the wiring + shape contract.

2. Wrapper backward parity: the ``autograd.Function`` wrappers in
   ``forward/ops.py`` produce gradients bit-identical to PyTorch's
   native backward when no LRP consumer is in the chain. Public
   contract — wrappers must be safe to drop into normal training.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import autoLRP as autolrp
from tests._cfg import on_linear
from autoLRP import BASE
from autoLRP import LRPConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_lrp(x_data, op_fn, **cfg_kwargs):
    """Wrap input → op → lrp → return relevance."""
    x = autolrp.tensor(x_data.clone())
    y = op_fn(x)
    cfg = LRPConfig(**cfg_kwargs) if cfg_kwargs else None
    y.sum().lrp(config=cfg)
    return x.relevance


def _assert_ok(R, expected_shape):
    assert R is not None
    assert R.shape == expected_shape
    assert torch.isfinite(R).all()


def _grad(fn, *inputs):
    """Autograd backward against ``inputs``; return grads in input order."""
    outs = [t.detach().clone().requires_grad_(t.requires_grad) for t in inputs]
    fn(*outs).sum().backward()
    return [t.grad.detach() if t.grad is not None else None for t in outs]


# ===========================================================================
# Op coverage — one parametrized table over op factories
# ---------------------------------------------------------------------------
# Factories rather than pre-instantiated modules so each test runs against
# fresh weights drawn under the autouse ``seed`` fixture from conftest.py.
# ===========================================================================

def _matmul_op():
    w = nn.Parameter(torch.randn(16, 8))
    return lambda x: x @ w


def _add_op():
    b = torch.randn(2, 8)
    return lambda x: x + b


def _sub_op():
    b = torch.randn(2, 8)
    return lambda x: x - b


def _mul_broadcast_op():
    g = torch.randn(1, 8, 1, 1)        # broadcasts over (H, W)
    return lambda x: x * g


_OP_CASES = [
    # (name, op_factory, input_shape)
    # Linear family
    ('linear_bias',      lambda: nn.Linear(16, 8, bias=True).eval(),  (2, 16)),
    ('linear_nobias',    lambda: nn.Linear(16, 8, bias=False).eval(), (2, 16)),
    ('matmul',           _matmul_op,                                  (2, 16)),
    # Convolutions
    ('conv1d',           lambda: nn.Conv1d(3, 4, 3, padding=1).eval(),
                                                                      (1, 3, 16)),
    ('conv2d',           lambda: nn.Conv2d(3, 4, 3, padding=1).eval(),
                                                                      (1, 3, 8, 8)),
    ('conv3d',           lambda: nn.Conv3d(3, 4, 3, padding=1).eval(),
                                                                      (1, 3, 4, 4, 4)),
    ('conv2d_stride',    lambda: nn.Conv2d(3, 8, 3, stride=2, padding=1).eval(),
                                                                      (1, 3, 7, 7)),
    ('conv2d_depthwise', lambda: nn.Conv2d(8, 8, 3, padding=1, groups=8).eval(),
                                                                      (1, 8, 8, 8)),
    # Norms
    ('layernorm',        lambda: nn.LayerNorm(16).eval(),             (2, 4, 16)),
    ('batchnorm',        lambda: nn.BatchNorm2d(8).eval(),            (2, 8, 4, 4)),
    ('groupnorm',        lambda: nn.GroupNorm(4, 8).eval(),           (2, 8, 4, 4)),
    # Arithmetic
    ('add_two_tensor',   _add_op,                                     (2, 8)),
    ('sub_two_tensor',   _sub_op,                                     (2, 8)),
    ('mul_broadcast',    _mul_broadcast_op,                           (1, 8, 4, 4)),
    ('div_scalar',       lambda: lambda x: x / 2.0,                   (2, 8)),
]


@pytest.mark.parametrize("name,op_factory,input_shape", _OP_CASES,
                         ids=[c[0] for c in _OP_CASES])
def test_op_produces_finite_input_shaped_R(name, op_factory, input_shape):
    """Each supported op produces R with input shape and no NaN/Inf."""
    R = _run_lrp(torch.randn(*input_shape), op_factory())
    _assert_ok(R, input_shape)


# Activations — long list, distinct axis.
@pytest.mark.parametrize("act_cls", [
    nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid,
    nn.ELU, nn.CELU, nn.SELU, nn.Softplus, nn.Hardswish, nn.Hardsigmoid,
    nn.LogSigmoid, nn.Mish, nn.Hardtanh,
])
def test_activation_produces_finite_input_shaped_R(act_cls):
    R = _run_lrp(torch.randn(2, 8), act_cls().eval())
    _assert_ok(R, (2, 8))


# BMM — distinct parametrize axis (BmmBackward rule entry).
@pytest.mark.parametrize("bmm_rule", ['epsilon', 'detach_lhs'])
def test_bmm_two_data_operands(bmm_rule):
    # Both operands live: with a constant rhs, autograd does not save
    # _saved_self and the installer falls back (loudly) to native --
    # that path has its own test in test_loud_fallback.
    B = autolrp.tensor(torch.randn(2, 6, 5))
    R = _run_lrp(torch.randn(2, 4, 6),
                 lambda x, _B=B: torch.bmm(x, _B),
                 rule={**BASE, 'BmmBackward': bmm_rule})
    _assert_ok(R, (2, 4, 6))


# Reductions — distinct parametrize axis (dim, keepdim).
@pytest.mark.parametrize("dim,keepdim", [
    (None, False), (-1, False), (-1, True), ((1, 2), False),
])
def test_mean_reduction(dim, keepdim):
    op = (lambda x: x.mean()) if dim is None else (
        lambda x: x.mean(dim=dim, keepdim=keepdim))
    R = _run_lrp(torch.randn(2, 4, 8), op)
    _assert_ok(R, (2, 4, 8))


@pytest.mark.parametrize("dim,keepdim", [(None, False), (-1, False), ((1,), True)])
def test_sum_reduction(dim, keepdim):
    op = (lambda x: x.sum()) if dim is None else (
        lambda x: x.sum(dim=dim, keepdim=keepdim))
    R = _run_lrp(torch.randn(2, 4, 8), op)
    _assert_ok(R, (2, 4, 8))


# Softmax — three modes.
@pytest.mark.parametrize("mode", ['passthrough', 'detach', 'jacobian'])
def test_softmax_mode_produces_finite_R(mode):
    R = _run_lrp(torch.randn(2, 8), lambda x: torch.softmax(x, dim=-1),
                 softmax=mode)
    _assert_ok(R, (2, 8))


# ===========================================================================
# autograd.Function wrapper backward parity.
# ---------------------------------------------------------------------------
# Wrappers in ``forward/ops.py`` must produce gradients bit-identical to
# PyTorch's native backward when there's no LRP consumer in the chain.
# Public contract: wrappers must be safe to drop into normal training.
# ===========================================================================

class TestWrapperBackwardParity:
    def test_softmax(self):
        from autoLRP.forward.ops import Softmax as _LRPSoftmax
        x = torch.randn(2, 8, requires_grad=True)
        g_native, = _grad(lambda t: torch.softmax(t, dim=-1), x)
        g_wrap, = _grad(lambda t: _LRPSoftmax.apply(t, -1), x)
        torch.testing.assert_close(g_native, g_wrap, atol=1e-6, rtol=0)

    def test_add(self):
        from autoLRP.forward.ops import Add as _LRPAdd
        a = torch.randn(2, 8, requires_grad=True)
        b = torch.randn(2, 8, requires_grad=True)
        ga_n, gb_n = _grad(lambda x, y: x + y, a, b)
        ga_w, gb_w = _grad(_LRPAdd.apply, a, b)
        torch.testing.assert_close(ga_n, ga_w, atol=1e-6, rtol=0)
        torch.testing.assert_close(gb_n, gb_w, atol=1e-6, rtol=0)

    def test_sub(self):
        from autoLRP.forward.ops import Sub as _LRPSub
        a = torch.randn(2, 8, requires_grad=True)
        b = torch.randn(2, 8, requires_grad=True)
        ga_n, gb_n = _grad(lambda x, y: x - y, a, b)
        ga_w, gb_w = _grad(_LRPSub.apply, a, b)
        torch.testing.assert_close(ga_n, ga_w, atol=1e-6, rtol=0)
        torch.testing.assert_close(gb_n, gb_w, atol=1e-6, rtol=0)

    @pytest.mark.parametrize("dim,keepdim", [
        (None, False), (-1, False), (-1, True), ((1, 2), False),
    ])
    def test_mean(self, dim, keepdim):
        from autoLRP.forward.ops import Mean as _LRPMean
        x = torch.randn(2, 4, 8, requires_grad=True)
        native = (lambda t: t.mean()) if dim is None else (
            lambda t: t.mean(dim=dim, keepdim=keepdim))
        g_n, = _grad(native, x)
        g_w, = _grad(lambda t: _LRPMean.apply(t, dim, keepdim), x)
        torch.testing.assert_close(g_n, g_w, atol=1e-6, rtol=0)

    def test_sum(self):
        from autoLRP.forward.ops import Sum as _LRPSum
        x = torch.randn(2, 4, 8, requires_grad=True)
        g_n, = _grad(lambda t: t.sum(dim=-1), x)
        g_w, = _grad(lambda t: _LRPSum.apply(t, -1, False), x)
        torch.testing.assert_close(g_n, g_w, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Forward parity: the wrapped tensor must reproduce native call forms
# exactly (regression pins for real bugs).
# ---------------------------------------------------------------------------

class TestWrappedForwardParity:
    @pytest.mark.parametrize("call", [
        lambda t: t.mean(1, True),            # positional keepdim
        lambda t: t.sum(1, True),
        lambda t: t.mean(dim=None, keepdim=True),
        lambda t: t.sum(dim=None, keepdim=True),
        lambda t: t.mean(dim=(-3, 2), keepdim=False),   # mixed-sign dims
        lambda t: t.sum(dim=(-3, 2), keepdim=True),
    ])
    def test_reduction_call_forms(self, call):
        data = torch.randn(2, 3, 4, 5)
        native = call(data)
        wrapped = call(autolrp.tensor(data.clone()))
        assert tuple(wrapped.shape) == tuple(native.shape)
        torch.testing.assert_close(
            torch.Tensor(wrapped.detach()), native, atol=1e-6, rtol=0)
        wrapped.sum().lrp()   # and the graph must still run LRP end to end

    @pytest.mark.parametrize("mask_kind", ['bool', 'float', 'causal'])
    def test_sdpa_decomposed_matches_native_mask_semantics(self, mask_kind):
        """SDPA bool masks mean True = MAY attend; the decomposition must
        keep that polarity (regression: it used to mask the True side)."""
        torch.manual_seed(0)
        q = torch.randn(1, 2, 4, 8)
        k, v = torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)
        kw = {}
        if mask_kind == 'bool':
            m = torch.zeros(4, 4, dtype=torch.bool)
            m[:, :2] = True
            kw['attn_mask'] = m
        elif mask_kind == 'float':
            m = torch.zeros(4, 4)
            m[:, 2:] = float('-inf')
            kw['attn_mask'] = m
        else:
            kw['is_causal'] = True
        native = F.scaled_dot_product_attention(q, k, v, **kw)
        xq = autolrp.tensor(q.clone())
        dec = F.scaled_dot_product_attention(xq, k, v, **kw)
        torch.testing.assert_close(
            torch.Tensor(dec.detach()), native, atol=1e-5, rtol=0)
