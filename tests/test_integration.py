"""End-to-end integration: robustness, reproducibility, edge cases.

Conservation matrix lives in ``test_conservation.py``; mode-flag wiring
in ``test_modes.py``; this file is everything else that requires a real
model + LRP run but isn't a property test of conservation.
"""
import pytest
import torch
import torch.nn as nn

import autolrp
from tests._cfg import on_linear
from autolrp import BASE
from autolrp import LRPConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mlp_no_bias():
    return nn.Sequential(
        nn.Linear(16, 32, bias=False), nn.GELU(),
        nn.Linear(32, 4, bias=False),
    ).eval()


# ---------------------------------------------------------------------------
# Numerical robustness
# ---------------------------------------------------------------------------

_CFGS_FOR_PARAM = [
    ('epsilon',   LRPConfig(rule=on_linear('epsilon'))),
    ('zplus',     LRPConfig(rule=on_linear('zplus'))),
    ('composite', LRPConfig.composite()),
]


@pytest.mark.parametrize("scale", [1e-4, 1e-2, 1.0, 1e2, 1e4])
@pytest.mark.parametrize("name,cfg", _CFGS_FOR_PARAM,
                          ids=[n for n, _ in _CFGS_FOR_PARAM])
def test_finite_under_input_scale(scale, name, cfg):
    """No NaN/Inf across 8 orders of magnitude of input."""
    model = _mlp_no_bias()
    data = torch.randn(1, 16) * scale
    x = autolrp.tensor(data)
    out = model(x)
    out[0, out.argmax(-1)].lrp(config=cfg)
    assert torch.isfinite(x.relevance).all()


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (1, 8),                # MLP
    (2, 8),                # batched
    (1, 4, 16),            # transformer / sequence
    (1, 3, 8, 8),          # CNN
])
def test_relevance_shape_matches_input(shape):
    x = autolrp.tensor(torch.randn(*shape))
    if len(shape) == 4:
        out = nn.Conv2d(shape[1], 4, 3, padding=1).eval()(x).flatten(1)
    else:
        out = nn.Linear(shape[-1], 4).eval()(x)
    out[0, 0].lrp()
    assert x.relevance.shape == x.shape


# ---------------------------------------------------------------------------
# Reproducibility — same input → same output, bit-exactly.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,cfg", _CFGS_FOR_PARAM,
                          ids=[n for n, _ in _CFGS_FOR_PARAM])
def test_bit_exact_reproduction(name, cfg):
    model = _mlp_no_bias()
    data = torch.randn(1, 16)

    def run():
        x = autolrp.tensor(data.clone())
        model(x)[0, 0].lrp(config=cfg)
        return x.relevance.clone()

    assert torch.equal(run(), run())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_scalar_output():
    """target=None on a scalar output works."""
    x = autolrp.tensor(torch.randn(1, 4))
    nn.Linear(4, 1)(x).sum().lrp()
    assert x.relevance is not None


def test_multiple_passes_per_target_dim():
    """Per-output-dim LRP on the same input — building block for BiLRP."""
    lin = nn.Linear(8, 4).eval()
    data = torch.randn(1, 8)
    relevances = []
    for d in range(4):
        x = autolrp.tensor(data.clone())
        out = lin(x)
        out[0, d].lrp(rule=on_linear('epsilon'))
        relevances.append(x.relevance.clone())

    assert len(relevances) == 4
    for r in relevances:
        assert r.shape == (1, 8)
        assert torch.isfinite(r).all()


# ---------------------------------------------------------------------------
# Fused scaled-dot-product-attention installer (decompose_attention=False)
# ---------------------------------------------------------------------------

import torch.nn.functional as F


class _MHA(nn.Module):
    def __init__(self, d=16, h=2, causal=False):
        super().__init__()
        self.h, self.d, self.causal = h, d, causal
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.out = nn.Linear(d, 4)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=self.causal)
        return self.out(self.o(a.transpose(1, 2).reshape(B, T, self.d)).mean(1))


class _GQA(nn.Module):
    def __init__(self, d=16, hq=4, hkv=2):
        super().__init__()
        self.hq, self.hkv, self.dh = hq, hkv, d // hq
        self.q = nn.Linear(d, hq * self.dh)
        self.k = nn.Linear(d, hkv * self.dh)
        self.v = nn.Linear(d, hkv * self.dh)
        self.o = nn.Linear(hq * self.dh, d)
        self.out = nn.Linear(d, 4)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).reshape(B, T, self.hq, self.dh).transpose(1, 2)
        k = self.k(x).reshape(B, T, self.hkv, self.dh).transpose(1, 2)
        v = self.v(x).reshape(B, T, self.hkv, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return self.out(self.o(a.transpose(1, 2).reshape(B, T, self.hq * self.dh)).mean(1))


def _lrp_relevance(model, data, decompose, cfg):
    autolrp.set_decompose_attention(decompose)
    try:
        x = autolrp.tensor(data.clone())
        out = model(x)
        out[0, int(out[0].argmax())].lrp(config=cfg)
        return x.relevance.clone()
    finally:
        autolrp.set_decompose_attention(True)


_SDPA_CFGS = [
    ('default', LRPConfig()),                                   # cplrp + passthrough
    ('attnlrp', LRPConfig(softmax='jacobian')),
    ('uniform', LRPConfig(softmax='passthrough',
                          rule={**BASE, 'BmmBackward': 'uniform'})),
]


@pytest.mark.parametrize("name,cfg", _SDPA_CFGS, ids=[n for n, _ in _SDPA_CFGS])
@pytest.mark.parametrize("model_fn,seed", [
    (lambda: _MHA(causal=False), 0),
    (lambda: _MHA(causal=True), 1),
    (lambda: _GQA(), 2),
], ids=["mha", "mha-causal", "gqa"])
def test_fused_sdpa_matches_decomposition(model_fn, seed, name, cfg):
    """install_sdpa (fused) reproduces the decomposed path to numerical precision."""
    torch.manual_seed(seed)
    model = model_fn().eval()
    data = torch.randn(1, 6, 16)
    r_dec = _lrp_relevance(model, data, True, cfg)
    r_fused = _lrp_relevance(model, data, False, cfg)
    assert torch.isfinite(r_fused).all()
    rel = (r_dec - r_fused).abs().max() / (r_dec.abs().max() + 1e-12)
    assert rel < 1e-4, f"{name}: rel diff {rel:.2e}"


def test_fused_sdpa_installer_registered():
    """The fused-attention backend node names dispatch to install_sdpa."""
    from autolrp.backward.install import install_sdpa
    for nm in ("ScaledDotProductEfficientAttentionBackward0",
               "ScaledDotProductFlashAttentionForCpuBackward0"):
        _, fn = autolrp.match_installer(nm)
        assert fn is install_sdpa


def test_decompose_toggle_state():
    """The decomposition toggle setter/getter/context-manager behave correctly
    (both modes are exercised end-to-end by the parity tests above)."""
    assert autolrp.get_decompose_attention() is True          # default on
    autolrp.set_decompose_attention(False)
    assert autolrp.get_decompose_attention() is False
    autolrp.set_decompose_attention(True)
    with autolrp.decompose_attention(False):
        assert autolrp.get_decompose_attention() is False
    assert autolrp.get_decompose_attention() is True          # restored on exit
