"""Suite-wide isolation. Every test runs against the same world state.

Three sources of cross-test bleed exist in the library by design, and
each one produced a real order-dependent failure or would under
modification:

1. RANDOMNESS. Several tests build unseeded models; whether a cell
   passed used to depend on how many draws earlier tests consumed
   (observed: the epsilon/cnn_nobias conservation cell failed only when
   collected after test_finite_under_input_scale). Fix: seed torch /
   random / numpy identically at the start of every test.

2. MUTABLE REGISTRIES. INSTALLERS, ANALYZERS, REWRITES and the
   decompose-attention flag are process-global and writable through the
   public API. A test that registers and fails before its ``finally``
   poisons every later test. Fix: snapshot before, restore after,
   unconditionally.

3. WARN-ONCE SETS. _UNMATCHED_WARNED and friends are process-global, so
   a warning consumed by one test is silent for the next -- any test
   asserting a warning fires becomes order-dependent. Fix: clear all
   four sets before every test.

Nothing here changes library behavior; it pins the state each test
starts from.
"""
import random

import pytest
import torch

try:
    import numpy as _np
except ImportError:                                    # numpy is optional here
    _np = None

from autolrp.backward import strategies as _strategies
from autolrp.backward import analysis as _analysis
from autolrp.backward import install as _install
from autolrp.backward import rules as _rules
from autolrp.backward import engine as _engine
from autolrp.forward import intercept as _intercept


@pytest.fixture(autouse=True)
def _isolate():
    # 1. deterministic randomness, identical for every test
    torch.manual_seed(0)
    random.seed(0)
    if _np is not None:
        _np.random.seed(0)

    # 2. snapshot mutable global state
    installers = dict(_strategies.INSTALLERS)
    analyzers = dict(_analysis.ANALYZERS)
    rewrites = dict(_intercept.REWRITES)
    decompose = _intercept.get_decompose_attention()

    # 3. fresh warn-once state (tests may assert a warning fires)
    _engine._UNMATCHED_WARNED.clear()
    _install._MISSING_STATE_WARNED.clear()
    _rules._GAMMA_DEGENERATION_WARNED.clear()
    _intercept._INPLACE_REMAP_WARNED.clear()

    yield

    _strategies.INSTALLERS.clear()
    _strategies.INSTALLERS.update(installers)
    _analysis.ANALYZERS.clear()
    _analysis.ANALYZERS.update(analyzers)
    _intercept.REWRITES.clear()
    _intercept.REWRITES.update(rewrites)
    _intercept.set_decompose_attention(decompose)
