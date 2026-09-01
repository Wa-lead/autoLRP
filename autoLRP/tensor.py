r""":class:`LRPTensor`, the wrapped input. It captures the gradient that
reaches it into ``.relevance`` and routes ops through
:mod:`autoLRP.forward.intercept`. Choose the attributed scalar by
slicing the output::

    x = autoLRP.tensor(image)
    out = model(x)
    out[0, k].lrp()                   # one class
    (out[0, a] - out[0, b]).lrp()     # class difference
    (out * mask).sum().lrp()          # a seed mask
    heatmap = x.relevance
"""
import torch
from typing import Optional

from .config import LRPConfig
from .forward.intercept import torch_function_handler
from .backward.engine import graph_lrp


class LRPTensor(torch.Tensor):
    r""":class:`torch.Tensor` subclass that captures backward signal as
    ``.relevance`` when :meth:`lrp` is called on a downstream scalar.
    """

    @staticmethod
    def __new__(cls, data):
        raw = data.detach() if isinstance(data, torch.Tensor) \
              else torch.as_tensor(data)
        instance = torch.Tensor._make_subclass(cls, raw)
        instance.requires_grad_(True)
        return instance

    def __init__(self, data):
        if hasattr(self, '_lrp_init'):
            return
        self._lrp_init = True
        self.relevance: Optional[torch.Tensor] = None
        self.register_hook(self._capture_relevance)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return torch_function_handler(cls, func, types, args, kwargs)

    def _capture_relevance(self, grad):
        r"""Gradient hook: store the backward signal as :attr:`relevance`."""
        self.relevance = grad.detach()
        return grad

    def lrp(self,
            rule: Optional[dict] = None,
            config: Optional[LRPConfig] = None):
        r"""Run LRP from this scalar and fill ``.relevance`` on the wrapped
        inputs; one ``backward`` per call, the forward is not re-run.
        ``config`` defaults to ``LRPConfig()`` (:data:`autoLRP.BASE`), or to
        ``LRPConfig(rule=rule)`` when only ``rule`` is given; passing both
        ``rule`` and ``config`` raises :class:`ValueError`. Returns ``self``,
        or the per-node dict when ``config.capture_layers`` is set.
        """
        if rule is not None and config is not None:
            raise ValueError(
                "Pass either rule= or config= to lrp(), not both: "
                "config already carries a rule, so rule= would conflict "
                "with config.rule. Use LRPConfig(rule=...) instead.")

        if self.grad_fn is None:
            if self.requires_grad:
                raise ValueError(
                    "lrp() was called on the wrapped input leaf itself, which "
                    "has no computation graph to propagate through. Run the "
                    "model on this tensor and call .lrp() on a downstream "
                    "scalar (e.g. out[0, k].lrp()).")
            raise RuntimeError(
                "Cannot run LRP on a tensor with no computation graph. "
                "Make sure you called autoLRP.tensor() on the input.")

        if config is None:
            config = LRPConfig() if rule is None else LRPConfig(rule=rule)

        layer_relevances = {} if config.capture_layers else None
        layer_R = graph_lrp(self, config=config,
                             layer_relevances=layer_relevances)

        if config.capture_layers:
            return layer_R
        return self

    def __repr__(self):
        r = 'set' if getattr(self, 'relevance', None) is not None else 'None'
        return f"LRPTensor(shape={list(self.shape)}, relevance={r})"


# ---------------------------------------------------------------------------
# Public constructor
# ---------------------------------------------------------------------------

def tensor(data) -> LRPTensor:
    r"""Wrap ``data`` as an :class:`LRPTensor`; every op on it keeps the
    subclass, so ``.lrp()`` on any downstream scalar fills
    ``.relevance`` here. ``data`` may be a :class:`torch.Tensor` or any
    array-like accepted by :func:`torch.as_tensor` (list, tuple,
    numpy array, scalar); non-convertible input raises :class:`TypeError`.
    """
    if not isinstance(data, torch.Tensor):
        try:
            data = torch.as_tensor(data)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError(
                f"Expected torch.Tensor or array-like convertible via "
                f"torch.as_tensor, got {type(data).__name__}") from exc
    return LRPTensor(data)
