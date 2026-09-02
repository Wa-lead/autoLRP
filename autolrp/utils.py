r"""Shared utility helpers."""

import torch
from typing import Optional
from .backward.analysis import parents


def find_bias(node, parent_idx: int = 0) -> Optional[torch.Tensor]:
    r"""The 1-D bias tensor of ``node`` from parent ``parent_idx`` of
    ``next_functions`` (0 for ``Addmm``, 2 for ``Convolution``), detached,
    or ``None``.
    """
    ps = parents(node, skip_aliases=False)
    if len(ps) <= parent_idx:
        return None
    bias_fn = ps[parent_idx]
    if bias_fn is not None and hasattr(bias_fn, 'variable'):
        v = bias_fn.variable
        if v.ndim == 1:
            return v.detach()
    return None
