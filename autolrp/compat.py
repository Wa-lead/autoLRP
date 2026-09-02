r"""Import-time compatibility probe: the installers read PyTorch's
PRIVATE ``_saved_*`` attributes, whose names follow each op's C++
signature (mul saves ``other``; bmm saves ``mat2``). A renamed attribute
does not raise -- ``getattr(..., None)`` becomes a silent native
fallback. One tiny graph per op at import fails loudly instead.
Skip with ``AUTOLRP_SKIP_SELFCHECK=1``."""
import os

import torch
from .backward.analysis import parents


def _probe_graphs():
    x = torch.randn(2, 3, requires_grad=True)
    w = torch.randn(4, 3, requires_grad=True)
    xb = torch.randn(1, 2, 3, requires_grad=True)
    b3 = torch.randn(1, 3, 4, requires_grad=True)
    xc = torch.randn(1, 2, 5, 5, requires_grad=True)
    wc = torch.randn(3, 2, 3, 3, requires_grad=True)
    yield ('MulBackward', x * x.sigmoid(), ('_saved_self', '_saved_other'))
    yield ('DivBackward', x / (x.sigmoid() + 1.0),
           ('_saved_self', '_saved_other'))
    yield ('BmmBackward', torch.bmm(xb, b3), ('_saved_self', '_saved_mat2'))
    yield ('AddmmBackward', torch.addmm(torch.zeros(2, 4), x, w.t()),
           ('_saved_mat1', '_saved_mat2'))
    yield ('MmBackward', x @ w.t(), ('_saved_self', '_saved_mat2'))
    yield ('ConvolutionBackward', torch.nn.functional.conv2d(xc, wc),
           ('_saved_input', '_saved_weight'))


def run_selfcheck():
    if os.environ.get('AUTOLRP_SKIP_SELFCHECK'):
        return
    problems = []
    for opname, out, attrs in _probe_graphs():
        node = out.grad_fn
        while node is not None and opname not in node.name():
            ps = parents(node, skip_aliases=False)
            node = ps[0] if ps else None
        if node is None:
            problems.append(f"{opname}: node not found in probe graph")
            continue
        for at in attrs:
            if getattr(node, at, None) is None:
                problems.append(
                    f"{node.name()}: expected saved attribute {at!r} missing "
                    f"(exposes: {[d for d in dir(node) if d.startswith('_saved')]})")
    if problems:
        raise ImportError(
            "autoLRP self-check failed on torch " + torch.__version__
            + " -- the private saved-state API changed; relevance would "
            "silently degrade to plain gradients.\n  "
            + "\n  ".join(problems)
            + "\n(escape hatch: AUTOLRP_SKIP_SELFCHECK=1)")
