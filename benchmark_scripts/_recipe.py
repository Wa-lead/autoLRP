r"""Benchmark autoLRP configs.

SINGLE SOURCE OF TRUTH FOR THE autoLRP CONFIG (the analogue of _metrics.py
for the attribution side). Every autoLRP script imports build_config() so the
recipe lives in exactly one place.

Config vocabulary (current library): rule= is a dict keyed by exact canonical
node names ('AddmmBackward') or analyzer fact names ('weights_operand');
there are no aliases and no 'default' sentinel — start from autoLRP.BASE and
override entries. A node no key addresses is a hard error.

The text/EQA configuration:

    rule:       gamma on the linear projections. PyTorch emits AddmmBackward
                for nn.Linear-with-bias and MmBackward for matmul-without-bias,
                so gamma is keyed on both; every other family keeps its BASE
                entry (epsilon on conv/bmm, proportional on mul/add).
    softmax:    'passthrough' for the baseline text/EQA config; 'jacobian'
                (the AttnLRP-style DeepTaylor softmax, attn='attnlrp') as the
                ablation/fix where noted.
    layernorm:  'passthrough' -- all R to the input. (The library default is
                'identity', so this must be set explicitly.)
    activation: 'passthrough'.
    attention:  QK^T / attn*V are BmmBackward. BASE's 'BmmBackward': 'epsilon'
                is the bilinear epsilon with the 2z denominator — the knob
                formerly spelled bilinear='full'.

  gamma_linear: 1.0 for text (IMDB/Wiki), 0.001 for EQA (BERT/RoBERTa/T5);
                overridable per run.

  IMPORTANT runtime note (the NaN trap): install_addmm reads node._saved_mat1;
  if the model's params have requires_grad=False, autograd elides mat1 and the
  hook silently produces wrong/NaN relevance. from_pretrained leaves params at
  requires_grad=True by default, but call ensure_param_grads(model) before
  attributing to be safe.
"""
from __future__ import annotations


# Benchmark gamma_linear values.
TEXT_GAMMA = 1.0
EQA_GAMMA = 0.001


def apply_bilinear(rule: dict, choice: str) -> None:
    r"""Write the historic bilinear= knob into exact-key rule entries.

    'full'    -> BmmBackward epsilon (2z-denominator bilinear epsilon)
    'uniform' -> BmmBackward uniform (AttnLRP's uniform split)
    'cplrp'   -> BmmBackward epsilon + detach the softmax-weights operand
                 (Ali et al. 2022), same entries attn='cplrp' compiles to.
    """
    if choice == 'full':
        rule['BmmBackward'] = 'epsilon'
        rule.pop('weights_operand', None)
    elif choice == 'uniform':
        rule['BmmBackward'] = 'uniform'
        rule.pop('weights_operand', None)
    elif choice == 'cplrp':
        rule['BmmBackward'] = 'epsilon'
        rule['weights_operand'] = ('detach', {'by': 'weights_operand'})
    else:
        raise ValueError(f"bilinear choice {choice!r}; "
                         "expected 'full', 'uniform' or 'cplrp'")


def make_rule(*, gamma_linear=None, conv_gamma=None, bilinear: str = 'full') -> dict:
    r"""BASE with the benchmark's overrides applied. Returns a plain dict so
    callers can override further entries before building the LRPConfig."""
    from autoLRP import BASE
    rule = dict(BASE)
    if gamma_linear is not None:
        rule['AddmmBackward'] = ('gamma', {'gamma': gamma_linear})
        rule['MmBackward'] = ('gamma', {'gamma': gamma_linear})
    if conv_gamma is not None:
        rule['ConvolutionBackward'] = ('gamma', {'gamma': conv_gamma})
    apply_bilinear(rule, bilinear)
    return rule


def build_config(gamma_linear: float, *, softmax: str = 'passthrough'):
    r"""Build the benchmark autoLRP LRPConfig.

    Args:
        gamma_linear: gamma on AddmmBackward/MmBackward (linear projections).
        softmax: 'passthrough' (baseline) or 'jacobian' (AttnLRP-style).
    Returns:
        LRPConfig instance.
    """
    from autoLRP import LRPConfig
    return LRPConfig(
        rule=make_rule(gamma_linear=gamma_linear),
        softmax=softmax,
        layernorm='passthrough',
        activation='passthrough',
    )


def build_text_config(gamma_linear: float = TEXT_GAMMA, *, softmax='passthrough'):
    r"""Config for the text tasks (IMDB/Wiki). gamma_linear=1.0."""
    return build_config(gamma_linear, softmax=softmax)


def build_eqa_config(gamma_linear: float = EQA_GAMMA, *, softmax='passthrough'):
    r"""Config for the EQA tasks (BERT/RoBERTa/T5). gamma_linear=0.001."""
    return build_config(gamma_linear, softmax=softmax)


def ensure_param_grads(model) -> None:
    r"""Force requires_grad=True on all params so autograd saves the linear-layer
    input activation (node._saved_mat1). Without this, install_addmm's hook sees
    mat1=None and produces wrong/NaN relevance (the elision trap documented in
    install.py). Harmless when params already require grad.
    """
    for p in model.parameters():
        p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Vision recipes (different from text/EQA: gamma on convolutions).
#
#   VGG: gamma on CONV only (conv_gamma=125), NOT on linears; the VGG row
#        uses unfiltered conv-gamma=125 (relevance_filter available as a knob).
#   ViT: gamma on BOTH the patch-embed conv (125) and the transformer
#        linears (0.001).
#
# autoLRP keys gamma on ConvolutionBackward (conv) and AddmmBackward/MmBackward
# (linears). BmmBackward epsilon (BASE) for ViT attention; VGG has no attention.
# ---------------------------------------------------------------------------

VGG_CONV_GAMMA = 125.0
VIT_CONV_GAMMA = 125.0
VIT_MM_GAMMA = 0.001


def build_vgg_config(conv_gamma: float = VGG_CONV_GAMMA, *, relevance_filter: float = 1.0):
    r"""VGG config: gamma on convolutions (conv_gamma=125), epsilon on the
    classifier linears. No attention (CNN)."""
    from autoLRP import LRPConfig
    return LRPConfig(
        rule=make_rule(conv_gamma=conv_gamma),
        relevance_filter=relevance_filter,
    )


def build_vit_config(conv_gamma: float = VIT_CONV_GAMMA, mm_gamma: float = VIT_MM_GAMMA,
                     *, softmax: str = 'jacobian', bilinear: str = 'full'):
    r"""ViT config: gamma on the patch-embed conv (125) AND the transformer
    linears (0.001), BmmBackward epsilon on attention, epsilon elsewhere.
    softmax='jacobian' (the fix): passthrough under-attributes the attention
    block (+0.3700 -> +0.4127 ABPC at n=100). The keyword overrides exist for
    the runner's --vit-softmax / --vit-bilinear ablation."""
    from autoLRP import LRPConfig
    return LRPConfig(
        rule=make_rule(gamma_linear=mm_gamma, conv_gamma=conv_gamma,
                       bilinear=bilinear),
        softmax=softmax,
        layernorm='passthrough',
        activation='passthrough',
    )
