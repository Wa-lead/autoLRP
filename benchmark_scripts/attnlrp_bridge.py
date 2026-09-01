r"""AttnLRP bridge for HF transformers lxt doesn't natively support.

lxt's ``DEFAULT_MAP`` only covers llama/qwen/bert/gpt2/gemma + torchvision-ViT.
RoBERTa and T5 are **old-style attention** (no shared ``eager_attention_forward``),
so lxt's ``patch_attention`` (the modern-interface trick used for HF ViT) does
NOT apply, and a plain ``monkey_patch(modeling_roberta)`` raises "not yet
supported". The faithful fix mirrors lxt's BERT support: a copy of the model's
self-attention forward with the AttnLRP uniform rule injected
(``divide_gradient(scores, 2)`` after Q·Kᵀ and ``divide_gradient(context, 2)``
after softmax·V), plus identity rules on the norms / activations / dropout and
the uniform rule on T5's gated-MLP element-wise multiply.

FRAGILITY: ``roberta_self_attention_forward`` / ``t5_attention_forward`` are
verbatim copies of transformers 4.52.3 forwards with two lines added each. They
are pinned to that version and must be re-synced if transformers is upgraded.
This is inherent to lxt-AttnLRP (and why the paper marks RoBERTa/T5 N/A).
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
from functools import partial

import torch
from torch import nn

from lxt.efficient.core import monkey_patch
from lxt.efficient.rules import divide_gradient, identity_rule_implicit
from lxt.efficient.patches import (patch_method, layer_norm_forward,
                                   rms_norm_forward, non_linear_forward, dropout_forward)


# ---------------------------------------------------------------------------
# RoBERTa — copy of transformers 4.52.3 RobertaSelfAttention.forward with the
# two AttnLRP divide_gradient lines injected (matches lxt's bert.py rule).
# ---------------------------------------------------------------------------
def roberta_self_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor]:
    mixed_query_layer = self.query(hidden_states)
    is_cross_attention = encoder_hidden_states is not None

    if is_cross_attention and past_key_value is not None:
        key_layer = past_key_value[0]
        value_layer = past_key_value[1]
        attention_mask = encoder_attention_mask
    elif is_cross_attention:
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask
    elif past_key_value is not None:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
    else:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

    query_layer = self.transpose_for_scores(mixed_query_layer)

    use_cache = past_key_value is not None
    if self.is_decoder:
        past_key_value = (key_layer, value_layer)

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = divide_gradient(attention_scores, 2)  # <-- AttnLRP uniform rule

    if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        query_length, key_length = query_layer.shape[2], key_layer.shape[2]
        if use_cache:
            position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        else:
            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        distance = position_ids_l - position_ids_r

        positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

        if self.position_embedding_type == "relative_key":
            relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores
        elif self.position_embedding_type == "relative_key_query":
            relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs = self.dropout(attention_probs)

    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = divide_gradient(context_layer, 2)  # <-- AttnLRP uniform rule

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    if self.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs


# ---------------------------------------------------------------------------
# T5 — copy of transformers 4.52.3 T5Attention.forward + the two divide_gradient
# lines. The relative position_bias is added to scores AFTER the rule (as in the
# original), which is fine: the uniform rule acts on the Q·Kᵀ matmul gradient.
# ---------------------------------------------------------------------------
def t5_attention_forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
    cache_position=None,
):
    batch_size, seq_length = hidden_states.shape[:2]
    is_cross_attention = key_value_states is not None

    query_states = self.q(hidden_states)
    query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    if past_key_value is not None:
        is_updated = past_key_value.is_updated.get(self.layer_idx)
        if is_cross_attention:
            curr_past_key_value = past_key_value.cross_attention_cache
        else:
            curr_past_key_value = past_key_value.self_attention_cache

    current_states = key_value_states if is_cross_attention else hidden_states
    if is_cross_attention and past_key_value is not None and is_updated:
        key_states = curr_past_key_value.key_cache[self.layer_idx]
        value_states = curr_past_key_value.value_cache[self.layer_idx]
    else:
        key_states = self.k(current_states)
        value_states = self.v(current_states)
        key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        if past_key_value is not None:
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True

    scores = torch.matmul(query_states, key_states.transpose(3, 2))
    scores = divide_gradient(scores, 2)  # <-- AttnLRP uniform rule

    if position_bias is None:
        key_length = key_states.shape[-2]
        real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, seq_length, key_length), device=scores.device, dtype=scores.dtype
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(
                real_seq_length, key_length, device=scores.device, cache_position=cache_position
            )
            position_bias = position_bias[:, :, -seq_length:, :]

        if mask is not None:
            causal_mask = mask[:, :, :, : key_states.shape[-2]]
            position_bias = position_bias + causal_mask

    if self.pruned_heads:
        mask = torch.ones(position_bias.shape[1])
        mask[list(self.pruned_heads)] = 0
        position_bias_masked = position_bias[:, mask.bool()]
    else:
        position_bias_masked = position_bias

    # out-of-place (orig T5 uses `scores += ...`): divide_gradient wraps `scores`
    # in a custom autograd Function whose output is a view — an in-place += on it
    # is forbidden ("view is being modified inplace").
    scores = scores + position_bias_masked

    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = divide_gradient(attn_output, 2)  # <-- AttnLRP uniform rule

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, -1, self.inner_dim)
    attn_output = self.o(attn_output)

    outputs = (attn_output, past_key_value, position_bias)

    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs


# ---------------------------------------------------------------------------
# T5 gated MLP (GeGLU) — identity rule on the activation + uniform rule on the
# element-wise gate*up multiply (lxt's gated_mlp_forward targets LLaMA attr
# names gate_proj/up_proj/down_proj, so T5's wi_0/wi_1/wo need their own copy).
# ---------------------------------------------------------------------------
def t5_gated_mlp_forward(self, hidden_states):
    hidden_gelu = identity_rule_implicit(self.act, self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = divide_gradient(hidden_states, 2)  # <-- uniform rule on the gate multiply
    hidden_states = self.dropout(hidden_states)
    if (
        isinstance(self.wo.weight, torch.Tensor)
        and hidden_states.dtype != self.wo.weight.dtype
        and self.wo.weight.dtype != torch.int8
    ):
        hidden_states = hidden_states.to(self.wo.weight.dtype)
    return self.wo(hidden_states)


# ---------------------------------------------------------------------------
# apply: build the per-model patch_map and monkey_patch the modeling module.
# Call BEFORE from_pretrained (patches the classes the model will instantiate),
# with the model loaded under attn_implementation="eager".
# ---------------------------------------------------------------------------
def apply_attnlrp(task: str) -> bool:
    r"""Patch the modeling module for `task` so AttnLRP fires. Returns True if a
    patch was applied, False if the task is not a bridged transformer."""
    import transformers.activations as acts

    if task == "bert":
        import transformers.models.bert.modeling_bert as bm
        monkey_patch(bm)  # native lxt DEFAULT_MAP (bert.attnLRP via replace_module)
        return True

    if task == "roberta":
        import transformers.models.roberta.modeling_roberta as rm
        monkey_patch(rm, patch_map={
            nn.LayerNorm: partial(patch_method, layer_norm_forward),
            acts.GELUActivation: partial(patch_method, non_linear_forward, keep_original=True),
            nn.Dropout: partial(patch_method, dropout_forward),
            rm.RobertaSelfAttention: partial(patch_method, roberta_self_attention_forward),
        })
        return True

    if task == "t5":
        import transformers.models.t5.modeling_t5 as tm
        monkey_patch(tm, patch_map={
            tm.T5LayerNorm: partial(patch_method, rms_norm_forward),
            tm.T5DenseGatedActDense: partial(patch_method, t5_gated_mlp_forward),
            nn.Dropout: partial(patch_method, dropout_forward),
            tm.T5Attention: partial(patch_method, t5_attention_forward),
        })
        return True

    return False
