r"""Fused ops written out, during the forward, as the ordinary ops they
compute, so every step is a node the rules handle. Each function takes
the fused call's ``(args, kwargs)`` and returns what the fused call
would have returned, with the fused op's own signature, so the intercept
passes the call's arguments through. The tensors are LRPTensors, so the
ops used here are intercepted in turn. A new fused op gets a function
here and a line in the intercept's table.

Masks are added to the scores as a large negative constant instead of
``masked_fill`` with ``-inf``: an add is a node the rules handle, and
``softmax(-1e9)`` is 0 in float32 and float64.
"""
import math

import torch
import torch.nn.functional as F


MASK_FILL = -1.0e9          # finite: -inf times 0 is NaN in the gradient


# ---------------------------------------------------------------------------
# The step both fused ops share
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(q, k, v, scale, bias=None, dropout_p=0.0):
    r"""``softmax(q @ k^T * scale + bias) @ v``, the step both fused ops
    compute, as five nodes. Differs from ``F.scaled_dot_product_attention``
    in taking the mask as an additive ``bias`` (``MASK_FILL`` where
    attention is forbidden) and returning ``(output, attention weights)``.
    ``q``, ``k``, ``v`` are ``(..., T, D)``, ``(..., S, D)``, ``(..., S, Dv)``."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if bias is not None:
        scores = scores + bias
    weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p)
    return torch.matmul(weights, v), weights


def causal_bias(T, S, like):
    """``MASK_FILL`` above the diagonal of a ``(T, S)`` table, 0 elsewhere."""
    return torch.triu(torch.full((T, S), MASK_FILL, device=like.device, dtype=like.dtype), diagonal=1)


def mask_bias(mask, masked_out, like):
    """An additive bias from a mask: a bool mask becomes ``MASK_FILL`` where
    ``masked_out(mask)`` is ``True`` (the two fused ops read a bool mask
    with opposite polarity), a float mask is already additive."""
    if mask.dtype == torch.bool:
        return masked_out(mask).to(like.dtype) * MASK_FILL
    return mask


# ---------------------------------------------------------------------------
# scaled_dot_product_attention(query, key, value, attn_mask=None,
#     dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)
# ---------------------------------------------------------------------------

def decompose_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                   scale=None, enable_gqa=False):
    r"""SDPA as matmul, scale, mask, softmax, matmul; the signature of
    ``F.scaled_dot_product_attention``. A bool ``attn_mask`` means *may
    attend* where ``True``. Grouped-query attention (fewer key/value
    heads than query heads) repeats the key/value heads."""
    q, k, v = query, key, value
    heads_q, heads_kv = q.shape[-3], k.shape[-3]
    if heads_q != heads_kv:
        if heads_q % heads_kv:
            raise ValueError(f"GQA: {heads_q} query heads are not a multiple of {heads_kv} key/value heads")
        k = k.repeat_interleave(heads_q // heads_kv, dim=-3)
        v = v.repeat_interleave(heads_q // heads_kv, dim=-3)

    bias = None
    if is_causal:
        bias = causal_bias(q.shape[-2], k.shape[-2], q)
    if attn_mask is not None:
        m = mask_bias(attn_mask, lambda mask: ~mask, q)
        bias = m if bias is None else bias + m

    out, _ = scaled_dot_product_attention(q, k, v, q.shape[-1] ** -0.5 if scale is None else scale, bias, dropout_p)
    return out


# ---------------------------------------------------------------------------
# multi_head_attention_forward: nn.MultiheadAttention's kernel
# ---------------------------------------------------------------------------

def decompose_mha(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                  bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias,
                  training=True, key_padding_mask=None, need_weights=True, attn_mask=None,
                  use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None,
                  v_proj_weight=None, static_k=None, static_v=None, average_attn_weights=True,
                  is_causal=False):
    r"""``multi_head_attention_forward`` as its projections, the attention
    step per head, and the output projection; the signature of the torch
    function. Inputs are ``(T, B, E)`` (the module has already handled
    ``batch_first``). A bool mask means *do not attend* where ``True``, the
    opposite of SDPA. ``bias_k``, ``bias_v``, ``add_zero_attn`` and
    ``static_k``/``static_v`` are not decomposed."""
    H = num_heads
    for name, arg in (('bias_k', bias_k), ('bias_v', bias_v), ('static_k', static_k), ('static_v', static_v)):
        if arg is not None:
            raise NotImplementedError(f"attention decomposition does not support {name}")
    if add_zero_attn:
        raise NotImplementedError("attention decomposition does not support add_zero_attn=True")

    unbatched = query.dim() == 2
    if unbatched:
        query, key, value = query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
    T, B, E = query.shape
    S, D = key.shape[0], E // H

    # 1. Projections. Self-attention on one tensor: one packed linear, as
    # PyTorch does it. Otherwise one linear per operand, from the packed
    # weight split in three or from the three separate weights.
    if not use_separate_proj_weight and query is key and key is value:
        q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    else:
        b_q, b_k, b_v = in_proj_bias.chunk(3) if in_proj_bias is not None else (None, None, None)
        if use_separate_proj_weight:
            w_q, w_k, w_v = q_proj_weight, k_proj_weight, v_proj_weight
        else:
            w_q, w_k, w_v = in_proj_weight.chunk(3)
        q, k, v = F.linear(query, w_q, b_q), F.linear(key, w_k, b_k), F.linear(value, w_v, b_v)

    # 2. Heads: (T, B, E) -> (B*H, T, D). reshape, not view: a chunk of
    # the packed projection is not contiguous when B > 1.
    q = q.reshape(T, B * H, D).transpose(0, 1)
    k = k.reshape(S, B * H, D).transpose(0, 1)
    v = v.reshape(S, B * H, D).transpose(0, 1)

    # 3. Masks, as one additive bias of shape (B*H or 1, T, S).
    bias = None
    if is_causal:
        bias = causal_bias(T, S, q)
    if attn_mask is not None:
        m = mask_bias(attn_mask, lambda mask: mask, q)
        if m.dim() == 2:
            m = m.unsqueeze(0)
        bias = m if bias is None else bias + m
    if key_padding_mask is not None:
        kpm = mask_bias(key_padding_mask, lambda mask: mask, q)          # (B, S)
        kpm = kpm.unsqueeze(1).expand(B, H, S).reshape(B * H, 1, S)
        bias = kpm if bias is None else bias + kpm

    # 4. Attention, merge the heads, project out.
    dropout_p = dropout_p if training else 0.0
    out, weights = scaled_dot_product_attention(q, k, v, 1.0 / math.sqrt(D), bias, dropout_p)
    out = out.transpose(0, 1).reshape(T, B, E)
    out = F.linear(out, out_proj_weight, out_proj_bias)
    if unbatched:
        out = out.squeeze(1)

    if not need_weights:
        return out, None
    weights = weights.view(B, H, T, S)
    if average_attn_weights:
        weights = weights.mean(dim=1)
    return out, weights.squeeze(0) if unbatched else weights
