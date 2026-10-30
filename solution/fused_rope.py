"""Fused Rotary Position Embedding (RoPE) Triton Kernel for Qwen2.5.

Combines (q * cos) + (rotate_half(q) * sin) into a single Triton kernel pass.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def fused_rope_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    out_k_ptr,
    stride_q_bs,
    stride_q_h,
    stride_q_s,
    stride_q_d,
    stride_cos_bs,
    stride_cos_s,
    stride_cos_d,
    n_tokens,
    n_heads_q,
    n_heads_k,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid // n_heads_q
    head_idx = pid % n_heads_q

    if token_idx >= n_tokens:
        return

    half_dim = head_dim // 2
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < half_dim

    # Offsets for first half [0, half_dim) and second half [half_dim, head_dim)
    idx_first = cols
    idx_second = cols + half_dim

    # Load cos and sin for this token
    cos1 = tl.load(cos_ptr + token_idx * stride_cos_s + idx_first * stride_cos_d, mask=mask, other=0.0).to(tl.float32)
    cos2 = tl.load(cos_ptr + token_idx * stride_cos_s + idx_second * stride_cos_d, mask=mask, other=0.0).to(tl.float32)
    sin1 = tl.load(sin_ptr + token_idx * stride_cos_s + idx_first * stride_cos_d, mask=mask, other=0.0).to(tl.float32)
    sin2 = tl.load(sin_ptr + token_idx * stride_cos_s + idx_second * stride_cos_d, mask=mask, other=0.0).to(tl.float32)

    # Load Q
    q_base = q_ptr + token_idx * stride_q_s + head_idx * stride_q_h
    q1 = tl.load(q_base + idx_first * stride_q_d, mask=mask, other=0.0).to(tl.float32)
    q2 = tl.load(q_base + idx_second * stride_q_d, mask=mask, other=0.0).to(tl.float32)

    # rotate_half(q): [-q2, q1]
    out_q1 = q1 * cos1 - q2 * sin1
    out_q2 = q2 * cos2 + q1 * sin2

    out_q_base = out_q_ptr + token_idx * stride_q_s + head_idx * stride_q_h
    tl.store(out_q_base + idx_first * stride_q_d, out_q1.to(out_q_ptr.dtype.element_ty), mask=mask)
    tl.store(out_q_base + idx_second * stride_q_d, out_q2.to(out_q_ptr.dtype.element_ty), mask=mask)


def patch_qwen2_rope():
    """Patch Qwen2 apply_rotary_pos_emb if supported."""
    try:
        from transformers.models.qwen2 import modeling_qwen2

        def fast_rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def fast_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            # Fused multiply-add in float16
            q_embed = torch.addcmul(q * cos, fast_rotate_half(q), sin)
            k_embed = torch.addcmul(k * cos, fast_rotate_half(k), sin)
            return q_embed, k_embed

        modeling_qwen2.apply_rotary_pos_emb = fast_apply_rotary_pos_emb
    except (ImportError, AttributeError):
        pass

