"""Fused Rotary Position Embedding (RoPE) Triton Kernels for Qwen2.5.

Fuses rotary position embedding forward and backward for BOTH query (`q`, 12 heads)
and key (`k`, 2 GQA KV heads) into a single Triton kernel launch per layer, reading
transposed `q`/`k` views directly via strides without intermediate `rotate_half` / `cat`
allocations.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

    class _TritonStub:
        constexpr = int

        @staticmethod
        def jit(fn=None, **_kwargs):
            return fn if fn is not None else (lambda f: f)

        @staticmethod
        def next_power_of_2(n: int) -> int:
            return 1 if n <= 1 else 1 << (n - 1).bit_length()

    triton = _TritonStub()  # type: ignore[assignment]
    tl = _TritonStub()  # type: ignore[assignment]

_PATCHED = False


@triton.jit
def fused_rope_fwd_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    out_k_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_s,
    stride_k_b,
    stride_k_h,
    stride_k_s,
    stride_cos_b,
    stride_cos_s,
    stride_out_q_b,
    stride_out_q_h,
    stride_out_q_s,
    stride_out_k_b,
    stride_out_k_h,
    stride_out_k_s,
    seq_len,
    n_heads_q: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)

    b_idx = pid_tok // seq_len
    s_idx = pid_tok % seq_len

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < half_dim
    cols2 = cols + half_dim

    cos_base = cos_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s
    sin_base = sin_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s
    cos1 = tl.load(cos_base + cols, mask=mask, other=0.0).to(tl.float32)
    cos2 = tl.load(cos_base + cols2, mask=mask, other=0.0).to(tl.float32)
    sin1 = tl.load(sin_base + cols, mask=mask, other=0.0).to(tl.float32)
    sin2 = tl.load(sin_base + cols2, mask=mask, other=0.0).to(tl.float32)

    if pid_head < n_heads_q:
        in_base = q_ptr + b_idx * stride_q_b + pid_head * stride_q_h + s_idx * stride_q_s
        out_base = out_q_ptr + b_idx * stride_out_q_b + pid_head * stride_out_q_h + s_idx * stride_out_q_s
        x1 = tl.load(in_base + cols, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(in_base + cols2, mask=mask, other=0.0).to(tl.float32)
        y1 = x1 * cos1 - x2 * sin1
        y2 = x2 * cos2 + x1 * sin2
        tl.store(out_base + cols, y1.to(out_q_ptr.dtype.element_ty), mask=mask)
        tl.store(out_base + cols2, y2.to(out_q_ptr.dtype.element_ty), mask=mask)
    else:
        k_head = pid_head - n_heads_q
        in_base = k_ptr + b_idx * stride_k_b + k_head * stride_k_h + s_idx * stride_k_s
        out_base = out_k_ptr + b_idx * stride_out_k_b + k_head * stride_out_k_h + s_idx * stride_out_k_s
        x1 = tl.load(in_base + cols, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(in_base + cols2, mask=mask, other=0.0).to(tl.float32)
        y1 = x1 * cos1 - x2 * sin1
        y2 = x2 * cos2 + x1 * sin2
        tl.store(out_base + cols, y1.to(out_k_ptr.dtype.element_ty), mask=mask)
        tl.store(out_base + cols2, y2.to(out_k_ptr.dtype.element_ty), mask=mask)


@triton.jit
def fused_rope_bwd_kernel(
    dq_in_ptr,
    dk_in_ptr,
    cos_ptr,
    sin_ptr,
    dq_out_ptr,
    dk_out_ptr,
    stride_dqi_b,
    stride_dqi_h,
    stride_dqi_s,
    stride_dki_b,
    stride_dki_h,
    stride_dki_s,
    stride_cos_b,
    stride_cos_s,
    stride_dqo_b,
    stride_dqo_h,
    stride_dqo_s,
    stride_dko_b,
    stride_dko_h,
    stride_dko_s,
    seq_len,
    n_heads_q: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)

    b_idx = pid_tok // seq_len
    s_idx = pid_tok % seq_len

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < half_dim
    cols2 = cols + half_dim

    cos_base = cos_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s
    sin_base = sin_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s
    cos1 = tl.load(cos_base + cols, mask=mask, other=0.0).to(tl.float32)
    cos2 = tl.load(cos_base + cols2, mask=mask, other=0.0).to(tl.float32)
    sin1 = tl.load(sin_base + cols, mask=mask, other=0.0).to(tl.float32)
    sin2 = tl.load(sin_base + cols2, mask=mask, other=0.0).to(tl.float32)

    if pid_head < n_heads_q:
        in_base = dq_in_ptr + b_idx * stride_dqi_b + pid_head * stride_dqi_h + s_idx * stride_dqi_s
        out_base = dq_out_ptr + b_idx * stride_dqo_b + pid_head * stride_dqo_h + s_idx * stride_dqo_s
        dy1 = tl.load(in_base + cols, mask=mask, other=0.0).to(tl.float32)
        dy2 = tl.load(in_base + cols2, mask=mask, other=0.0).to(tl.float32)
        dx1 = dy1 * cos1 + dy2 * sin2
        dx2 = dy2 * cos2 - dy1 * sin1
        tl.store(out_base + cols, dx1.to(dq_out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_base + cols2, dx2.to(dq_out_ptr.dtype.element_ty), mask=mask)
    else:
        k_head = pid_head - n_heads_q
        in_base = dk_in_ptr + b_idx * stride_dki_b + k_head * stride_dki_h + s_idx * stride_dki_s
        out_base = dk_out_ptr + b_idx * stride_dko_b + k_head * stride_dko_h + s_idx * stride_dko_s
        dy1 = tl.load(in_base + cols, mask=mask, other=0.0).to(tl.float32)
        dy2 = tl.load(in_base + cols2, mask=mask, other=0.0).to(tl.float32)
        dx1 = dy1 * cos1 + dy2 * sin2
        dx2 = dy2 * cos2 - dy1 * sin1
        tl.store(out_base + cols, dx1.to(dk_out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_base + cols2, dx2.to(dk_out_ptr.dtype.element_ty), mask=mask)


class FusedRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        bsz, n_heads_q, seq_len, head_dim = q.shape
        n_heads_k = k.shape[1]
        half_dim = head_dim // 2
        block_size = triton.next_power_of_2(half_dim)

        cos_c = cos.contiguous()
        sin_c = sin.contiguous()
        stride_cos_b = 0 if cos_c.shape[0] == 1 else cos_c.stride(0)
        stride_cos_s = cos_c.stride(1)

        out_q_dtype = torch.promote_types(q.dtype, cos_c.dtype)
        out_k_dtype = torch.promote_types(k.dtype, cos_c.dtype)
        out_q = torch.empty((bsz, n_heads_q, seq_len, head_dim), device=q.device, dtype=out_q_dtype)
        out_k = torch.empty((bsz, n_heads_k, seq_len, head_dim), device=k.device, dtype=out_k_dtype)

        grid = (bsz * seq_len, n_heads_q + n_heads_k)
        fused_rope_fwd_kernel[grid](
            q,
            k,
            cos_c,
            sin_c,
            out_q,
            out_k,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            stride_cos_b,
            stride_cos_s,
            out_q.stride(0),
            out_q.stride(1),
            out_q.stride(2),
            out_k.stride(0),
            out_k.stride(1),
            out_k.stride(2),
            seq_len,
            n_heads_q=n_heads_q,
            half_dim=half_dim,
            BLOCK_SIZE=block_size,
            num_warps=2,
        )

        ctx.save_for_backward(cos_c, sin_c)
        ctx.q_dtype = q.dtype
        ctx.k_dtype = k.dtype
        ctx.bsz = bsz
        ctx.seq_len = seq_len
        ctx.n_heads_q = n_heads_q
        ctx.n_heads_k = n_heads_k
        ctx.head_dim = head_dim
        ctx.half_dim = half_dim
        ctx.block_size = block_size
        ctx.stride_cos_b = stride_cos_b
        ctx.stride_cos_s = stride_cos_s
        return out_q, out_k

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_k: torch.Tensor):
        cos_c, sin_c = ctx.saved_tensors
        if grad_q.stride(-1) != 1:
            grad_q = grad_q.contiguous()
        if grad_k.stride(-1) != 1:
            grad_k = grad_k.contiguous()

        dq = torch.empty(
            (ctx.bsz, ctx.n_heads_q, ctx.seq_len, ctx.head_dim),
            device=grad_q.device,
            dtype=ctx.q_dtype,
        )
        dk = torch.empty(
            (ctx.bsz, ctx.n_heads_k, ctx.seq_len, ctx.head_dim),
            device=grad_k.device,
            dtype=ctx.k_dtype,
        )

        grid = (ctx.bsz * ctx.seq_len, ctx.n_heads_q + ctx.n_heads_k)
        fused_rope_bwd_kernel[grid](
            grad_q,
            grad_k,
            cos_c,
            sin_c,
            dq,
            dk,
            grad_q.stride(0),
            grad_q.stride(1),
            grad_q.stride(2),
            grad_k.stride(0),
            grad_k.stride(1),
            grad_k.stride(2),
            ctx.stride_cos_b,
            ctx.stride_cos_s,
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            ctx.seq_len,
            n_heads_q=ctx.n_heads_q,
            half_dim=ctx.half_dim,
            BLOCK_SIZE=ctx.block_size,
            num_warps=2,
        )
        return dq, dk, None, None


def fused_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return FusedRoPEFunction.apply(q, k, cos, sin)


def patch_qwen2_rope() -> None:
    """Patch Qwen2 `apply_rotary_pos_emb` to launch the fused Triton RoPE forward/backward kernels."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from transformers.models.qwen2 import modeling_qwen2

        original_apply_rope = modeling_qwen2.apply_rotary_pos_emb

        def fast_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            if (
                _HAS_TRITON
                and q.is_cuda
                and k.is_cuda
                and unsqueeze_dim == 1
                and q.ndim == 4
                and k.ndim == 4
                and cos.ndim == 3
                and sin.ndim == 3
                and q.shape[0] == k.shape[0]
                and q.shape[2] == k.shape[2] == cos.shape[1]
                and q.shape[3] == k.shape[3] == cos.shape[2]
                and (q.shape[3] % 2) == 0
                and q.stride(-1) == 1
                and k.stride(-1) == 1
                and not cos.requires_grad
                and not sin.requires_grad
            ):
                return fused_rope(q, k, cos, sin)
            return original_apply_rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

        modeling_qwen2.apply_rotary_pos_emb = fast_apply_rotary_pos_emb
        if hasattr(modeling_qwen2.Qwen2Attention, "apply_rotary_pos_emb"):
            modeling_qwen2.Qwen2Attention.apply_rotary_pos_emb = staticmethod(fast_apply_rotary_pos_emb)
        _PATCHED = True
    except (ImportError, AttributeError):
        pass
