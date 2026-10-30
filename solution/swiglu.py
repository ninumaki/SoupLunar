"""Fused SwiGLU with Group-128 Symmetric INT8 Activation Compression in Triton.

Replaces the starter example kernels with LunarTeam's 2D block-tiled INT8 activation-compressed
SwiGLU (`int8_swiglu_fwd_kernel` and `int8_swiglu_bwd_kernel`) tailored for Qwen2.5 on Tesla T4:

  - Why it matters on Qwen2.5-1.5B (`intermediate_size = 8960`, 28 layers):
    Uncompressed `SwiGLU` saves full `(gate, up)` tensors in `fp16` (`2 x [4, 512, 8960] x 2B` =
    `70.0 MiB/layer x 28 = 1,960 MiB`), accounting for nearly one-third of total QLoRA VRAM!
  - Forward (`int8_swiglu_fwd_kernel`):
    Processes 2D `[GROUPS_PER_CTA, GROUP_SIZE]` (`[8, 128]`) tiles in registers, computing exact
    `fp32`-accumulated `silu(gate) * up` (`0.000%` forward quantization error on loss/entropy/eval)
    while simultaneously quantizing `(gate, up)` into symmetric group-128 `int8` (`[-127, 127]`)
    with `fp32` per-group scales — saving ~950 MiB of peak VRAM.
  - Backward (`int8_swiglu_bwd_kernel`):
    Streams compact `int8` `(gate_q, up_q)` and per-group scales from GDDR6 (cutting saved-tensor
    memory read bandwidth by ~48% on bandwidth-bound T4), reconstructs `(gate, up)` in SRAM, and
    fuses `dgate` and `dup` in a single pass.

Inspired by blockwise 8-bit activation quantization for memory-efficient LLM training
(e.g., Liu et al., GACT, ICML 2022; Xi et al., Jetfire, ICML 2024; Section 9 task hint on int8
activation compression). Implemented from scratch in Triton.
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
        def cdiv(x: int, y: int) -> int:
            return (x + y - 1) // y

    triton = _TritonStub()  # type: ignore[assignment]
    tl = _TritonStub()  # type: ignore[assignment]

GROUP_SIZE = 128
GROUPS_PER_CTA = 8
_PATCHED = False


@triton.jit
def int8_swiglu_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    gate_q_ptr,
    up_q_ptr,
    gate_scale_ptr,
    up_scale_ptr,
    n_elements,
    n_groups,
    GROUPS_PER_CTA: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SAVE_QUANT: tl.constexpr,
):
    """2D group-tiled SwiGLU forward with fused symmetric INT8 activation compression."""
    pid = tl.program_id(0)
    group_offs = pid * GROUPS_PER_CTA + tl.arange(0, GROUPS_PER_CTA)
    col_offs = tl.arange(0, GROUP_SIZE)
    elem_offs = group_offs[:, None] * GROUP_SIZE + col_offs[None, :]
    elem_mask = elem_offs < n_elements

    gate = tl.load(gate_ptr + elem_offs, mask=elem_mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + elem_offs, mask=elem_mask, other=0.0).to(tl.float32)

    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * up
    tl.store(out_ptr + elem_offs, out.to(out_ptr.dtype.element_ty), mask=elem_mask)

    if SAVE_QUANT:
        group_mask = group_offs < n_groups
        abs_gate = tl.abs(gate)
        abs_up = tl.abs(up)

        g_amax = tl.max(abs_gate, axis=1)
        u_amax = tl.max(abs_up, axis=1)

        g_scale = g_amax * (1.0 / 127.0)
        u_scale = u_amax * (1.0 / 127.0)

        g_inv = tl.where(g_amax > 0.0, 127.0 / g_amax, 0.0)
        u_inv = tl.where(u_amax > 0.0, 127.0 / u_amax, 0.0)

        # Non-negative [0.5, 127.5] -> .to(tl.int32) truncates toward zero == exact round-half-up
        g_abs_q = (tl.minimum(abs_gate * g_inv[:, None], 127.0) + 0.5).to(tl.int32)
        u_abs_q = (tl.minimum(abs_up * u_inv[:, None], 127.0) + 0.5).to(tl.int32)

        g_q = tl.where(gate >= 0.0, g_abs_q, -g_abs_q).to(tl.int8)
        u_q = tl.where(up >= 0.0, u_abs_q, -u_abs_q).to(tl.int8)

        tl.store(gate_q_ptr + elem_offs, g_q, mask=elem_mask)
        tl.store(up_q_ptr + elem_offs, u_q, mask=elem_mask)
        tl.store(gate_scale_ptr + group_offs, g_scale, mask=group_mask)
        tl.store(up_scale_ptr + group_offs, u_scale, mask=group_mask)


@triton.jit
def int8_swiglu_bwd_kernel(
    grad_ptr,
    gate_q_ptr,
    up_q_ptr,
    gate_scale_ptr,
    up_scale_ptr,
    dgate_ptr,
    dup_ptr,
    n_elements,
    n_groups,
    GROUPS_PER_CTA: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """2D group-tiled SwiGLU backward with on-the-fly INT8 activation dequantization."""
    pid = tl.program_id(0)
    group_offs = pid * GROUPS_PER_CTA + tl.arange(0, GROUPS_PER_CTA)
    col_offs = tl.arange(0, GROUP_SIZE)
    elem_offs = group_offs[:, None] * GROUP_SIZE + col_offs[None, :]
    elem_mask = elem_offs < n_elements
    group_mask = group_offs < n_groups

    grad = tl.load(grad_ptr + elem_offs, mask=elem_mask, other=0.0).to(tl.float32)
    g_q = tl.load(gate_q_ptr + elem_offs, mask=elem_mask, other=0).to(tl.float32)
    u_q = tl.load(up_q_ptr + elem_offs, mask=elem_mask, other=0).to(tl.float32)
    g_scale = tl.load(gate_scale_ptr + group_offs, mask=group_mask, other=0.0).to(tl.float32)
    u_scale = tl.load(up_scale_ptr + group_offs, mask=group_mask, other=0.0).to(tl.float32)

    gate = g_q * g_scale[:, None]
    up = u_q * u_scale[:, None]

    sig = tl.sigmoid(gate)
    silu_gate = gate * sig
    dup = grad * silu_gate
    dgate = grad * up * (sig + silu_gate * (1.0 - sig))

    tl.store(dgate_ptr + elem_offs, dgate.to(dgate_ptr.dtype.element_ty), mask=elem_mask)
    tl.store(dup_ptr + elem_offs, dup.to(dup_ptr.dtype.element_ty), mask=elem_mask)


class Int8CompressedSwiGLU(torch.autograd.Function):
    """Exact fp16/fp32 forward `silu(gate) * up` with group-128 INT8 compressed backward state."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        orig_shape = gate.shape
        out_dtype = gate.dtype
        gate_c = gate.contiguous()
        up_c = up.to(out_dtype).contiguous() if up.dtype != out_dtype else up.contiguous()

        n_elements = gate_c.numel()
        n_groups = triton.cdiv(n_elements, GROUP_SIZE)
        out = torch.empty_like(gate_c)

        need_grad = torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad)
        if need_grad:
            gate_q = torch.empty(n_elements, dtype=torch.int8, device=gate_c.device)
            up_q = torch.empty(n_elements, dtype=torch.int8, device=gate_c.device)
            gate_scale = torch.empty(n_groups, dtype=torch.float32, device=gate_c.device)
            up_scale = torch.empty(n_groups, dtype=torch.float32, device=gate_c.device)
        else:
            gate_q = gate_c
            up_q = up_c
            gate_scale = gate_c
            up_scale = up_c

        grid = (triton.cdiv(n_groups, GROUPS_PER_CTA),)
        int8_swiglu_fwd_kernel[grid](
            gate_c,
            up_c,
            out,
            gate_q,
            up_q,
            gate_scale,
            up_scale,
            n_elements,
            n_groups,
            GROUPS_PER_CTA=GROUPS_PER_CTA,
            GROUP_SIZE=GROUP_SIZE,
            SAVE_QUANT=need_grad,
            num_warps=4,
        )

        ctx.orig_shape = orig_shape
        ctx.out_dtype = out_dtype
        ctx.n_elements = n_elements
        ctx.n_groups = n_groups
        if need_grad:
            ctx.save_for_backward(gate_q, up_q, gate_scale, up_scale)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        gate_q, up_q, gate_scale, up_scale = ctx.saved_tensors
        grad_c = grad.to(ctx.out_dtype).contiguous() if grad.dtype != ctx.out_dtype else grad.contiguous()

        dgate = torch.empty(ctx.orig_shape, dtype=ctx.out_dtype, device=grad_c.device)
        dup = torch.empty(ctx.orig_shape, dtype=ctx.out_dtype, device=grad_c.device)

        grid = (triton.cdiv(ctx.n_groups, GROUPS_PER_CTA),)
        int8_swiglu_bwd_kernel[grid](
            grad_c,
            gate_q,
            up_q,
            gate_scale,
            up_scale,
            dgate,
            dup,
            ctx.n_elements,
            ctx.n_groups,
            GROUPS_PER_CTA=GROUPS_PER_CTA,
            GROUP_SIZE=GROUP_SIZE,
            num_warps=4,
        )
        return dgate, dup


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return Int8CompressedSwiGLU.apply(gate, up)


def patch_qwen2_mlp() -> None:
    """Route `Qwen2MLP.forward` through INT8 activation-compressed Triton SwiGLU."""
    global _PATCHED
    if _PATCHED:
        return

    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if not _HAS_TRITON or act != "silu" or not x.is_cuda:
            return original(self, x)
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

    modeling_qwen2.Qwen2MLP.forward = forward
    _PATCHED = True
