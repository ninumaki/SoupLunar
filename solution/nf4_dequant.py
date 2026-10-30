"""Fast Byte-Packed Fused NF4/FP4 Dequantization Kernel in Triton for bitsandbytes 4-bit weights.

Optimized for Tesla T4 (`sm_75`, 15GB GDDR6, fp16):
  - Byte-packed 2x-elements-per-thread vectorization (`BLOCK_BYTES = 512` -> `1024` weights/CTA):
    loads each packed `uint8` byte ONCE (unpacking high nibble `2*i` and low nibble `2*i + 1` in
    registers) and computes each block scale ONCE per byte pair (`half_block = blocksize // 2`),
    halving `packed_ptr`, `absmax_8bit`, `absmax_code`, and `absmax` memory transactions and
    interleaving pairs via `tl.join(...).reshape(BLOCK_BYTES * 2)` into a coalesced vector store.
  - Fuses nested double-quantization (`absmax_8bit`, `state2.code`, `state2.absmax`, `+ offset`)
    in a single Triton pass — eliminating 2 extra CUDA kernel launches per projection.
  - Caches validated tensor views directly on `QuantState` (`O(1)` lookup across ~25,000 calls)
    and reuses a single per-device transient scratch buffer (`~26.3 MiB`) across both forward
    (`_dequant_linear_fallback`) and backward (`MatMul4Bit.backward`) GEMMs when autograd is off.
"""

from __future__ import annotations

import math
from typing import Any, Optional

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

BLOCK_BYTES = 512  # 512 packed bytes = 1024 dequantized weights per CTA
_PATCHED = False
_CODE_CACHE: dict[tuple[str, torch.device], torch.Tensor] = {}
_DEQUANT_WORKSPACE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_code_table(quant_type: str, device: torch.device) -> torch.Tensor:
    key = (quant_type, device)
    table = _CODE_CACHE.get(key)
    if table is None:
        import bitsandbytes.functional as bnb_f

        table = bnb_f.get_4bit_type(quant_type, device=device).to(dtype=torch.float32).contiguous()
        _CODE_CACHE[key] = table
    return table


def _get_workspace(device: torch.device, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    """Return a reusable transient buffer sliced to `shape` (safe ONLY when autograd is disabled)."""
    numel = shape[0] * shape[1] if len(shape) == 2 else int(math.prod(shape))
    key = (device, dtype)
    buf = _DEQUANT_WORKSPACE.get(key)
    if buf is None or buf.numel() < numel:
        # Pre-size to at least Qwen2.5 MLP `[8960, 1536]` (13,762,560 elems = 26.25 MiB in fp16)
        alloc_elems = max(numel, 8960 * 1536)
        buf = torch.empty(alloc_elems, device=device, dtype=dtype)
        _DEQUANT_WORKSPACE[key] = buf
    return buf[:numel].view(shape)


@triton.jit
def fast_nf4_dequant_kernel(
    packed_ptr,
    absmax_ptr,
    absmax_8bit_ptr,
    absmax_code_ptr,
    offset_ptr,
    code_ptr,
    out_ptr,
    n_bytes,
    total_elements,
    half_block: tl.constexpr,
    blocksize2: tl.constexpr,
    NESTED: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    """Unpack 2 NF4/FP4 weights per byte, share blockwise/nested scale, interleave, and store."""
    pid = tl.program_id(0)
    byte_offs = pid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    byte_mask = byte_offs < n_bytes

    packed_bytes = tl.load(packed_ptr + byte_offs, mask=byte_mask, other=0).to(tl.int32)
    hi_nibbles = (packed_bytes >> 4) & 0x0F
    lo_nibbles = packed_bytes & 0x0F

    absmax_indices = byte_offs // half_block
    if NESTED:
        q_abs = tl.load(absmax_8bit_ptr + absmax_indices, mask=byte_mask, other=0).to(tl.int32)
        code2 = tl.load(absmax_code_ptr + q_abs, mask=byte_mask, other=0.0).to(tl.float32)
        scale2 = tl.load(absmax_ptr + (absmax_indices // blocksize2), mask=byte_mask, other=0.0).to(tl.float32)
        abs_offset = tl.load(offset_ptr).to(tl.float32)
        absmax = code2 * scale2 + abs_offset
    else:
        absmax = tl.load(absmax_ptr + absmax_indices, mask=byte_mask, other=0.0).to(tl.float32)

    val_hi = tl.load(code_ptr + hi_nibbles, mask=byte_mask, other=0.0).to(tl.float32) * absmax
    val_lo = tl.load(code_ptr + lo_nibbles, mask=byte_mask, other=0.0).to(tl.float32) * absmax

    pair = tl.reshape(tl.join(val_hi, val_lo), (BLOCK_BYTES * 2,))
    out_offs = pid * (BLOCK_BYTES * 2) + tl.arange(0, BLOCK_BYTES * 2)
    out_mask = out_offs < total_elements
    tl.store(out_ptr + out_offs, pair.to(out_ptr.dtype.element_ty), mask=out_mask)


def _launch_triton_dequant(
    packed: torch.Tensor,
    out_shape: Any,
    out_dtype: torch.dtype,
    blocksize: int,
    code: torch.Tensor,
    absmax: torch.Tensor,
    *,
    absmax_8bit: Optional[torch.Tensor] = None,
    absmax_code: Optional[torch.Tensor] = None,
    absmax_offset: Optional[torch.Tensor] = None,
    blocksize2: int = 256,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    packed_u8 = packed.view(torch.uint8) if packed.dtype != torch.uint8 else packed
    if not packed_u8.is_contiguous():
        packed_u8 = packed_u8.contiguous()

    total_elements = (
        out_shape[0] * out_shape[1]
        if isinstance(out_shape, (tuple, list)) and len(out_shape) == 2
        else int(math.prod(out_shape))
    )
    n_bytes = (total_elements + 1) >> 1
    if out is None:
        out = torch.empty(out_shape, device=packed.device, dtype=out_dtype)

    nested = absmax_8bit is not None
    if nested:
        assert absmax_code is not None and absmax_offset is not None
        absmax_8bit_c = absmax_8bit if absmax_8bit.is_contiguous() else absmax_8bit.contiguous()
        absmax_code_c = absmax_code if absmax_code.is_contiguous() else absmax_code.contiguous()
        offset_f32 = absmax_offset if absmax_offset.dtype == torch.float32 else absmax_offset.to(torch.float32)
    else:
        absmax_8bit_c = packed_u8
        absmax_code_c = code
        offset_f32 = code

    absmax_c = absmax if absmax.is_contiguous() else absmax.contiguous()
    code_c = code if code.is_contiguous() else code.contiguous()

    grid = (triton.cdiv(n_bytes, BLOCK_BYTES),)
    fast_nf4_dequant_kernel[grid](
        packed_u8,
        absmax_c,
        absmax_8bit_c,
        absmax_code_c,
        offset_f32,
        code_c,
        out,
        n_bytes,
        total_elements,
        half_block=blocksize >> 1,
        blocksize2=blocksize2,
        NESTED=nested,
        BLOCK_BYTES=BLOCK_BYTES,
        num_warps=4,
    )
    return out


def fast_dequantize_4bit(
    weight_data: torch.Tensor,
    quant_state: Any,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Triton-accelerated dequantization of a 4-bit bitsandbytes tensor with O(1) metadata cache."""
    target_dtype = out_dtype if out_dtype is not None else quant_state.dtype
    cached = getattr(quant_state, "_lunar_dequant_meta", None)
    w_ptr = weight_data.data_ptr()

    if cached is not None and cached[0] == w_ptr:
        (
            _,
            packed_u8,
            out_shape,
            total_elements,
            n_bytes,
            half_block,
            blocksize2,
            nested,
            code_c,
            absmax_c,
            absmax_8bit_c,
            absmax_code_c,
            offset_f32,
        ) = cached
    else:
        out_shape = tuple(quant_state.shape)
        total_elements = out_shape[0] * out_shape[1] if len(out_shape) == 2 else int(math.prod(out_shape))
        n_bytes = (total_elements + 1) >> 1
        blocksize = int(getattr(quant_state, "blocksize", 64))
        half_block = blocksize >> 1
        quant_type = getattr(quant_state, "quant_type", "nf4")

        code = getattr(quant_state, "code", None)
        if code is None or code.device != weight_data.device or code.dtype != torch.float32:
            code_c = _get_code_table(quant_type, weight_data.device)
        else:
            code_c = code if code.is_contiguous() else code.contiguous()

        packed_u8 = weight_data.view(torch.uint8) if weight_data.dtype != torch.uint8 else weight_data
        if not packed_u8.is_contiguous():
            packed_u8 = packed_u8.contiguous()

        nested = bool(getattr(quant_state, "nested", False))
        if nested:
            state2 = quant_state.state2
            absmax_c = state2.absmax if state2.absmax.is_contiguous() else state2.absmax.contiguous()
            absmax_8bit_c = (
                quant_state.absmax if quant_state.absmax.is_contiguous() else quant_state.absmax.contiguous()
            )
            absmax_code_c = state2.code if state2.code.is_contiguous() else state2.code.contiguous()
            raw_off = quant_state.offset
            offset_f32 = raw_off if raw_off.dtype == torch.float32 else raw_off.to(torch.float32)
            blocksize2 = int(getattr(state2, "blocksize", 256))
        else:
            absmax_c = quant_state.absmax if quant_state.absmax.is_contiguous() else quant_state.absmax.contiguous()
            absmax_8bit_c = packed_u8
            absmax_code_c = code_c
            offset_f32 = code_c
            blocksize2 = 256

        try:
            quant_state._lunar_dequant_meta = (
                w_ptr,
                packed_u8,
                out_shape,
                total_elements,
                n_bytes,
                half_block,
                blocksize2,
                nested,
                code_c,
                absmax_c,
                absmax_8bit_c,
                absmax_code_c,
                offset_f32,
            )
        except AttributeError:
            pass

    if out is None:
        out = torch.empty(out_shape, device=weight_data.device, dtype=target_dtype)

    grid = (triton.cdiv(n_bytes, BLOCK_BYTES),)
    fast_nf4_dequant_kernel[grid](
        packed_u8,
        absmax_c,
        absmax_8bit_c,
        absmax_code_c,
        offset_f32,
        code_c,
        out,
        n_bytes,
        total_elements,
        half_block=half_block,
        blocksize2=blocksize2,
        NESTED=nested,
        BLOCK_BYTES=BLOCK_BYTES,
        num_warps=4,
    )

    if weight_data.ndim == 2 and weight_data.shape[0] == 1 and weight_data.shape[1] > 1:
        return out.t()
    return out


def patch_bnb_dequant() -> None:
    """Patch bitsandbytes `dequantize_4bit`, `MatMul4Bit.backward`, and `_dequant_linear_fallback`."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import bitsandbytes.functional as bnb_f

        original_dequant = bnb_f.dequantize_4bit

        def dequantize_4bit_triton(
            A: torch.Tensor,
            quant_state: Any = None,
            absmax: Optional[torch.Tensor] = None,
            out: Optional[torch.Tensor] = None,
            blocksize: Optional[int] = None,
            quant_type: str = "fp4",
        ) -> torch.Tensor:
            if (
                _HAS_TRITON
                and A.is_cuda
                and quant_state is not None
                and getattr(quant_state, "quant_type", quant_type) in ("nf4", "fp4")
                and getattr(quant_state, "dtype", None) in (torch.float16, torch.float32)
                and getattr(quant_state, "blocksize", 64) in (32, 64, 128, 256, 512, 1024, 2048, 4096)
            ):
                return fast_dequantize_4bit(A, quant_state, out=out)
            return original_dequant(
                A,
                quant_state=quant_state,
                absmax=absmax,
                out=out,
                blocksize=blocksize,
                quant_type=quant_type,
            )

        bnb_f.dequantize_4bit = dequantize_4bit_triton
    except (ImportError, AttributeError):
        pass

    # Patch MatMul4Bit.backward to write directly into the reusable scratch buffer in grad_output.dtype
    try:
        import bitsandbytes.autograd._functions as bnb_autograd

        original_bwd = bnb_autograd.MatMul4Bit.backward

        @staticmethod  # type: ignore[misc]
        def triton_matmul4bit_backward(ctx, grad_output: torch.Tensor):
            if ctx.is_empty:
                return original_bwd(ctx, grad_output)
            req_grad_a, _, _, req_grad_bias, _ = ctx.needs_input_grad
            _, B = ctx.tensors
            qstate = ctx.state
            if (
                _HAS_TRITON
                and req_grad_a
                and B is not None
                and B.is_cuda
                and qstate is not None
                and grad_output.dtype in (torch.float16, torch.float32)
                and getattr(qstate, "quant_type", None) in ("nf4", "fp4")
                and getattr(qstate, "blocksize", 64) in (32, 64, 128, 256, 512, 1024, 2048, 4096)
                and not (B.ndim == 2 and B.shape[0] == 1 and B.shape[1] > 1)
            ):
                grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias) if req_grad_bias else None
                out_shape = tuple(qstate.shape)
                ws = _get_workspace(B.device, grad_output.dtype, out_shape)
                B_dq = fast_dequantize_4bit(B, qstate, out=ws, out_dtype=grad_output.dtype)
                grad_a = torch.matmul(grad_output, B_dq)
                return grad_a, None, None, grad_bias, None
            return original_bwd(ctx, grad_output)

        bnb_autograd.MatMul4Bit.backward = triton_matmul4bit_backward
    except (ImportError, AttributeError):
        pass

    try:
        import bitsandbytes.backends.cuda.ops as bnb_cuda_ops

        original_fallback = bnb_cuda_ops._dequant_linear_fallback

        def triton_dequant_linear_fallback(
            A: torch.Tensor,
            B: torch.Tensor,
            shapeB: Any,
            absmax: torch.Tensor,
            blocksize: int,
            quant_type: str,
            bias: Optional[torch.Tensor] = None,
            absmax_8bit: Optional[torch.Tensor] = None,
            absmax_code: Optional[torch.Tensor] = None,
            absmax_offset: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if (
                _HAS_TRITON
                and A.is_cuda
                and A.dtype in (torch.float16, torch.float32)
                and quant_type in ("nf4", "fp4")
                and blocksize in (32, 64, 128, 256, 512, 1024, 2048, 4096)
            ):
                code = _get_code_table(quant_type, A.device)
                out_shape = tuple(shapeB)
                can_reuse_ws = not (torch.is_grad_enabled() and A.requires_grad)
                ws = _get_workspace(A.device, A.dtype, out_shape) if can_reuse_ws else None
                B_dq = _launch_triton_dequant(
                    B,
                    out_shape=out_shape,
                    out_dtype=A.dtype,
                    blocksize=blocksize,
                    code=code,
                    absmax=absmax,
                    absmax_8bit=absmax_8bit,
                    absmax_code=absmax_code,
                    absmax_offset=absmax_offset,
                    blocksize2=256,
                    out=ws,
                )
                return torch.nn.functional.linear(A, B_dq, bias)
            return original_fallback(
                A,
                B,
                shapeB,
                absmax,
                blocksize,
                quant_type,
                bias=bias,
                absmax_8bit=absmax_8bit,
                absmax_code=absmax_code,
                absmax_offset=absmax_offset,
            )

        bnb_cuda_ops._dequant_linear_fallback = triton_dequant_linear_fallback
    except (ImportError, AttributeError):
        pass

    _PATCHED = True
