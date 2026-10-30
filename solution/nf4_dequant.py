"""Fast Fused NF4/FP4 Dequantization Kernel in Triton for bitsandbytes 4-bit weights.

Optimized for Tesla T4 (`sm_75`, 15GB GDDR6, fp16):
  - Fixes bitsandbytes 4-bit nibble packing order: even index -> high 4 bits `(byte >> 4) & 0x0F`,
    odd index -> low 4 bits `byte & 0x0F`.
  - Fuses nested double-quantization (`absmax_8bit`, `state2.code`, `state2.absmax`, and `+ offset`)
    directly inside the single Triton kernel pass — eliminating 2 extra CUDA kernel launches per
    projection (`cdequantize_blockwise` + `aten::add`).
  - Hooks BOTH `bitsandbytes.functional.dequantize_4bit` (`MatMul4Bit.backward`) AND
    `bitsandbytes.backends.cuda.ops._dequant_linear_fallback` (`gemm_4bit` forward on T4 where
    `M = batch * seq > 64`), accelerating all 588 4-bit dequantizations per QLoRA step.
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

BLOCK_SIZE = 1024
_PATCHED = False
_CODE_CACHE: dict[tuple[str, torch.device], torch.Tensor] = {}


def _get_code_table(quant_type: str, device: torch.device) -> torch.Tensor:
    key = (quant_type, device)
    table = _CODE_CACHE.get(key)
    if table is None:
        import bitsandbytes.functional as bnb_f

        table = bnb_f.get_4bit_type(quant_type, device=device).to(dtype=torch.float32).contiguous()
        _CODE_CACHE[key] = table
    return table


@triton.jit
def fast_nf4_dequant_kernel(
    packed_ptr,
    absmax_ptr,
    absmax_8bit_ptr,
    absmax_code_ptr,
    offset_ptr,
    code_ptr,
    out_ptr,
    total_elements,
    blocksize: tl.constexpr,
    blocksize2: tl.constexpr,
    NESTED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Unpack 4-bit indices, fuse optional nested absmax dequant + offset, scale in fp32, store fp16/fp32."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # bitsandbytes packs even elements in high 4 bits (>> 4) and odd elements in low 4 bits (& 0x0F)
    byte_indices = offsets >> 1
    is_odd = (offsets & 1) == 1

    packed_bytes = tl.load(packed_ptr + byte_indices, mask=mask, other=0).to(tl.int32)
    nibbles = tl.where(is_odd, packed_bytes & 0x0F, (packed_bytes >> 4) & 0x0F)

    absmax_indices = offsets // blocksize
    if NESTED:
        q_abs = tl.load(absmax_8bit_ptr + absmax_indices, mask=mask, other=0).to(tl.int32)
        code2 = tl.load(absmax_code_ptr + q_abs, mask=mask, other=0.0).to(tl.float32)
        scale2 = tl.load(absmax_ptr + (absmax_indices // blocksize2), mask=mask, other=0.0).to(tl.float32)
        abs_offset = tl.load(offset_ptr).to(tl.float32)
        absmax = code2 * scale2 + abs_offset
    else:
        absmax = tl.load(absmax_ptr + absmax_indices, mask=mask, other=0.0).to(tl.float32)

    code_val = tl.load(code_ptr + nibbles, mask=mask, other=0.0).to(tl.float32)
    dequant_val = code_val * absmax
    tl.store(out_ptr + offsets, dequant_val.to(out_ptr.dtype.element_ty), mask=mask)


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

    total_elements = int(math.prod(out_shape))
    if out is None:
        out = torch.empty(out_shape, device=packed.device, dtype=out_dtype)

    nested = absmax_8bit is not None
    if nested:
        assert absmax_code is not None and absmax_offset is not None
        absmax_8bit_c = absmax_8bit.contiguous()
        absmax_code_c = absmax_code.contiguous()
        offset_f32 = absmax_offset if absmax_offset.dtype == torch.float32 else absmax_offset.to(torch.float32)
    else:
        absmax_8bit_c = packed_u8
        absmax_code_c = code
        offset_f32 = code

    absmax_c = absmax.contiguous()
    code_c = code.contiguous()

    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    fast_nf4_dequant_kernel[grid](
        packed_u8,
        absmax_c,
        absmax_8bit_c,
        absmax_code_c,
        offset_f32,
        code_c,
        out,
        total_elements,
        blocksize=blocksize,
        blocksize2=blocksize2,
        NESTED=nested,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


def fast_dequantize_4bit(
    weight_data: torch.Tensor,
    quant_state: Any,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Triton-accelerated dequantization of a 4-bit bitsandbytes tensor with nested double-quant support."""
    out_shape = tuple(quant_state.shape)
    out_dtype = quant_state.dtype
    blocksize = int(getattr(quant_state, "blocksize", 64))
    quant_type = getattr(quant_state, "quant_type", "nf4")
    code = getattr(quant_state, "code", None)
    if code is None or code.device != weight_data.device or code.dtype != torch.float32:
        code = _get_code_table(quant_type, weight_data.device)

    if getattr(quant_state, "nested", False):
        state2 = quant_state.state2
        res = _launch_triton_dequant(
            weight_data,
            out_shape=out_shape,
            out_dtype=out_dtype,
            blocksize=blocksize,
            code=code,
            absmax=state2.absmax,
            absmax_8bit=quant_state.absmax,
            absmax_code=state2.code,
            absmax_offset=quant_state.offset,
            blocksize2=int(getattr(state2, "blocksize", 256)),
            out=out,
        )
    else:
        res = _launch_triton_dequant(
            weight_data,
            out_shape=out_shape,
            out_dtype=out_dtype,
            blocksize=blocksize,
            code=code,
            absmax=quant_state.absmax,
            out=out,
        )

    if weight_data.shape[0] == 1:
        return res.t()
    return res


def patch_bnb_dequant() -> None:
    """Patch bitsandbytes backward (`dequantize_4bit`) and forward (`_dequant_linear_fallback`) paths."""
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
            ):
                code = _get_code_table(quant_type, A.device)
                B_dq = _launch_triton_dequant(
                    B,
                    out_shape=shapeB,
                    out_dtype=A.dtype,
                    blocksize=blocksize,
                    code=code,
                    absmax=absmax,
                    absmax_8bit=absmax_8bit,
                    absmax_code=absmax_code,
                    absmax_offset=absmax_offset,
                    blocksize2=256,
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
