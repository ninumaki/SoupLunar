"""Fast NF4 Dequantization Kernel in Triton for bitsandbytes 4-bit weights.

Replaces standard bitsandbytes dequantization with a fused Triton kernel,
unpacking 4-bit nibbles and multiplying with absmax scales in float32
before storing back in float16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024


@triton.jit
def fast_nf4_dequant_kernel(
    packed_ptr,
    absmax_ptr,
    code_ptr,
    out_ptr,
    total_elements,
    blocksize: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Unpack 4-bit NF4 indices, look up code, scale by absmax, store fp16."""
    pid = tl.program_id(0)
    # Each thread block handles BLOCK_SIZE output elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Each byte holds two 4-bit elements: low 4 bits (even), high 4 bits (odd)
    byte_indices = offsets // 2
    is_odd = (offsets % 2) == 1

    packed_bytes = tl.load(packed_ptr + byte_indices, mask=mask, other=0).to(tl.int32)
    nibbles = tl.where(is_odd, (packed_bytes >> 4) & 0x0F, packed_bytes & 0x0F)

    # Absmax block index: element index // blocksize (typically 64)
    absmax_indices = offsets // blocksize
    absmax = tl.load(absmax_ptr + absmax_indices, mask=mask, other=1.0).to(tl.float32)

    # NF4 lookup table: 16 code points
    code_val = tl.load(code_ptr + nibbles, mask=mask, other=0.0).to(tl.float32)

    # Dequantized value in float32, then stored as fp16
    dequant_val = code_val * absmax
    tl.store(out_ptr + offsets, dequant_val.to(out_ptr.dtype.element_ty), mask=mask)


def fast_dequantize_4bit(
    weight_data: torch.Tensor,
    quant_state: Any,
) -> torch.Tensor:
    """Triton-accelerated dequantization of a 4-bit bitsandbytes tensor."""
    out_shape = quant_state.shape
    total_elements = weight_data.numel() * 2

    out = torch.empty(out_shape, device=weight_data.device, dtype=quant_state.dtype)

    # Handle double quantization if present
    absmax = quant_state.absmax
    if getattr(quant_state, "nested", False):
        import bitsandbytes.functional as bnb_f
        absmax = bnb_f.dequantize_blockwise(
            quant_state.absmax,
            quant_state.state2,
        )

    code = quant_state.code.to(weight_data.device)
    blocksize = getattr(quant_state, "blocksize", 64)

    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    fast_nf4_dequant_kernel[grid](
        weight_data,
        absmax,
        code,
        out,
        total_elements,
        blocksize=blocksize,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


def patch_bnb_dequant():
    """Patch bitsandbytes dequantize_4bit to use the fast Triton kernel."""
    try:
        import bitsandbytes.functional as bnb_f
        original_dequant = bnb_f.dequantize_4bit

        def dequantize_4bit_triton(weight, quant_state=None, quant_type="nf4"):
            if quant_type == "nf4" and quant_state is not None and weight.is_cuda:
                try:
                    return fast_dequantize_4bit(weight, quant_state)
                except Exception:
                    return original_dequant(weight, quant_state=quant_state, quant_type=quant_type)
            return original_dequant(weight, quant_state=quant_state, quant_type=quant_type)

        bnb_f.dequantize_4bit = dequantize_4bit_triton
    except (ImportError, AttributeError):
        pass

