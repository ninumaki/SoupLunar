"""Your solution for Soup Kernel Challenge (LunarTeam).

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. Patch whatever you need (transformers, peft,
bitsandbytes, Soup itself) so the training run uses your Triton kernels.
"""

from .dynamic_padding import patch_dynamic_padding
from .fused_cross_entropy import patch_soup_fused_cross_entropy
from .fused_rmsnorm import patch_qwen2_rmsnorm
from .fused_rope import patch_qwen2_rope
from .kbit_mp import patch_kbit_mixed_precision
from .nf4_dequant import patch_bnb_dequant
from .swiglu import patch_qwen2_mlp

_APPLIED = False


def apply() -> None:
    """Apply all LunarTeam Triton kernels and batch/mixed-precision optimizations idempotently."""
    global _APPLIED
    if _APPLIED:
        return

    # 1. Bekbolat's Core GPU Kernels
    patch_bnb_dequant()               # Byte-Packed Fused NF4/FP4 + Double-Quant Triton Dequantization (Fwd + Bwd)
    patch_soup_fused_cross_entropy()  # Fused Chunked Cross-Entropy + Entropy/Accuracy Triton Kernels

    # 2. Nurdaulet's Element-Wise & Activation-Compressed Model Kernels
    patch_qwen2_mlp()                 # Fused SwiGLU with Group-128 Symmetric INT8 Activation Compression
    patch_qwen2_rmsnorm()             # Fused RMSNorm (Forward + Backward) in Triton
    patch_qwen2_rope()                # Fused Q+K RoPE Rotary Position Embedding (Forward + Backward) in Triton

    # 3. Darkhan's Batch, Attention Layout & Mixed-Precision Optimizations
    patch_dynamic_padding()           # Pre-H2D Pad Truncation, 8-Token Tensor Core Alignment & Causal SDPA GQA Unlock
    patch_kbit_mixed_precision()      # Norm-Only FP32 K-Bit Prep (FP16 Residual/Embed/LMHead) & Zero-Clone LoRA 4-Bit

    _APPLIED = True
