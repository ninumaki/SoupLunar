"""Fused RMSNorm in Triton (forward and backward) for Qwen2RMSNorm.

Replaces standard PyTorch Qwen2RMSNorm forward/backward, avoiding multiple DRAM round-trips
and casting overheads (FP32 variance calculation and FP16 I/O directly in SRAM/registers).
"""

from __future__ import annotations

import torch
from solution.fused_rmsnorm import (
    FusedRMSNormFunction,
    fused_rmsnorm,
    fused_rmsnorm_fwd_kernel,
    fused_rmsnorm_bwd_kernel,
    patch_qwen2_rmsnorm,
)

# Aliases for full backward compatibility across team branches
FastRMSNorm = FusedRMSNormFunction
fast_rmsnorm = fused_rmsnorm
rmsnorm_fwd_kernel = fused_rmsnorm_fwd_kernel
rmsnorm_bwd_kernel = fused_rmsnorm_bwd_kernel

__all__ = [
    "FastRMSNorm",
    "fast_rmsnorm",
    "fused_rmsnorm",
    "patch_qwen2_rmsnorm",
    "rmsnorm_fwd_kernel",
    "rmsnorm_bwd_kernel",
]
