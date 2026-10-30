"""Fused RMSNorm Triton Kernels for Qwen2.5.

Replaces standard Qwen2RMSNorm with a single-pass fused kernel (forward & backward)
operating in SRAM with float32 accumulation for Kaggle T4 GPU.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def fused_rmsnorm_fwd_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    rstd_ptr,
    stride_x,
    stride_y,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x_row = x_ptr + row_idx * stride_x
    y_row = y_ptr + row_idx * stride_y

    # Load input in fp32
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Variance and rstd (reciprocal standard deviation)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row_idx, rstd)

    # Normalized output
    y = x * rstd * w
    tl.store(y_row + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def fused_rmsnorm_bwd_kernel(
    dx_ptr,
    dy_ptr,
    x_ptr,
    w_ptr,
    rstd_ptr,
    stride_dx,
    stride_dy,
    stride_x,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    dx_row = dx_ptr + row_idx * stride_dx
    dy_row = dy_ptr + row_idx * stride_dy
    x_row = x_ptr + row_idx * stride_x

    dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)

    # Analytical gradient for RMSNorm
    w_dy = w * dy
    sum_x_w_dy = tl.sum(x * w_dy, axis=0)
    dx = rstd * (w_dy - (x / n_cols) * (rstd * rstd) * sum_x_w_dy)

    tl.store(dx_row + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


class FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        n_rows, n_cols = x_2d.shape

        block_size = triton.next_power_of_2(n_cols)
        y = torch.empty_like(x_2d)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)

        grid = (n_rows,)
        fused_rmsnorm_fwd_kernel[grid](
            x_2d,
            y,
            weight,
            rstd,
            x_2d.stride(0),
            y.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )

        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.n_cols = n_cols
        ctx.block_size = block_size
        return y.view(*orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight, rstd = ctx.saved_tensors
        dy_2d = dy.reshape(-1, ctx.n_cols).contiguous()
        n_rows = x_2d.shape[0]

        dx = torch.empty_like(x_2d)
        grid = (n_rows,)
        fused_rmsnorm_bwd_kernel[grid](
            dx,
            dy_2d,
            x_2d,
            weight,
            rstd,
            dx.stride(0),
            dy_2d.stride(0),
            x_2d.stride(0),
            ctx.n_cols,
            BLOCK_SIZE=ctx.block_size,
            num_warps=4,
        )

        # Weight gradient
        dw = (dy_2d.float() * (x_2d.float() * rstd.unsqueeze(1))).sum(dim=0).to(weight.dtype)
        return dx.view(*ctx.orig_shape), dw, None


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return FusedRMSNormFunction.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Patch Qwen2RMSNorm to use the fused Triton kernel."""
    try:
        from transformers.models.qwen2 import modeling_qwen2

        original_forward = modeling_qwen2.Qwen2RMSNorm.forward

        def forward(self, hidden_states):
            if not hidden_states.is_cuda:
                return original_forward(self, hidden_states)
            return fused_rmsnorm(hidden_states, self.weight, self.variance_epsilon)

        modeling_qwen2.Qwen2RMSNorm.forward = forward
    except (ImportError, AttributeError):
        pass

