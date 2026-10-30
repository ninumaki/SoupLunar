"""Fused RMSNorm in Triton (forward and backward) for Qwen2RMSNorm.

Replaces standard PyTorch Qwen2RMSNorm forward/backward, avoiding multiple DRAM round-trips
and casting overheads (FP32 variance calculation and FP16 I/O directly in SRAM/registers).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    rrms_ptr,
    stride_x_row,
    stride_out_row,
    stride_rrms_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_row = x_ptr + row_idx * stride_x_row
    out_row = out_ptr + row_idx * stride_out_row

    x = tl.load(x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / n_cols
    rrms = tl.rsqrt(var + eps)

    tl.store(rrms_ptr + row_idx * stride_rrms_row, rrms)

    norm_x = x * rrms
    out = norm_x * w
    tl.store(out_row + col_offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def rmsnorm_bwd_kernel(
    grad_ptr,
    x_ptr,
    w_ptr,
    rrms_ptr,
    dx_ptr,
    stride_grad_row,
    stride_x_row,
    stride_rrms_row,
    stride_dx_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    grad_row = grad_ptr + row_idx * stride_grad_row
    x_row = x_ptr + row_idx * stride_x_row
    dx_row = dx_ptr + row_idx * stride_dx_row

    dy = tl.load(grad_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.load(rrms_ptr + row_idx * stride_rrms_row)

    dy_w = dy * w
    # dx = rrms * (dy*w - (x * rrms^2 / n_cols) * sum(dy*w*x))
    sum_dy_w_x = tl.sum(dy_w * x, axis=0)
    dx = rrms * (dy_w - (x * (rrms * rrms) / n_cols) * sum_dy_w_x)

    tl.store(dx_row + col_offsets, dx.to(dx_ptr.dtype.element_ty), mask=mask)


class FastRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        weight = weight.contiguous()
        n_rows, n_cols = x_2d.shape

        out = torch.empty_like(x_2d)
        rrms = torch.empty(n_rows, dtype=torch.float32, device=x.device)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        rmsnorm_fwd_kernel[(n_rows,)](
            x_2d,
            weight,
            out,
            rrms,
            x_2d.stride(0),
            out.stride(0),
            1,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(x_2d, weight, rrms)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.shape = x.shape
        return out.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x_2d, weight, rrms = ctx.saved_tensors
        n_rows, n_cols = x_2d.shape
        grad_2d = grad_output.reshape(-1, n_cols).contiguous()

        dx = torch.empty_like(x_2d)
        rmsnorm_bwd_kernel[(n_rows,)](
            grad_2d,
            x_2d,
            weight,
            rrms,
            dx,
            grad_2d.stride(0),
            x_2d.stride(0),
            1,
            dx.stride(0),
            n_cols,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )
        # dweight calculation: sum(grad * norm_x)
        # norm_x = x * rrms[:, None]
        norm_x = x_2d * rrms.unsqueeze(-1).to(x_2d.dtype)
        dweight = (grad_2d * norm_x).sum(dim=0).to(weight.dtype)
        return dx.reshape(ctx.shape), dweight, None


def fast_rmsnorm(x, weight, eps=1e-6):
    return FastRMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Patches Qwen2RMSNorm with our custom Triton kernel."""
    from transformers.models.qwen2 import modeling_qwen2

    original_forward = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return original_forward(self, hidden_states)
        return fast_rmsnorm(hidden_states, self.weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = forward
