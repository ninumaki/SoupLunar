"""Correctness of your kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (2, 64, 1536)])
def test_rmsnorm_matches_pytorch(dtype, shape):
    from solution.rmsnorm import fast_rmsnorm

    torch.manual_seed(42)
    hidden_size = shape[-1]
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)
    eps = 1e-6

    # Reference PyTorch implementation (as in Qwen2RMSNorm)
    x_f32 = x.float()
    var = x_f32.pow(2).mean(-1, keepdim=True)
    ref = weight.float() * (x_f32 * torch.rsqrt(var + eps))
    ref = ref.to(dtype)
    ref.backward(grad)
    ref_dx, ref_dw = x.grad.clone(), weight.grad.clone()
    x.grad = weight.grad = None

    # Custom Triton implementation
    out = fast_rmsnorm(x, weight, eps)
    out.backward(grad)

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad, ref_dx, rtol=tol, atol=tol)
    torch.testing.assert_close(weight.grad, ref_dw, rtol=tol, atol=tol)

