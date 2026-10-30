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
def test_fused_cross_entropy_matches_pytorch(dtype):
    from solution.fused_cross_entropy import fused_chunk_terms

    torch.manual_seed(42)
    b, c, h = 2, 64, 1536
    vocab_size = 152064

    head = torch.nn.Linear(h, vocab_size, bias=False, device="cuda", dtype=dtype)
    piece = torch.randn((b, c, h), device="cuda", dtype=dtype, requires_grad=True)
    target = torch.randint(0, vocab_size, (b, c), device="cuda", dtype=torch.long)
    target[0, 10:20] = -100

    ref_logits = head(piece)
    ref_loss = torch.nn.functional.cross_entropy(
        ref_logits.float().reshape(-1, vocab_size),
        target.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    labelled = (target != -100).sum()
    ref_mean_loss = ref_loss / labelled.clamp(min=1)
    ref_mean_loss.backward()
    ref_piece_grad = piece.grad.clone()
    piece.grad = None

    with torch.no_grad():
        ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        ref_entropy = -(torch.exp(ref_log_probs) * ref_log_probs).sum(-1)
        ref_correct = ((ref_logits.argmax(dim=-1) == target) & (target != -100)).sum()

    total_loss, entropy, token_loss, correct, piece_grad = fused_chunk_terms(
        head, piece, target, ignore_index=-100, denominator=labelled, with_grad=True
    )

    tol = 3e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(total_loss, ref_loss, rtol=tol, atol=tol)
    torch.testing.assert_close(entropy.float(), ref_entropy.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(correct.long(), ref_correct.long())
    torch.testing.assert_close(piece_grad.float(), ref_piece_grad.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_rmsnorm_matches_pytorch(dtype):
    from solution.fused_rmsnorm import fused_rmsnorm

    torch.manual_seed(42)
    shape = (4, 128, 1536)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.ones(shape[-1], device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn(shape, device="cuda", dtype=dtype)

    # Reference
    variance = x.float().pow(2).mean(-1, keepdim=True)
    ref_y = (x.float() * torch.rsqrt(variance + 1e-6)) * w.float()
    ref_y.backward(dy.float())
    ref_dx = x.grad.clone()
    ref_dw = w.grad.clone()
    x.grad = w.grad = None

    # Triton Fused
    y = fused_rmsnorm(x, w, eps=1e-6)
    y.backward(dy)

    tol = 3e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(y.float(), ref_y, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), ref_dx, rtol=tol, atol=tol)
    torch.testing.assert_close(w.grad.float(), ref_dw, rtol=tol, atol=tol)
