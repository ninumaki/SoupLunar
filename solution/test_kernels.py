"""Correctness of LunarTeam's Triton kernels and batch optimizations against PyTorch / HF / BnB.

Run on GPU:  `python -m pytest solution -q`
Run on CPU:  CPU batch/padding & RewindTrainer tests execute; CUDA Triton tests skip cleanly.
"""

from __future__ import annotations

import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


# ---------------------------------------------------------------------------
# 1. CPU & GPU Batch / Padding / Rewind Compatibility Tests (Darkhan - Role 3)
# ---------------------------------------------------------------------------


def test_truncate_batch_to_max_len_preserves_tokens_and_aligns_8():
    from solution.dynamic_padding import is_pure_right_padded_2d, truncate_batch_to_max_len

    bsz, orig_len = 4, 512
    lengths = [40, 115, 89, 219]  # max = 219 -> ceil_8(219) = 224
    input_ids = torch.zeros((bsz, orig_len), dtype=torch.long)
    attention_mask = torch.zeros((bsz, orig_len), dtype=torch.long)
    labels = torch.full((bsz, orig_len), -100, dtype=torch.long)

    for i, length in enumerate(lengths):
        input_ids[i, :length] = torch.arange(1, length + 1)
        attention_mask[i, :length] = 1
        labels[i, 10:length] = input_ids[i, 10:length]

    orig_num_tokens = int(attention_mask.sum().item())
    orig_labelled = int((labels != -100).sum().item())

    assert is_pure_right_padded_2d(attention_mask) is True

    out = truncate_batch_to_max_len(
        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels},
        multiple_of=8,
    )
    assert out["input_ids"].shape == (4, 224)
    assert out["attention_mask"].shape == (4, 224)
    assert out["labels"].shape == (4, 224)
    assert int(out["attention_mask"].sum().item()) == orig_num_tokens
    assert int((out["labels"] != -100).sum().item()) == orig_labelled
    assert is_pure_right_padded_2d(out["attention_mask"]) is True

    # Left-padded or non-monotonic masks must NOT be marked pure right-padded
    left_padded = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.long)
    hole_mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
    empty_row = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.long)
    assert is_pure_right_padded_2d(left_padded) is False
    assert is_pure_right_padded_2d(hole_mask) is False
    assert is_pure_right_padded_2d(empty_row) is False


def test_right_padded_causal_sdpa_elision_exact_equivalence():
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    from solution.dynamic_padding import patch_dynamic_padding, truncate_batch_to_max_len

    torch.manual_seed(7)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attn_implementation="sdpa",
    )
    model_ref = Qwen2ForCausalLM(cfg).eval()

    bsz, orig_len = 3, 64
    lengths = [11, 27, 19]  # max = 27 -> aligned_len = 32
    input_ids = torch.zeros((bsz, orig_len), dtype=torch.long)
    attention_mask = torch.zeros((bsz, orig_len), dtype=torch.long)
    labels = torch.full((bsz, orig_len), -100, dtype=torch.long)
    for i, length in enumerate(lengths):
        input_ids[i, :length] = torch.randint(1, 250, (length,))
        attention_mask[i, :length] = 1
        labels[i, 3:length] = input_ids[i, 3:length]

    out_ref = model_ref(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    out_ref.loss.backward()
    grad_ref = model_ref.model.layers[0].self_attn.q_proj.weight.grad.clone()
    model_ref.zero_grad(set_to_none=True)

    patch_dynamic_padding()
    trimmed = truncate_batch_to_max_len(
        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels},
        multiple_of=8,
    )
    trimmed["attention_mask"]._soup_right_padded = True

    out_opt = model_ref(**trimmed)
    out_opt.loss.backward()
    grad_opt = model_ref.model.layers[0].self_attn.q_proj.weight.grad.clone()

    torch.testing.assert_close(out_opt.loss, out_ref.loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad_opt, grad_ref, rtol=1e-5, atol=1e-6)
    for i, length in enumerate(lengths):
        torch.testing.assert_close(
            out_opt.logits[i, :length],
            out_ref.logits[i, :length],
            rtol=1e-5,
            atol=1e-6,
        )


def test_dynamic_padding_with_rewind_trainer_records_exact_row_losses(tmp_path):
    """Verify Trainer._prepare_inputs truncation happens BEFORE RewindTrainer.compute_loss
    reads `labels = inputs["labels"]`, keeping `rewind.jsonl` row_tokens and row_loss exact."""
    from soup_cli.trainer.rewind_hf import attach_rewind_state, make_rewind_trainer_class
    from transformers import Trainer, TrainingArguments
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    from solution.dynamic_padding import patch_dynamic_padding

    patch_dynamic_padding()
    torch.manual_seed(13)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attn_implementation="sdpa",
    )
    model = Qwen2ForCausalLM(cfg).train()

    class _Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self.rows = []
            for length in (12, 19, 14, 21):  # max=21 -> ceil_8(21)=24 << 64
                ids = torch.zeros(64, dtype=torch.long)
                mask = torch.zeros(64, dtype=torch.long)
                lbl = torch.full((64,), -100, dtype=torch.long)
                ids[:length] = torch.randint(1, 120, (length,))
                mask[:length] = 1
                lbl[4:length] = ids[4:length]
                self.rows.append({"input_ids": ids, "attention_mask": mask, "labels": lbl})

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    records = []

    class _Sink:
        def record_batch(self, **kwargs):
            records.append(kwargs)

    RewindCls = make_rewind_trainer_class(Trainer)
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=2,
        max_steps=1,
        use_cpu=True,
        report_to="none",
        save_strategy="no",
    )
    trainer = RewindCls(model=model, args=args, train_dataset=_Dataset())
    state = attach_rewind_state(trainer, _Sink())

    dl = trainer.get_train_dataloader()
    raw_batch = next(iter(dl))
    assert raw_batch["input_ids"].shape[1] == 64

    prepared = trainer._prepare_inputs(raw_batch)
    assert prepared["input_ids"].shape[1] <= 24
    assert prepared["input_ids"].shape[1] % 8 == 0
    assert getattr(prepared["attention_mask"], "_soup_right_padded", False) is True

    loss = trainer.compute_loss(model, prepared)
    assert torch.isfinite(loss)
    assert not state.failed
    assert len(records) == 1
    assert len(records[0]["rows"]) == 2
    assert all(t > 0 for t in records[0]["row_tokens"])


def test_kbit_mixed_precision_keeps_embeddings_and_biases_fp16_and_norms_fp32():
    """Verify 4-bit preparation keeps frozen embed_tokens/lm_head/biases in fp16 and norms in fp32."""
    import peft
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    from solution.kbit_mp import patch_kbit_mixed_precision

    patch_kbit_mixed_precision()
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(cfg).to(dtype=torch.float16)
    model.is_loaded_in_4bit = True  # simulate 4-bit loaded flag

    prepared = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    assert all(not p.requires_grad for p in prepared.parameters())
    assert prepared.model.embed_tokens.weight.dtype == torch.float16
    assert prepared.lm_head.weight.dtype == torch.float16
    assert prepared.model.layers[0].self_attn.q_proj.bias.dtype == torch.float16
    assert prepared.model.layers[0].input_layernorm.weight.dtype == torch.float32
    assert prepared.model.layers[0].post_attention_layernorm.weight.dtype == torch.float32
    assert prepared.model.norm.weight.dtype == torch.float32


# ---------------------------------------------------------------------------
# 2. CUDA Triton Kernel Correctness Tests (Roles 1, 2, 3)
# ---------------------------------------------------------------------------


@cuda_only
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
    fwd_tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=fwd_tol, atol=fwd_tol)

    # Backward uses group-128 symmetric INT8 compressed activations (~48.4% VRAM saved)
    cos_dgate = torch.nn.functional.cosine_similarity(gate.grad.float().flatten(), ref_dgate.flatten(), dim=0)
    cos_dup = torch.nn.functional.cosine_similarity(up.grad.float().flatten(), ref_dup.flatten(), dim=0)
    assert float(cos_dgate) > 0.9998
    assert float(cos_dup) > 0.9998
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=2.5e-2, atol=5e-2)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=2.5e-2, atol=5e-2)

    with torch.no_grad():
        out_ng = swiglu(gate, up)
        torch.testing.assert_close(out_ng.float(), ref, rtol=fwd_tol, atol=fwd_tol)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_cross_entropy_matches_pytorch(dtype):
    from solution.fused_cross_entropy import fused_chunk_terms, fused_chunked_causal_lm_loss

    torch.manual_seed(42)
    b, c, h = 2, 160, 1536
    vocab_size = 151936

    head = torch.nn.Linear(h, vocab_size, bias=False, device="cuda", dtype=dtype)
    head.weight.requires_grad_(False)
    piece = torch.randn((b, c, h), device="cuda", dtype=dtype, requires_grad=True)
    target = torch.randint(0, vocab_size, (b, c), device="cuda", dtype=torch.long)
    target[0, 10:40] = -100
    target[1, 120:] = -100

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

    total_loss, entropy, _token_loss, correct, piece_grad = fused_chunk_terms(
        head, piece, target, ignore_index=-100, denominator=labelled, with_grad=True
    )

    tol = 3e-3 if dtype == torch.float16 else 1e-4
    assert torch.isfinite(entropy).all(), "entropy contains non-finite values (NaN/Inf)"
    torch.testing.assert_close(total_loss, ref_loss, rtol=tol, atol=tol)
    torch.testing.assert_close(entropy.float(), ref_entropy.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(correct.long(), ref_correct.long())
    torch.testing.assert_close(piece_grad.float(), ref_piece_grad.float(), rtol=tol, atol=tol)

    # Also test multi-chunk autograd wrapper (`chunk=64` across `c=160`) under AMP GradScaler scale=65536.0
    piece2 = piece.detach().clone().requires_grad_(True)
    chunk_loss, step = fused_chunked_causal_lm_loss(
        piece2, head, target, chunk=64, shift_labels=target
    )
    assert torch.isfinite(step.token_entropy).all()
    (chunk_loss * 65536.0).backward()
    assert torch.isfinite(piece2.grad).all(), "scaled backward overflowed under GradScaler=65536"
    torch.testing.assert_close(chunk_loss.float(), ref_mean_loss.float(), rtol=tol, atol=tol)
    torch.testing.assert_close((piece2.grad.float() / 65536.0), ref_piece_grad.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(step.correct.long(), ref_correct.long())


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("train_weight", [True, False])
def test_fused_rmsnorm_matches_pytorch(dtype, train_weight):
    from solution.fused_rmsnorm import fused_rmsnorm

    torch.manual_seed(42)
    shape = (4, 128, 1536)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=train_weight)
    dy = torch.randn(shape, device="cuda", dtype=dtype)

    variance = x.float().pow(2).mean(-1, keepdim=True)
    ref_y = w * (x.float() * torch.rsqrt(variance + 1e-6)).to(dtype)
    ref_y.backward(dy)
    ref_dx = x.grad.clone()
    ref_dw = w.grad.clone() if train_weight else None
    x.grad = None
    if train_weight:
        w.grad = None

    y = fused_rmsnorm(x, w, eps=1e-6)
    y.backward(dy)

    tol = 3e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(y.float(), ref_y.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), ref_dx.float(), rtol=tol, atol=tol)
    if train_weight:
        torch.testing.assert_close(w.grad.float(), ref_dw.float(), rtol=tol, atol=tol)
    else:
        assert w.grad is None


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_rope_matches_pytorch(dtype):
    from transformers.models.qwen2.modeling_qwen2 import rotate_half

    from solution.fused_rope import fused_rope

    torch.manual_seed(17)
    bsz, seq_len, n_q, n_k, head_dim = 4, 137, 12, 2, 128
    # Match Qwen2Attention transposed layout: [B, S, H, D].transpose(1, 2) -> [B, H, S, D]
    q_raw = torch.randn((bsz, seq_len, n_q, head_dim), device="cuda", dtype=dtype, requires_grad=True)
    k_raw = torch.randn((bsz, seq_len, n_k, head_dim), device="cuda", dtype=dtype, requires_grad=True)
    cos = torch.randn((1, seq_len, head_dim), device="cuda", dtype=dtype)
    sin = torch.randn((1, seq_len, head_dim), device="cuda", dtype=dtype)
    dq_out = torch.randn((bsz, n_q, seq_len, head_dim), device="cuda", dtype=dtype)
    dk_out = torch.randn((bsz, n_k, seq_len, head_dim), device="cuda", dtype=dtype)

    q_ref = q_raw.transpose(1, 2)
    k_ref = k_raw.transpose(1, 2)
    ref_q = (q_ref * cos.unsqueeze(1)) + (rotate_half(q_ref) * sin.unsqueeze(1))
    ref_k = (k_ref * cos.unsqueeze(1)) + (rotate_half(k_ref) * sin.unsqueeze(1))
    loss_ref = (ref_q.float() * dq_out.float()).sum() + (ref_k.float() * dk_out.float()).sum()
    loss_ref.backward()
    ref_dq_raw = q_raw.grad.clone()
    ref_dk_raw = k_raw.grad.clone()
    q_raw.grad = k_raw.grad = None

    out_q, out_k = fused_rope(q_raw.transpose(1, 2), k_raw.transpose(1, 2), cos, sin)
    assert out_q.transpose(1, 2).is_contiguous()
    assert out_k.transpose(1, 2).is_contiguous()
    loss_tri = (out_q.float() * dq_out.float()).sum() + (out_k.float() * dk_out.float()).sum()
    loss_tri.backward()

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out_q.float(), ref_q.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(out_k.float(), ref_k.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(q_raw.grad.float(), ref_dq_raw.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(k_raw.grad.float(), ref_dk_raw.float(), rtol=tol, atol=tol)


@cuda_only
@pytest.mark.parametrize("compress_statistics", [False, True])
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_fast_nf4_dequant_matches_bitsandbytes(compress_statistics, quant_type):
    import bitsandbytes.functional as bnb_f

    from solution.nf4_dequant import fast_dequantize_4bit

    torch.manual_seed(99)
    w = torch.randn((8960, 1536), device="cuda", dtype=torch.float16)
    packed, qstate = bnb_f.quantize_4bit(
        w,
        blocksize=64,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
    )
    ref = bnb_f.dequantize_4bit(packed, quant_state=qstate)
    tri = fast_dequantize_4bit(packed, qstate)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)
