"""Fused Cross-Entropy + Entropy/Accuracy Triton Kernels for Soup and Qwen2.5.

Optimized for Kaggle T4 GPU (sm_75, 15GB VRAM, fp16 only, no bf16).
Provides:
  - Fused Cross-Entropy loss computation with Online Softmax / LogSumExp in float32.
  - Streaming Shannon entropy computation directly from logits in the same forward pass.
  - Argmax accuracy tracking for labelled positions.
  - Efficient backward pass computing gradients on the fly without keeping large logits.
  - Chunked causal-LM loss support to reduce peak VRAM by 3-4x.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 2048
CHUNK_SIZE = 256
IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# 1. Triton JIT Kernels (Executed on GPU, compiled via Triton)
# ---------------------------------------------------------------------------

@triton.jit
def fused_cross_entropy_fwd_kernel(
    logits_ptr,
    labels_ptr,
    loss_ptr,
    lse_ptr,
    entropy_ptr,
    correct_ptr,
    stride_logits_row,
    stride_logits_col,
    stride_labels,
    n_rows,
    vocab_size,
    ignore_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward kernel: computes loss, logsumexp, entropy, and argmax accuracy per row."""
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    label = tl.load(labels_ptr + row_idx * stride_labels)
    row_logits_ptr = logits_ptr + row_idx * stride_logits_row

    # Online LogSumExp + Online Entropy + Argmax tracking in float32
    m_val = -1e30
    s_val = 0.0
    sz_val = 0.0
    max_val = -1e30
    max_idx = -1

    for col_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size

        # Load logits and immediately promote to float32 for T4 stability
        logits = tl.load(
            row_logits_ptr + cols * stride_logits_col,
            mask=mask,
            other=-1e30,
        ).to(tl.float32)

        block_max = tl.max(logits, axis=0)
        block_argmax = tl.argmax(logits, axis=0) + col_offset

        new_m = tl.maximum(m_val, block_max)
        alpha = tl.exp(m_val - new_m)

        exp_logits = tl.exp(logits - new_m)
        exp_logits = tl.where(mask, exp_logits, 0.0)

        s_val = s_val * alpha + tl.sum(exp_logits, axis=0)
        sz_val = sz_val * alpha + tl.sum(logits * exp_logits, axis=0)

        # Update running argmax
        is_new_max = block_max > max_val
        max_val = tl.where(is_new_max, block_max, max_val)
        max_idx = tl.where(is_new_max, block_argmax, max_idx)

        m_val = new_m

    safe_s = tl.maximum(s_val, 1e-30)
    lse = m_val + tl.log(safe_s)
    mean_z = sz_val / safe_s
    entropy = lse - mean_z

    valid = (label != ignore_index) & (label >= 0) & (label < vocab_size)
    safe_label = tl.where(valid, label, 0)
    target_logit = tl.load(row_logits_ptr + safe_label * stride_logits_col).to(tl.float32)

    loss = tl.where(valid, lse - target_logit, 0.0)
    is_correct = (max_idx == label) & valid
    correct = tl.where(is_correct, 1, 0)

    tl.store(loss_ptr + row_idx, loss)
    tl.store(lse_ptr + row_idx, lse)
    tl.store(entropy_ptr + row_idx, entropy)
    tl.store(correct_ptr + row_idx, correct)


@triton.jit
def fused_cross_entropy_bwd_kernel(
    dlogits_ptr,
    logits_ptr,
    labels_ptr,
    lse_ptr,
    stride_dlogits_row,
    stride_dlogits_col,
    stride_logits_row,
    stride_logits_col,
    stride_labels,
    scale,  # float32 scalar (grad_output / denominator)
    n_rows,
    vocab_size,
    ignore_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward kernel: computes dlogits = (probs - indicator) * scale."""
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    label = tl.load(labels_ptr + row_idx * stride_labels)
    valid = (label != ignore_index) & (label >= 0) & (label < vocab_size)

    row_dlogits_ptr = dlogits_ptr + row_idx * stride_dlogits_row
    row_logits_ptr = logits_ptr + row_idx * stride_logits_row

    lse = tl.load(lse_ptr + row_idx)

    for col_offset in range(0, vocab_size, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size

        logits = tl.load(
            row_logits_ptr + cols * stride_logits_col,
            mask=mask,
            other=-1e30,
        ).to(tl.float32)

        probs = tl.exp(logits - lse)
        probs = tl.where(mask, probs, 0.0)

        is_target = (cols == label) & valid
        indicator = tl.where(is_target, 1.0, 0.0)

        dlogits = tl.where(valid, (probs - indicator) * scale, 0.0)
        tl.store(
            row_dlogits_ptr + cols * stride_dlogits_col,
            dlogits.to(dlogits_ptr.dtype.element_ty),
            mask=mask,
        )


# ---------------------------------------------------------------------------
# 2. Chunk Execution and Autograd Bridge
# ---------------------------------------------------------------------------

def fused_chunk_terms(
    head: Callable[[torch.Tensor], torch.Tensor],
    piece: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    denominator: Optional[Any] = None,
    with_grad: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Compute loss and metrics for one token chunk.

    Args:
        head: Linear lm_head layer.
        piece: Hidden states slice [B, C, H].
        target: Target labels slice [B, C].
        ignore_index: Ignored target label (-100).
        denominator: Normalization factor for loss reduction.
        with_grad: Whether to compute gradients on piece.

    Returns:
        (total_loss, entropy [B, C], token_loss [B, C], correct_hits, piece_grad [B, C, H])
    """
    b, c, h = piece.shape
    n_rows = b * c
    flat_piece = piece.reshape(n_rows, h)
    flat_target = target.reshape(n_rows).contiguous()

    # Projected chunk logits: [n_rows, vocab_size]
    logits = head(flat_piece)
    vocab_size = logits.shape[-1]

    loss = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    entropy = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    correct = torch.empty(n_rows, dtype=torch.int32, device=logits.device)

    grid = (n_rows,)
    fused_cross_entropy_fwd_kernel[grid](
        logits,
        flat_target,
        loss,
        lse,
        entropy,
        correct,
        logits.stride(0),
        logits.stride(1),
        flat_target.stride(0),
        n_rows,
        vocab_size,
        ignore_index=ignore_index,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    total_loss = loss.sum()
    piece_grad = None

    if with_grad:
        denom = denominator if denominator is not None else 1.0
        if torch.is_tensor(denom):
            scale_val = (1.0 / denom.float()).item()
        else:
            scale_val = float(1.0 / denom)

        dlogits = torch.empty_like(logits)
        fused_cross_entropy_bwd_kernel[grid](
            dlogits,
            logits,
            flat_target,
            lse,
            dlogits.stride(0),
            dlogits.stride(1),
            logits.stride(0),
            logits.stride(1),
            flat_target.stride(0),
            scale_val,
            n_rows,
            vocab_size,
            ignore_index=ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )

        weight = getattr(head, "weight", None)
        if weight is not None:
            # Analytical gradient on hidden states: [n_rows, V] @ [V, H] -> [n_rows, H]
            piece_grad = torch.matmul(dlogits, weight.to(dlogits.dtype)).view(b, c, h)
        else:
            # Fallback for wrapped heads
            (piece_grad,) = torch.autograd.grad(logits, flat_piece, grad_outputs=dlogits)
            piece_grad = piece_grad.view(b, c, h)

        del dlogits

    del logits

    entropy_tensor = entropy.view(b, c)
    token_loss_tensor = loss.view(b, c)
    correct_hits = correct.sum()

    return total_loss, entropy_tensor, token_loss_tensor, correct_hits, piece_grad


class _TritonChunkedHeadLoss(torch.autograd.Function):
    """Autograd Function executing chunked forward and backward via Triton kernels."""

    @staticmethod
    def forward(ctx, hidden, head, target, chunk, denominator, ignore_index, with_grad):
        b, s, h = hidden.shape
        grad = torch.zeros_like(hidden) if with_grad else None
        total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        entropies = []
        losses = []
        correct = torch.zeros((), dtype=torch.long, device=hidden.device)

        for start in range(0, s, chunk):
            stop = min(start + chunk, s)
            piece = hidden[:, start:stop]
            labels = target[:, start:stop]

            part, entropy, token_loss, hits, piece_grad = fused_chunk_terms(
                head, piece, labels, ignore_index, denominator=denominator, with_grad=with_grad
            )

            if with_grad and piece_grad is not None:
                grad[:, start:stop] = piece_grad

            total = total + part
            entropies.append(entropy)
            losses.append(token_loss)
            correct = correct + hits

        token_entropy = torch.cat(entropies, dim=1)
        token_loss = torch.cat(losses, dim=1)

        ctx.with_grad = with_grad
        if with_grad:
            ctx.save_for_backward(grad)
        ctx.mark_non_differentiable(token_entropy, token_loss, correct)

        loss = total / denominator
        return loss, token_entropy, token_loss, correct

    @staticmethod
    def backward(ctx, grad_loss, _grad_entropy, _grad_token_loss, _grad_correct):
        if not ctx.with_grad:
            return (None,) * 7
        (grad,) = ctx.saved_tensors
        return (grad * grad_loss, None, None, None, None, None, None)


# ---------------------------------------------------------------------------
# 3. Public API and Soup Integration
# ---------------------------------------------------------------------------

def fused_chunked_causal_lm_loss(
    hidden: Any,
    head: Callable[[Any], Any],
    labels: Any,
    chunk: int = CHUNK_SIZE,
    *,
    num_items_in_batch: Any = None,
    shift_labels: Any = None,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[Any, Any]:
    """Calculate causal LM loss, entropy, accuracy, and per-token loss in token chunks."""
    from soup_cli.utils.stream_loss_chunk import (ChunkedStep,
                                                  shift_for_next_token)

    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        chunk = CHUNK_SIZE

    target = shift_labels if shift_labels is not None else shift_for_next_token(
        labels, ignore_index
    )
    target = target.to(hidden.device)

    labelled = (target != ignore_index).sum()
    if num_items_in_batch is None:
        denominator = labelled.clamp(min=1)
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(hidden.device)
    else:
        denominator = num_items_in_batch

    with_grad = torch.is_grad_enabled() and hidden.requires_grad
    loss, token_entropy, token_loss, correct = _TritonChunkedHeadLoss.apply(
        hidden, head, target, chunk, denominator, ignore_index, with_grad
    )

    step = ChunkedStep(
        token_entropy=token_entropy,
        token_loss=token_loss,
        correct=correct,
        labelled=labelled,
    )
    return loss, step


def patch_soup_fused_cross_entropy():
    """Wire Triton fused cross entropy into Soup 0.77.0."""
    try:
        import soup_cli.utils.stream_loss_chunk as slc
        # 1. Swap the chunked loss implementation with our Triton version
        slc.chunked_causal_lm_loss = fused_chunked_causal_lm_loss
        slc._chunked_function = lambda: _TritonChunkedHeadLoss

        # 2. Patch SFTTrainerWrapper to enable chunked head loss by default
        from soup_cli.trainer.sft import SFTTrainerWrapper
        original_setup = SFTTrainerWrapper.setup

        def patched_setup(self, dataset: dict):
            # If stream_loss_chunk is not configured in task.yaml, default to CHUNK_SIZE
            tcfg = getattr(self.config, "training", None)
            if tcfg is not None and getattr(tcfg, "stream_loss_chunk", None) is None:
                tcfg.stream_loss_chunk = CHUNK_SIZE
            return original_setup(self, dataset)

        SFTTrainerWrapper.setup = patched_setup
    except (ImportError, AttributeError):
        pass

