"""Dynamic In-Batch Padding & Right-Padded Causal SDPA Optimization for Soup SFT.

Designed for Kaggle T4 GPU (sm_75, 15GB VRAM, fp16) while strictly preserving:
  - Exact batch order and batch composition (no reordering or cross-row mixing).
  - Exact `num_tokens` per step (`attention_mask.sum()`).
  - Exact `rewind.jsonl` row IDs, token counts, and per-row losses (`RewindTrainer` safe).
  - Exact `entropy` and `mean_token_accuracy` metrics logged by TRL `SFTTrainer`.

Optimizations:
  1. CPU-side in-batch trailing pad truncation (`Trainer._prepare_inputs`) BEFORE H2D transfer
     and BEFORE `RewindTrainer.compute_loss` snapshots `inputs["labels"]` (zero GPU syncs,
     zero shape mismatch in `rewind.jsonl`).
  2. Tensor-Core sequence length alignment (`pad_to_multiple_of = 8` on CUDA `SFTTrainer`
     collator without mutating `TrainingArguments`), making `M = batch_size * seq_len` a
     multiple of 32 for `hmma.1688.f32` Tensor Core tiles and SDPA blocks.
  3. Pure right-padded 2D `attention_mask` elision inside `Qwen2Model.forward`:
     In Transformers 5.16 (`sdpa_attention.py`), passing a 2D `attention_mask` materializes a
     4D `[B, 1, S, S]` mask and disables `enable_gqa=True`, forcing 6x `repeat_kv` copies across
     all 28 layers and falling off fused causal SDPA on T4. For causal LM (`is_causal=True`),
     right-padded keys `k >= L_b` are already unreachable from every real query `q < L_b`, while
     keeping `inputs["attention_mask"]` intact in the batch dict preserves exact `entropy` and
     `num_tokens` accounting in `SFTTrainer.compute_loss`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

_SEQUENCE_KEYS = (
    "input_ids",
    "attention_mask",
    "labels",
    "position_ids",
    "completion_mask",
    "assistant_masks",
    "shift_labels",
    "loss_weights",
)

_PATCHED = False


def is_pure_right_padded_2d(attention_mask: torch.Tensor) -> bool:
    """Return True iff `attention_mask` is `[B, S]` where each row is `1..1` followed by `0..0`
    with at least one real token (`m[:, 0] == 1`). Must be called on CPU to avoid GPU syncs.
    """
    if not torch.is_tensor(attention_mask) or attention_mask.ndim != 2:
        return False
    bsz, seq_len = attention_mask.shape
    if bsz <= 0 or seq_len <= 0:
        return False
    m = attention_mask != 0
    if not bool(m[:, 0].all().item()):
        return False
    if seq_len == 1:
        return True
    return bool((m[:, 1:] <= m[:, :-1]).all().item())


def truncate_batch_to_max_len(inputs: Mapping[str, Any], multiple_of: int = 8) -> dict[str, Any]:
    """Trim trailing padding columns across all sequence tensors in `inputs`, aligned to `multiple_of`."""
    out = dict(inputs)
    attention_mask = out.get("attention_mask")
    input_ids = out.get("input_ids")
    if (
        attention_mask is None
        or input_ids is None
        or not torch.is_tensor(attention_mask)
        or not torch.is_tensor(input_ids)
        or attention_mask.ndim != 2
        or input_ids.ndim != 2
    ):
        return out

    orig_len = int(input_ids.shape[1])
    if orig_len <= 0 or attention_mask.shape[1] != orig_len:
        return out

    active_cols = (attention_mask != 0).any(dim=0)
    labels = out.get("labels")
    if labels is not None and torch.is_tensor(labels) and labels.ndim == 2 and labels.shape[1] == orig_len:
        active_cols = active_cols | (labels != -100).any(dim=0)

    col_idx = torch.arange(1, orig_len + 1, device=active_cols.device, dtype=torch.int64)
    last_idx = int((col_idx * active_cols.to(torch.int64)).max().item())
    if last_idx <= 0:
        return out

    if multiple_of > 1:
        aligned_len = min(orig_len, ((last_idx + multiple_of - 1) // multiple_of) * multiple_of)
    else:
        aligned_len = min(orig_len, last_idx)

    if aligned_len < orig_len:
        for key in _SEQUENCE_KEYS:
            val = out.get(key)
            if val is not None and torch.is_tensor(val) and val.ndim >= 2 and val.shape[1] == orig_len:
                out[key] = val[:, :aligned_len].contiguous()

    return out


def patch_dynamic_padding() -> None:
    """Install CPU pre-H2D pad truncation, 8-token Tensor Core alignment, and causal SDPA GQA unlock."""
    global _PATCHED
    if _PATCHED:
        return

    # 1. Patch Trainer._prepare_inputs (runs BEFORE H2D transfer & BEFORE RewindTrainer snapshots labels)
    try:
        from transformers import Trainer

        original_prepare_inputs = Trainer._prepare_inputs

        def patched_prepare_inputs(self, inputs: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
            right_padded = False
            if isinstance(inputs, Mapping):
                attn = inputs.get("attention_mask")
                if torch.is_tensor(attn) and not attn.is_cuda:
                    inputs = truncate_batch_to_max_len(inputs, multiple_of=8)
                    attn_after = inputs.get("attention_mask")
                    if torch.is_tensor(attn_after) and not attn_after.is_cuda:
                        right_padded = is_pure_right_padded_2d(attn_after)

            prepared = original_prepare_inputs(self, inputs)
            if right_padded and isinstance(prepared, Mapping):
                gpu_attn = prepared.get("attention_mask")
                if torch.is_tensor(gpu_attn):
                    gpu_attn._soup_right_padded = True
            return prepared

        Trainer._prepare_inputs = patched_prepare_inputs
    except (ImportError, AttributeError):
        pass

    # 2. Align CUDA SFTTrainer batches to multiples of 8 on the collator (leaves TrainingArguments untouched)
    try:
        from trl.trainer.sft_trainer import SFTTrainer

        original_sft_init = SFTTrainer.__init__

        def patched_sft_init(self, *args: Any, **kwargs: Any) -> None:
            original_sft_init(self, *args, **kwargs)
            device = getattr(getattr(self, "args", None), "device", None)
            collator = getattr(self, "data_collator", None)
            if (
                device is not None
                and getattr(device, "type", None) == "cuda"
                and collator is not None
                and getattr(collator, "pad_to_multiple_of", None) is None
                and not getattr(collator, "padding_free", False)
            ):
                collator.pad_to_multiple_of = 8

        SFTTrainer.__init__ = patched_sft_init
    except (ImportError, AttributeError):
        pass

    # 3. Elide redundant 4D causal mask in Qwen2Model.forward when batch is verified pure right-padded
    try:
        from transformers.models.qwen2 import modeling_qwen2

        original_qwen2_model_forward = modeling_qwen2.Qwen2Model.forward

        def patched_qwen2_model_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            **kwargs: Any,
        ):
            if (
                attention_mask is not None
                and getattr(attention_mask, "_soup_right_padded", False)
                and past_key_values is None
                and not use_cache
            ):
                attention_mask = None
            return original_qwen2_model_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )

        modeling_qwen2.Qwen2Model.forward = patched_qwen2_model_forward
    except (ImportError, AttributeError):
        pass

    _PATCHED = True
