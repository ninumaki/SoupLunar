"""QLoRA Mixed-Precision & PEFT Adapter Forward Optimization for Tesla T4 (`sm_75`).

Addresses two major memory and throughput bottlenecks in standard HuggingFace + PEFT 4-bit training:

1. Selective `prepare_model_for_kbit_training` Norm-Only FP32 Upcast:
   Upstream `peft.prepare_model_for_kbit_training` casts EVERY non-`Params4bit` parameter in the
   model to `torch.float32` — including `Qwen2.5-1.5B`'s frozen `embed_tokens` / tied `lm_head`
   (`[151936, 1536]`, 233.4M params) and frozen `Linear4bit` QKV biases:
     - Permanently inflates `embed_tokens.weight` from 445.1 MiB (`fp16`) to 890.2 MiB (`fp32`).
     - Forces a 445.1 MiB `fp32 -> fp16` weight copy of `lm_head.weight` on every training step.
     - Promotes the entire 28-layer residual stream (`residual + sublayer_out`) to `fp32`, doubling
       all 56 `RMSNorm` saved activations (`672 MiB -> 336 MiB`) and forcing all 56 `o_proj` and
       `down_proj` backward GEMMs off 65-TFLOPS FP16 Tensor Cores onto 8.1-TFLOPS FP32 SIMT cores!
     - Forces 84 `bias.to(fp16)` kernel launches per step inside `bnb.nn.Linear4bit.forward`.
   Keeping frozen `embed_tokens` / `lm_head` and frozen `Linear4bit` biases in `fp16` while
   upcasting only normalization layer weights (`RMSNorm` / `LayerNorm`) to `fp32` saves ~1.23 GiB
   of peak VRAM and unlocks FP16 Tensor Cores across all backward projections, while leaving all
   trainable LoRA weights in `torch.float32` (`F32`).

2. Zero-Clone Fused Scaling in `peft.tuners.lora.bnb.Linear4bit.forward`:
   Upstream `lora.Linear4bit.forward` unconditionally clones `base_layer(x)` (`result.clone()`) and
   executes separate `output * scaling` + `result + output` kernels on all 56 LoRA projections per
   step. Because `MatMul4Bit.forward` already returns a fresh contiguous tensor and does not save
   `output` for backward, replacing `clone() + mul + add` with a single `torch.add(result, lora_out,
   alpha=scaling)` eliminates 112 CUDA kernel launches and `[B, S, D]` allocations per step while
   preserving exact `dropout(x)`, RNG state, and AMP autocast semantics.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import torch

_PATCHED = False


def _collect_norm_param_ids(model: torch.nn.Module) -> set[int]:
    """Return `id(param)` for parameters belonging directly to normalization submodules."""
    norm_ids: set[int] = set()
    norm_base = getattr(torch.nn.modules.normalization, "_NormBase", ())
    for mod in model.modules():
        cls_name = mod.__class__.__name__.lower()
        if "norm" in cls_name or (norm_base and isinstance(mod, norm_base)):
            for param in mod.parameters(recurse=False):
                norm_ids.add(id(param))
    return norm_ids


def _selective_prepare_4bit_model(
    model: torch.nn.Module,
    use_gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    auto_clear_cache: bool = True,
) -> torch.nn.Module:
    """Freeze base weights and upcast ONLY normalization parameters to FP32 on 4-bit models."""
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    for _name, param in model.named_parameters():
        param.requires_grad = False

    norm_param_ids = _collect_norm_param_ids(model)
    casted_any = False
    for param in model.parameters():
        if (
            id(param) in norm_param_ids
            and param.dtype in (torch.float16, torch.bfloat16)
            and param.__class__.__name__ != "Params4bit"
        ):
            param.data = param.data.to(torch.float32)
            casted_any = True

    if auto_clear_cache and casted_any and torch.cuda.is_available():
        torch.cuda.empty_cache()

    if use_gradient_checkpointing:
        if "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(_module, _input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )
        if not supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
            warnings.warn(
                "gradient_checkpointing_kwargs is not supported in this version of transformers.",
                FutureWarning,
            )
        gc_enable_kwargs = (
            {} if not supports_gc_kwargs else {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
        )
        model.gradient_checkpointing_enable(**gc_enable_kwargs)

    return model


def patch_kbit_mixed_precision() -> None:
    """Patch PEFT k-bit model preparation and 4-bit LoRA forward pass idempotently."""
    global _PATCHED
    if _PATCHED:
        return

    # 1. Patch peft.prepare_model_for_kbit_training across all PEFT re-export namespaces
    try:
        import peft
        import peft.utils as peft_utils
        import peft.utils.other as peft_other

        original_prepare = peft_other.prepare_model_for_kbit_training

        def patched_prepare_model_for_kbit_training(
            model: Any,
            use_gradient_checkpointing: bool = True,
            gradient_checkpointing_kwargs: dict[str, Any] | None = None,
            auto_clear_cache: bool = True,
        ) -> Any:
            is_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
            quant_method = getattr(model, "quantization_method", None)
            is_other_quant = quant_method in ("gptq", "aqlm", "eetq", "torchao", "hqq") or bool(
                getattr(model, "hqq_quantized", False)
            )
            if is_4bit and not is_other_quant and isinstance(model, torch.nn.Module):
                return _selective_prepare_4bit_model(
                    model,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
                    auto_clear_cache=auto_clear_cache,
                )
            return original_prepare(
                model,
                use_gradient_checkpointing=use_gradient_checkpointing,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
                auto_clear_cache=auto_clear_cache,
            )

        peft_other.prepare_model_for_kbit_training = patched_prepare_model_for_kbit_training
        peft_utils.prepare_model_for_kbit_training = patched_prepare_model_for_kbit_training
        peft.prepare_model_for_kbit_training = patched_prepare_model_for_kbit_training
    except (ImportError, AttributeError):
        pass

    # 2. Patch peft.tuners.lora.bnb.Linear4bit.forward to eliminate defensive clone + extra mul
    try:
        import peft.tuners.lora.bnb as peft_lora_bnb
        from peft.tuners.lora.layer import VARIANT_KWARG_KEYS

        if hasattr(peft_lora_bnb, "Linear4bit"):
            original_lora_4bit_forward = peft_lora_bnb.Linear4bit.forward

            def fast_lora_4bit_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
                if (
                    self.disable_adapters
                    or self.merged
                    or "adapter_names" in kwargs
                    or any(k in kwargs and kwargs[k] is not None for k in VARIANT_KWARG_KEYS)
                    or any(a in self.lora_variant for a in self.active_adapters)
                ):
                    return original_lora_4bit_forward(self, x, *args, **kwargs)

                self._check_forward_args(x, *args, **kwargs)
                for k in VARIANT_KWARG_KEYS:
                    kwargs.pop(k, None)

                result = self.base_layer(x, *args, **kwargs)
                requires_conversion = not torch.is_autocast_enabled()
                expected_dtype = result.dtype

                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A:
                        continue
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]

                    x_in = self._cast_input_dtype(x, lora_A.weight.dtype) if requires_conversion else x
                    lora_out = lora_B(lora_A(dropout(x_in)))
                    if requires_conversion:
                        lora_out = lora_out.to(expected_dtype)
                    result = torch.add(result, lora_out, alpha=scaling)

                return result

            peft_lora_bnb.Linear4bit.forward = fast_lora_4bit_forward
    except (ImportError, AttributeError):
        pass

    _PATCHED = True
