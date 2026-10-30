# Submission: LunarTeam

Copy of `templates/SUBMISSION.md` filled in for our Decentrathon 6.0 submission.

## Links
- Repository, final commit: https://github.com/ninumaki/SoupLunar/commit/HEAD
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in our README):
  https://www.kaggle.com/code/togzhan/soup-kernel-challenge-t4?scriptVersionId=1
- Presentation: [PRESENTATION.md](./PRESENTATION.md) (`presentation.pptx`)

## Team (LunarTeam — 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Тоғжан (Togzhan) | Техникалық Құжаттама, Аналитика, Kaggle Релиз және Питч | @ninumaki | @togzhan | @togzhan_tg (Captain) |
| Бекболат (Bekbolat) | Техникалық Лид & GPU Kernel Архитекторы (Fused Cross-Entropy & NF4) | @bekbolat | @bekbolat | — |
| Нұрдәулет (Nurdaulet) | Triton Element-Wise Kernels & Monkey-Patching Инженері (RMSNorm, SwiGLU, apply) | @nurdaulet | @nurdaulet | — |
| Дархан (Darkhan) | Batch Optimization, Хакатон Harness & Тестілеу Инженері (Unpadding, Tests) | @darkhan | @darkhan | — |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `rmsnorm_fwd_kernel` | `solution/rmsnorm.py` | `Qwen2RMSNorm.forward` (FP16/FP32 fused normalization & weight scaling) | ~4.2% |
| `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | `Qwen2RMSNorm.backward` (Fused gradient computation $dx$ & $dW$ in SRAM) | ~3.6% |
| *(Бірге өз кернелдеріміздің үлесі)* | `solution/rmsnorm.py` | **Барлық Qwen2RMSNorm қабаттары (Forward + Backward)** | **7.84%** (талап: $\ge 5\%$) |
| `example_swiglu_fwd_kernel` | `solution/swiglu.py` | `Qwen2MLP` SwiGLU activation forward (starter kit example) | ~2.6% |
| `example_swiglu_bwd_kernel` | `solution/swiglu.py` | `Qwen2MLP` SwiGLU activation backward (starter kit example) | ~2.5% |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- **Triton GPU Tiling & Compilation**: Tillet et al. (2019), "Triton: An Intermediate Language and Compiler for Tiled Neural Network Generators" (https://arxiv.org/abs/2104.04469).
- **RMSNorm formulation**: Zhang & Sennrich (2019), "Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467).
- **SwiGLU activation concept**: Shazeer (2020), "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202).
- **Autograd SRAM reuse concept**: Liger-Kernel & Unsloth architecture whitepapers (ideation only; all code in `solution/rmsnorm.py` written from scratch in Triton).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 81.4 s | 66.2 s | speedup 1.230x (steps: 1.349x) |
| Peak GPU memory, MiB | 14,352 MiB | 11,840 MiB | base/ours 1.212x (үнемделген: 2,512 MiB) |
| Eval loss (public rows) | 2.0504 | 2.0501 | -0.01% vs base (толық сақталған) |

- [x] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
- **Қай операцияны жеделдеттік және неге:** 44% уақыт алатын элементтік операциялар (RMSNorm) мен 14.3 ГБ жад алатын аралық тензорларды қысқарту үшін.
- **Кернелдер қалай құрылған:** Блоктық параллелизм ($BLOCK\_SIZE=2048$), SRAM деңгейіндегі редукция, Fast FP32 есептеу және FP16 I/O, Autograd backward градиенттерін чип ішінде есептеу.
- **Нәтижелер:** 1.23x толық жылдамдық, 2.5 ГБ VRAM үнемдеу (14.3 GB -> 11.8 GB), loss өзгеріссіз.
- **Шектеулер және келесі қадамдар:** T4-те bfloat16 жоқтығы, қысқа 64 қадамдағы JIT компиляция overhead-і, болашақта Fused Cross-Entropy және Sequence Packing енгізу.

## Limitations
- **T4 Architecture:** bfloat16 қолдауы жоқ, сол себепті есептеулер FP32 accumulator + FP16 I/O арқылы шектелген.
- **Short run overhead:** 64 қадамдық бенчмаркте алғашқы 2 қадам Triton компиляциясына кеткендіктен уақыттың 2-3 секунды компиляцияға жұмсалады; ұзақ оқытуда нәтиже одан да жоғары.
- **Next steps:** 152k сөздікке арналған Chunked Cross-Entropy Loss біріктіру және NF4 dequantization GEMM-ді жазу.
