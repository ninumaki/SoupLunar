# Submission: LunarTeam

Copy of `templates/SUBMISSION.md` filled in for our Decentrathon 6.0 submission (Technical Whitepaper & Empirical QA Edition).

## Links
- Repository, final commit: https://github.com/ninumaki/SoupLunar/commit/HEAD
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in our README):
  https://www.kaggle.com/code/ruekdp/notebookd4dc02b276
- Presentation: [PRESENTATION.md](./PRESENTATION.md) (`presentation.pptx`)
- Automated ML Research & QA Verification Script: [verify_research_qa.py](./verify_research_qa.py)

## Team (LunarTeam — 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Бекболат (Bekbolat) | Техникалық Лид, Капитан & GPU Kernel Архитекторы (Fused Chunked Cross-Entropy & NF4 Dequant) | @bekbolat | @bekbolat | @bekbolat_tg (Captain) |
| Тоғжан (Togzhan) | ML Research Engineer & Quality Assurance Lead (Convergence Math, VRAM Roofline, SQLite Audit & Whitepaper) | @ninumaki | @ruekdp | — |
| Нұрдәулет (Nurdaulet) | Triton Element-Wise Kernels & Monkey-Patching Инженері (RMSNorm, SwiGLU, RoPE, `apply()`) | @nurdaulet | @nurdaulet | — |
| Дархан (Darkhan) | Batch Optimization, Хакатон Harness & Тестілеу Инженері (Dynamic Padding, 79 PyTest, Rules) | @darkhan | @darkhan | — |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `fast_nf4_dequant_kernel` | `solution/nf4_dequant.py` | 4-bit NF4 register LUT dequantization (`bitsandbytes.functional.dequantize_4bit`) | ~4.2% |
| `fused_cross_entropy_fwd/bwd_kernel` | `solution/fused_cross_entropy.py` | Full $[2\times 2048\times 151936]$ logits via $C=128$ Online LogSumExp + Streaming Entropy | ~3.8% |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/fused_rmsnorm.py` | 57x `Qwen2RMSNorm` forward & backward (FP32 variance + SRAM $dx$, skips frozen $dW$) | ~3.1% |
| `fused_rope_fwd/bwd_kernel` | `solution/fused_rope.py` | `Qwen2RotaryEmbedding` in-place strided cos/sin rotation in SRAM | ~2.2% |
| *(Total own kernels)* | `solution/` | **All custom Triton kernels (Steps 3+)** | **13.3%** (check 4 requirement: $\ge 5\%$) |
| `int8_swiglu_fwd/bwd_kernel` | `solution/swiglu.py` | `Qwen2MLP` SwiGLU with group-128 symmetric INT8 compressed backward state | ~5.1% |
| `dynamic_padding` | `solution/dynamic_padding.py` | Dynamic in-batch trailing-pad truncation (-35.2% $O(S^2)$ attention FLOPs) | — |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- **Triton GPU Tiling & Compilation**: Tillet et al. (2019), "Triton: An Intermediate Language and Compiler for Tiled Neural Network Generators" (https://arxiv.org/abs/2104.04469).
- **RMSNorm formulation**: Zhang & Sennrich (2019), "Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467).
- **Online Softmax & Streaming Entropy**: Milakov & Gimelshein (2018), "Online normalizer calculation for softmax" (https://arxiv.org/abs/1805.02867).
- **SwiGLU activation concept**: Shazeer (2020), "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 79.38 s | 65.10 s | speedup 1.219x (time x0.820, 14.3s faster) |
| Peak GPU memory, MiB | 14,345 MiB | 4,231 MiB | base/ours 3.390x (үнемделген: 10,114 MiB / 10.1 GB!) |
| Eval loss (public rows) | 2.04970 | 2.04956 | -0.007% vs base (шегі <= 0.5%, сапа жақсарды) |
| Own kernel time share | 0.0% | 13.3% | PASS (талап: >= 5.0%, 2.6 есе асты) |

- [x] All 10 checks of `harness/check_rules.py` passed (10/10 PASS: ALL CHECKS PASSED: True)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits
- [x] 21 unit tests in `solution/test_kernels.py` passed (21 passed in 18.02s)
- [x] 79 Soup regression tests in `tests/` passed (79 passed in 0.38s)
- [x] `verify_research_qa.py` verified convergence (Delta L1 = 0.011%, LSE Entropy gap = 0.222%, SQLite 0 NaN/Inf)

## Presentation covers
- **Қай операцияны жеделдеттік және неге:** 44% уақыт алатын элементтік операциялар (57x RMSNorm, 28x SwiGLU, RoPE), $O(S^2)$ артық паддинг және $1.24\text{ ГБ}$ логиттер тензорын материализациялайтын Cross-Entropy шығынын жою.
- **Кернелдер мен VRAM математикасы қалай құрылған:** SRAM деңгейіндегі редукция, FP32 аккумуляторларындағы $dx$ градиенттері, $C=128$ чанк арқылы жадты $1.24\text{ ГБ}$-тан $78\text{ МБ}$-қа түсіру және Streaming Shannon Entropy формуласы.
- **Нәтижелер мен ресми көрсеткіштер:** Ресми Kaggle T4 өлшемінде Peak VRAM **14,345 MiB -> 4,231 MiB (3.390x үнемдеу, -10.1 ГБ)**, оқыту уақыты **79.38s -> 65.10s (1.219x speedup)**, `Eval Loss` **-0.007%**, кернелдер үлесі **13.3%**, барлық 10 тексеріс **100% PASS**.
- **Шектеулер және келесі қадамдар:** Tesla T4-те `bfloat16` жоқтығы, 64 қадамдық бенчмарктегі алғашқы 2 қадамның JIT компиляция шығыны.
