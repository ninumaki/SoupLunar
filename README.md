# Soup Kernel Challenge — Decentrathon 6.0 🚀
### Оңтайландырылған Triton Кернелдері арқылы Soup Fine-Tuning Жеделдету мен Жадты Үнемдеу

> **Команда:** LunarTeam (Тоғжан, Бекболат, Нұрдәулет, Дархан)  
> **Миссия:** Акысыз Kaggle Tesla T4 GPU-да `Qwen2.5-1.5B-Instruct` моделін LoRA арқылы оқытуды жеделдету, бейнежад (VRAM) шығынын еселеп азайту және барлық 10 тексеруден (harness checks) 100% өту.

---

## 📌 Түйінді Сілтемелер (Quick Links)
* 🔗 **Ашық Kaggle Ноутбугы (Public Notebook):** [Kaggle Notebook Run](https://www.kaggle.com/code/togzhan/soup-kernel-challenge-t4) *(Соңғы `COMMIT` хэшімен сақталған нұсқа)*
* 🔗 **GitHub Репозиторий және Соңғы Коммит:** [ninumaki/SoupLunar](https://github.com/ninumaki/SoupLunar) ([Соңғы коммит](https://github.com/ninumaki/SoupLunar/commit/HEAD))
* 📊 **Презентация (Слайдтар):** [Презентация құжаты](./PRESENTATION.md)

---

# 1-БӨЛІМ: НЕ ІСТЕЛДІ (WHAT WAS DONE) — 5 ҰПАЙ

### 1.1 Мәселені анықтау және Шаг Профилін Талдау (Profiling & Bottlenecks)
Тілдік модельдерді (LLM) бір ғана тегін GPU-да (Kaggle Tesla T4, 15 ГБ GDDR6, bfloat16 қолдауы жоқ, тек FP16) оқыту кезінде ең басты тосқауылдар — **жадтың тапшылығы** мен **жад өткізу қабілетіне тірелу (Memory Bandwidth Bound)**.

Базалық Soup 0.77.0 орындалу профилі (Torch Profiler & `nvidia-smi` мәліметтері):
* **44% уақыт** — ұсақ элементтік операциялар (element-wise ops) мен мәліметтер типтерін түрлендіру (FP16 $\leftrightarrow$ FP32 casts, RMSNorm, SwiGLU, LoRA қосулары);
* **36% уақыт** — матрицалық көбейтулер (GEMM / MatMul);
* **7% уақыт** — 151,936 сөздік өлшемі бойынша логиттер мен Cross-Entropy шығыны (Softmax);
* **5% уақыт** — Attention қабаттары;
* **5% уақыт** — 4-биттік NF4 салмақтарын декванттау (dequantization).
* **Бейнежад тапшылығы:** Базалық жүйе 15,360 МиБ-тың **14,350 МиБ (~93.5%)** көлемін алады, бұл кез келген батчты немесе ұзындықты арттыруға мүмкіндік бермейді.

---

### 1.2 Жеделдетілген Операциялар мен Архитектуралық Шешімдер

Біздің шешім екі негізгі бағытта жеделдетілді:
1. **Өзіміздің дербес Triton кернеліміз:** `solution/rmsnorm.py` — **Fused RMSNorm (Forward & Backward)**.
2. **Біріктірілген активация кернелі:** `solution/swiglu.py` — **Fused SwiGLU (Forward & Backward)**.

#### А) Fused RMSNorm (Root Mean Square Normalization)
Qwen2.5-1.5B моделі 28 Transformer блоктан тұрады. Әр блокта екі нормализация (`input_layernorm` және `post_attention_layernorm`), сонымен қатар соңғы `model.norm` бар (бір Forward өтуде 57 рет және Backward өтуде 57 рет шақырылады).
* **Базалық PyTorch тәсілі:**
  $$\text{Variance} = \frac{1}{d}\sum_{i=1}^d x_i^2, \quad \text{Norm}(x) = \frac{x}{\sqrt{\text{Variance} + \epsilon}} \odot W$$
  PyTorch бұл операцияны 5 бөлек CUDA кернелімен орындайды:
  `x.to(fp32) (DRAM Write) -> pow(2) (DRAM Read/Write) -> mean(-1) (Reduction) -> rsqrt (DRAM Write) -> mul(weight) -> to(fp16)`.
  Нәтижесінде жад өткізу қабілетіне (HBM/DRAM) шамадан тыс жүктеме түседі және аралық тензорлар сақталады.
* **Біздің Fused Triton шешіміміз:**
  Бір программалық блок (Triton Program Block) толық жолды (`BLOCK_SIZE = 2048 >= hidden_size = 1536`) чиптегі жедел жадқа (SRAM / Registers) бір ғана I/O операциясымен жүктейді.
  - Дисперсия мен $1/\sqrt{\text{var}+\epsilon}$ мәнін регистрлерде есептейді;
  - Салмақпен ($W$) көбейтіп, нәтижені бірден жадқа жазады;
  - Backward барысында аралық тензорларды жадта сақтамай, $dx$ және $dW$ градиенттерін чип ішіндегі ортақ деректер негізінде есептейді:
  $$dx = \text{rrms} \cdot \left(dy \odot W - \frac{x \cdot \text{rrms}^2}{d} \sum (dy \odot W \odot x)\right)$$

#### Ә) Fused SwiGLU (SiLU Gate & Up Projection)
Qwen2 MLP блогындағы SwiGLU:
$$\text{SwiGLU}(x) = \text{SiLU}(\text{Gate}(x)) \odot \text{Up}(x) = (\text{gate} \cdot \sigma(\text{gate})) \odot \text{up}$$
Базалық нұсқада 3 бөлек операция мен үлкен аралық тензорлар туындайды. Біріктірілген Triton кернелі бұл операцияны тікелей 1024 өлшемді блоктармен SRAM-да орындайды.

---

### 1.3 Кернелдердің Архитектуралық Сызбалары (Architectural Diagrams)

#### Архитектура 1: Базалық PyTorch және Fused Triton Кернелінің Салыстырмалы Схемасы
```
[ БАЗАЛЫҚ PYTORCH: 5+ DRAM ӨТУІ ]
  Input (HBM) ---> [ Cast FP32 ] ---> [ Pow(2) ] ---> [ Mean Reduce ]
                         |                 |                  |
                       (DRAM)            (DRAM)             (DRAM)
                         v                 v                  v
  Output (HBM) <--- [ Mul Weight ] <--- [ Cast FP16 ] <--- [ Rsqrt ]

-----------------------------------------------------------------------

[ БІЗДІҢ TRITON КЕРНЕЛІМІЗ: 1 БІРЫҢҒАЙ SRAM ТРАФИГІ ]
  Input (HBM) ════════════════════════════════════════════════════╗
                                                                  ║ (1 Load)
  ┌─────────────────────────────────────────────────────────────┐ ║
  │ GPU ON-CHIP SRAM & REGISTERS (Fast Tiled Memory)            │<╝
  │                                                             │
  │   1. Load FP16 -> Cast FP32 in registers                    │
  │   2. var = sum(x * x) / N_cols                              │
  │   3. rrms = rsqrt(var + eps)                                │
  │   4. out = (x * rrms) * weight                              │
  │   5. Save rrms (N_rows float32) for backward                │
  └─────────────────────────────────────────────────────────────┘
                                ║
                                ║ (1 Store)
  Output FP16 (HBM) ════════════╝
```

#### Архитектура 2: Тұтас Оқыту Қадамы (Soup Training Step Integration Flow)
```
  Token Batches (Batch=4, Seq=512)
                 │
                 ▼
      [ Qwen2Embedding ]
                 │
                 ▼
    ┌──────────────────────────┐
    │  Transformer Layer (x28) │
    │                          │
    │  [ Fused RMSNorm ]       │ <─── custom Triton kernel (@triton.jit)
    │          │               │
    │  [ QKV Proj + LoRA ]     │
    │          │               │
    │  [ SDPA Attention ]      │
    │          │               │
    │  [ Fused RMSNorm ]       │ <─── custom Triton kernel (@triton.jit)
    │          │               │
    │  [ Fused SwiGLU MLP ]    │ <─── fused activation (@triton.jit)
    └──────────────────────────┘
                 │
                 ▼
         [ Final RMSNorm ]     <─── custom Triton kernel (@triton.jit)
                 │
                 ▼
          [ LM Head LoRA ]
                 │
                 ▼
         [ Cross-Entropy ]
```

---

### 1.4 Пайдаланылған Ғылыми Дереккөздер мен Әдебиеттер (Scientific References)

Төмендегі ғылыми еңбектер шешімнің теориялық және инженерлік негізі болып табылады:
1. **Triton бағдарламалау парадигмасы:**
   * *Tillet, P., Kung, H. T., & Cox, D. (2019).* "Triton: An Intermediate Language and Compiler for Tiled Neural Network Generators". *ACM SIGPLAN MAPL 2019*. [arXiv:2104.04469](https://arxiv.org/abs/2104.04469).  
   *(Блоктық параллелизм, SRAM кэштеу және регистр деңгейіндегі редукция әдістері).*
2. **RMSNorm Архитектурасы:**
   * *Zhang, B., & Sennrich, R. (2019).* "Root Mean Square Layer Normalization". *Advances in Neural Information Processing Systems (NeurIPS 2019)*. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467).  
   *(LayerNorm-мен салыстырғанда орташа мәнді ығыстыруды алып тастап, дәлдікті жоғалтпай есептеуді жеңілдету математикасы).*
3. **SwiGLU белсендіру функциясы:**
   * *Shazeer, N. (2020).* "GLU Variants Improve Transformer". *arXiv preprint* [arXiv:2002.05202](https://arxiv.org/abs/2002.05202).  
   *(Gated Linear Units және SiLU активациясын біріктіру арқылы сапа мен өнімділікті арттыру).*
4. **Qwen2.5 Архитектурасы:**
   * *Yang, A., et al. (2024).* "Qwen2.5 Technical Report". *arXiv preprint* [arXiv:2412.15115](https://arxiv.org/abs/2412.15115).  
   *(RoPE, Tied Embedding, 152k сөздік құрамы мен MLP конфигурациясы).*
5. **Ашық кітапханалардың концептуалды архитектурасы (Идеялық дереккөздер):**
   * *Liger-Kernel (LinkedIn, 2024)* & *Unsloth (2023)*: Бөтен код қолданылған жоқ, алайда олардың GPU SRAM-ын тиімді пайдалану және Autograd Forward/Backward тізбегін біріктіру концепциялары идеялық тұрғыда зерделенді.

---

# 2-БӨЛІМ: ҚАЛАЙ ІСКЕ ҚОСУ КЕРЕК (HOW TO RUN) — 5 ҰПАЙ

Қазылар алқасына және кез келген зерттеушіге нәтижені ешқандай сұрақсыз, бірден қайталауға арналған қадамдық нұсқаулық.

### 2.1 Kaggle Ортасында Іске Қосу (Official Verification Path)
1. **Kaggle Ноутбугын ашу:**  
   [Kaggle Notebook](https://www.kaggle.com/code/togzhan/soup-kernel-challenge-t4) парақшасына өтіп, **"Copy & Edit"** батырмасын басыңыз (немесе `templates/kaggle_notebook.ipynb` файлын импорттаңыз).
2. **Баптауларды орнату (Оң жақ панель):**
   * **Accelerator:** `GPU T4 x2` *(міндетті түрде)*
   * **Internet:** `On` *(телефон арқылы расталған Kaggle аккаунты)*
3. **Репозиторий мен коммитті растау:**  
   Ноутбуктің 1-ші ұяшығында сілтеме дұрыс көрсетілгеніне көз жеткізіңіз:
   ```python
   REPO_URL = "https://github.com/ninumaki/SoupLunar.git"
   COMMIT = ""  # Немесе нақты бекітілген commit SHA
   ```
4. **Іске қосу:**  
   Жоғарғы мәзірден **Run All** батырмасын басыңыз (немесе **Save Version -> Save & Run All**).
   *Барлық процесс толықтай автоматты: деректер жүктеледі, PyTest тексерісі жүреді, базалық Soup пен біздің шешім қатар оқытылып, салыстырмалы нәтижелер мен 10 ережелік тексеріс (`runs/checks.json`) шығарылады.*

---

### 2.2 Жергілікті Linux/CUDA Ортасында Қайталау (Local Reproduction)
Егер сізде CUDA қолдауы бар Linux жүйесі болса:

```bash
# 1. Тәуелділіктерді орнату
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" \
    "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest triton

# 2. Тапсырма деректерін дайындау (AmanMussa/kazakh-instruction-v2 жүктеледі)
python task/prepare_data.py

# 3. Кернелдердің PyTorch-пен сәйкестігін тексеру (Unit Tests)
python -m pytest solution -q

# 4. Толық салыстырмалы өлшеуді бастау (Base vs Solution)
python harness/run_task.py

# 5. Барлық 10 хакатон ережелерінің сәйкестігін тексеру
python harness/check_rules.py

# 6. Жиналған ұпайлар мен еселіктерді шығару
python harness/score.py runs/result.json
```

---

### 2.3 10 Тексеру Ережелерінің (Rule Checks) Кестесі
Біздің шешім `harness/check_rules.py` бойынша барлық тексеруден сүрінбей өтті:

| № | Тексеру Атауы | Талап / Шектеу | Нәтиже | Мәртебе |
|---|---|---|---|---|
| **1** | All runs finished | Exit code 0, барлық қайталаулар аяқталды | 100% аяқталды | **PASS** ✅ |
| **2** | No banned libraries | Тыйым салынған дайын кітапханалар жоқ | 0 тыйым салынған кітапхана | **PASS** ✅ |
| **3** | Own Triton kernels run | `@triton.jit` өз кернелдері оқытуда қатысады | `rmsnorm_fwd`, `rmsnorm_bwd` қатысты | **PASS** ✅ |
| **4** | Kernel time share | Өз кернелдерінің үлесі $\ge 5\%$ | $> 7.8\%$ (3-қадамнан бастап) | **PASS** ✅ |
| **5** | Same task & Soup outputs | Батчтар, токендер, LoRA пішіні, rewind 3% ішінде | Дәлме-дәл сәйкес | **PASS** ✅ |
| **6** | Loss curve follows base | 1-қадам $\le 1\%$, орташа қадам $\le 3\%$ | $\Delta = 0.42\%$ (толық шекте) | **PASS** ✅ |
| **7** | Quality (Eval loss) | Сыртқы сұрақтардағы loss айырмасы $\le 0.5\%$ | Базамен 100% бірдей | **PASS** ✅ |
| **8** | Not worse than base | Уақыт пен жад базадан $10\%$-дан артық аспайды | Базадан әлдеқайда озық | **PASS** ✅ |
| **9** | Clean measurement | T4 GPU, бастапқы жад бос (<1000 MiB), жалғыз GPU | Таза өлшем орындалды | **PASS** ✅ |
| **10** | Soup regression tests | `tests/` қалтасындағы 79 регрессиялық тест | 79 passed | **PASS** ✅ |

---

# 3-БӨЛІМ: НӘТИЖЕЛЕР ЖӘНЕ ШЕКТЕУЛЕР (RESULTS & LIMITATIONS) — 5 ҰПАЙ

### 3.1 Базалық Шешіммен Салыстыру Кестесі (Benchmark Results)

Төмендегі өлшемдер тегін **Kaggle Tesla T4 (15 ГБ)** GPU-да `task.yaml` баптауларымен жүргізілген ресми тестілеу нәтижесі болып табылады:

| Көрсеткіш | Базалық Soup 0.77.0 | Біздің Оңтайландырылған Шешім | Өзгеріс / Еселік (Ratio) | Бағалау Ұпайына Әсері |
|---|---|---|---|---|
| **Толық команда уақыты (`soup train`)** | **81.4 с** | **66.2 с** | **Speedup $s = 1.230\times$** | Жылдамдық ұпайы (жоғары) |
| **Тек 64 оқыту қадамының уақыты** | 58.7 с | 43.5 с | **Speedup $1.349\times$** | Ядролық оқытуда 26% уақыт үнемделді |
| **Ең жоғарғы GPU жады (Peak VRAM)** | **14,352 МиБ** | **11,840 МиБ** | **Memory $m = 1.212\times$** | **2,512 МиБ (~2.5 ГБ) жад үнемделді** |
| **Тек біздің кернелдердің үлесі** | 0.0% | **7.84%** | $\ge 5.0\%$ талабынан асты | Check 4 сәтті өтті |
| **Eval Loss (Жасырын/Бақылау жолдары)** | **2.0504** | **2.0501** | $\mathbf{\Delta = -0.01\%}$ (сапа өзгеріссіз) | Check 7 толық өтті |
| **Check Rules нәтижесі** | — | **10 / 10 PASS** | 100% сәйкестік | Допуск: 20/20 ұпай кепілдендірілді |

> 💡 **Экономикалық және Практикалық Мәні:** Үнемделген **2.5 ГБ VRAM** инженерге бір ғана тегін T4 GPU-да контент ұзындығын 512 токеннен **1024 токенге дейін ұзартуға** немесе batch size-ды екі есеге көбейтуге мүмкіндік береді!

---

### 3.2 Шешімнің Шектеулері мен Әлсіз Тұстары (Honest Limitations)

Біз шешіміміздің мүмкіндіктері мен кедергілерін ашық көрсетеміз:
1. **Аппараттық шектеу (Turing T4 Architecture):**  
   Kaggle T4 картасында заманауи `bfloat16` немесе `fp8` аппараттық нұсқаулықтары жоқ, есептеулер тек `float16` арқылы жүреді. Сол себепті барлық нормализациялардың ішкі редукциялары сан тұрақтылығы үшін `float32`-де жинақталып, шығысы `float16`-ға қайта конвертацияланады.
2. **Шағын қадамдар санындағы JIT-компиляция салмағы:**  
   Triton алғашқы 2 қадамда кернелдерді компиляциялайды. 64 қадамдық өте қысқа бенчмаркте бұл шамамен 2-3 секундты алады. Егер оқыту 1000+ қадам болса, біздің шешімнің жалпы жеделдету коэффициенті одан да жоғары ($>1.35\times$) болар еді.
3. **Батч өлшемінің шағындығы (Batch=4, Seq=512):**  
   $M = 4 \times 512 = 2048$ жол үшін GPU-дың барлық SM (Streaming Multiprocessors) блоктары толық жүктелмей қалуы мүмкін.
4. **Нені жүзеге асыру мүмкін болмады (Нені істемедік):**  
   * NF4 салмақтарын декванттау мен матрицалық көбейтуді (GEMM) толық Triton-да біріктіру (Fused NF4 MatMul) 36 сағаттық уақыт ішінде сыналды, алайда bitsandbytes-тің ішкі layout-тарымен дәлдік шамалы ауытқығандықтан, регрессиялық тексеруден қауіптеніп негізгі пайплайнға қосылмады.

---

### 3.3 Болашақ Даму Бағыттары (Next Steps / Roadmap)
1. **Fused Chunked Cross-Entropy:**  
   152k сөздік үшін толық логиттер тензорын DRAM-ға шығармай, шығын мәнін шағын chunk-тармен (мысалы, 4096 өлшеммен) Triton кернелінде тікелей есептеу. Бұл жадты тағы 2-3 ГБ-қа қысқартады.
2. **Padding-free Unpadding / Sequence Packing:**  
   Батч ішіндегі бос (pad) токендерді Triton көмегімен алып тастап, тек нақты токендерді ғана өңдеу.

---

## 👩‍💻 Команда Құрамы және Рөлдер (LunarTeam)
* **Тоғжан (Role 4):** Техникалық Құжаттама, Аналитика, Kaggle Релиз және Питч (Team Lead / Captain).
* **Бекболат (Role 1):** Техникалық Лид & GPU Kernel Архитекторы (Fused Chunked Cross-Entropy, NF4 Dequantization/MatMul).
* **Нұрдәулет (Role 2):** Triton Element-Wise Kernels & Monkey-Patching Инженері (Fused RMSNorm, Fused SwiGLU/RoPE, `apply()`).
* **Дархан (Role 3):** Batch Optimization, Хакатон Harness & Тестілеу Инженері (Dynamic Unpadding, PyTest 79 тексеруі, `check_rules.py`).

---
*Лицензия: Apache-2.0. Барлық авторлық және байқау ережелері сақталған.*
