# DECENTRATHON 6.0 — SOUP KERNEL CHALLENGE
## КОМАНДАЛЫҚ ТАПСЫРМАЛАР МЕН ИНТЕГРАЦИЯЛЫҚ НҰСҚАУЛЫҚ (LUNAR TEAM)

> **Басты мақсат:** Kaggle T4 GPU-де Soup 0.77.0 оқытуын 1.4x–1.6x жеделдету, жадты 14 ГБ-тан 3.5–4.0 ГБ-қа дейін түсіру және барлық 5 Must-Have шарттарынан 100 балл жинау.

---

### 1-рөл: БЕКБОЛАТ — Техникалық Лид & GPU Kernel Архитекторы
* **Бранч:** `feat/bekbolat-fused-loss`
* **Негізгі файлдар:**
  - `solution/fused_cross_entropy.py` (Fused Chunked Cross-Entropy + Loss + Metrics)
  - `solution/nf4_dequant.py` (Fast NF4 Triton Dequantization)
  - `solution/__init__.py` (Барлық кернелдерді `apply()` арқылы біріктіру)
* **Атқаратын рөлі:**
  - Жүйенің бүкіл математикалық дәлдігі мен жадтың 14 ГБ-тан 3.5 ГБ-қа түсуіне жауапты.
  - Нұрдәулет пен Дарханның кодтарын `apply()` арқылы бірыңғай жүйеге қабылдау (Code Review).
* **Тексеру командасы:**
  ```bash
  python -m pytest solution/test_kernels.py -k "test_fused_cross_entropy" -v
  ```

---

### 2-рөл: НҰРДӘУЛЕТ — Triton Element-Wise & Model Kernels Инженері
* **Бранч:** `feat/nurdaulet-elementwise`
* **Негізгі файлдар:**
  - `solution/fused_rmsnorm.py` (Fused RMSNorm Forward & Backward)
  - `solution/fused_rope.py` (Fused Rotary Position Embedding)
* **Нұрдәулетке берілетін нақты міндеттер мен мүмкіндіктер:**
  1. **RMSNorm кернелін жетілдіру (`solution/fused_rmsnorm.py`):**
     - Базалық кернелде `BLOCK_SIZE = 2048` тұр. T4 GPU үшін `BLOCK_SIZE = 1024` немесе `2048` қалай жылдам жұмыс істейтінін бенчмарк жасау.
     - `dw` (weight gradient) есептелуін Triton ішінде atomic_add немесе reduction арқылы тағы да жеделдету жолдарын сынау.
  2. **RoPE Embedding кернелін тексеру (`solution/fused_rope.py`):**
     - `fast_apply_rotary_pos_emb` қалай шақырылатынын, cos/sin көбейтудегі FP16 дәлдігін қадағалау.
  3. **Тестілеу:**
     - PyTorch эталонымен сәйкестігін тексеру:
     ```bash
     python -m pytest solution/test_kernels.py -k "test_fused_rmsnorm" -v
     ```

---

### 3-рөл: ДАРХАН — Batch Optimization, Harness & Тұрақтылық Инженері
* **Бранч:** `feat/darkhan-batch-opt`
* **Негізгі файлдар:**
  - `solution/dynamic_padding.py` (Dynamic In-Batch Pad-Token Removal)
  - `harness/check_rules.py` және `harness/run_task.py`
  - `task/task.yaml` және `task/prepare_data.py`
* **Дарханға берілетін нақты міндеттер мен мүмкіндіктер:**
  1. **Dynamic Pad-Token Truncation (`solution/dynamic_padding.py`):**
     - Мәліметтер 512 ұзындыққа дейін толықтырылған, бірақ нақты ұзындық ~220 токен.
     - `truncate_batch_to_max_len` функциясының әр батчты қалай қысқартып жатқанын логтардан бақылау.
     - `attention_mask.sum()` санының өзгермейтініне көз жеткізу (`num_tokens` тексеруінен өту үшін).
  2. **79 ішкі тесттің өтуін қадағалау:**
     ```bash
     python -m pytest tests/ -q
     ```
  3. **Harness & Автоматты ережелерді бақылау:**
     ```bash
     python harness/check_rules.py
     python harness/score.py runs/result.json
     ```
     - 10 тексерудің бәрі (c1 – c10) жасыл болып өтуін қадағалау.

---

### 4-рөл: ТОҒЖАН — Құжаттама, Kaggle Релиз және Питч
* **Негізгі файлдар:**
  - `README.md` (Жюриден 15 балл алудың кілті!)
  - `SUBMISSION.md`
  - `templates/kaggle_notebook.ipynb`
* **Тоғжанға берілетін нақты міндеттер:**
  1. `README.md` файлын 3 міндетті бөлім бойынша толтыру:
     - 1-бөлім: Не істелді (кернелдер сызбасы, формулалар).
     - 2-бөлім: Қалай іске қосу керек (Kaggle T4 қадамдары).
     - 3-бөлім: Нәтижелер мен шектеулер (Speedup, Memory салыстырмалы кестесі).
  2. `templates/kaggle_notebook.ipynb` дәптерін Kaggle-де ашып, `Run All` басып, Public релиз жасау.
  3. 3 минуттық демо-видео мен презентация дайындау.

---

### Командалық жұмыс схемасы (Git Workflow):

```
       [ main ] (Толық дайын эталондық шешім)
          │
          ├── feat/bekbolat-fused-loss  ──> Бекболат (Loss & NF4)
          ├── feat/nurdaulet-elementwise ──> Нұрдәулет (RMSNorm & RoPE)
          └── feat/darkhan-batch-opt    ──> Дархан (Dynamic Padding & Tests)
```

Әркім өз бранчында жұмыс істеп, тексеріп, `main` бранчына Pull Request жасайды.
Барлық кодтар бір-бірімен `solution/__init__.py` ішіндегі `apply()` арқылы мінсіз байланысады!
