#!/usr/bin/env python3
"""
Generate a professional, AI-grader-ready 6-slide PowerPoint presentation (.pptx)
for Decentrathon 6.0 — Soup Kernel Challenge (LunarTeam).

Adheres strictly to the 4 mandatory questions, official Kaggle T4 measured results,
VRAM roofline math, and the 10/10 PASS rule checks.
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_FILE = Path(__file__).resolve().parent / "presentation.pptx"

# Modern Tech Palette
DARK_NAVY  = RGBColor(15, 23, 42)     # #0F172A (Slide Background)
CARD_BG    = RGBColor(30, 41, 59)     # #1E293B (Card Panels)
ACCENT_CYAN= RGBColor(6, 182, 212)    # #06B6D4 (Key Highlights)
ACCENT_EMER= RGBColor(16, 185, 129)   # #10B981 (Success Green)
TEXT_WHITE = RGBColor(248, 250, 252)  # #F8FAFC (Headings)
TEXT_MUTED = RGBColor(148, 163, 184)  # #94A3B8 (Subtitles)
TEXT_BODY  = RGBColor(226, 232, 240)  # #E2E8F0 (Body text)
BORDER_COL = RGBColor(51, 65, 85)     # #334155 (Subtle borders)


def set_bg(slide):
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    background.fill.solid()
    background.fill.fore_color.rgb = DARK_NAVY
    background.line.color.rgb = DARK_NAVY
    return background


def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=BORDER_COL):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1.2)
    return card


def add_header(slide, title_text, category_text="DECENTRATHON 6.0 — SOUP KERNEL CHALLENGE"):
    tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
    tf = tx_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.name = "Arial"
    p_cat.font.size = Pt(11)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ACCENT_CYAN

    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = "Arial"
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE
    p_title.space_before = Pt(4)


# ==============================================================================
# SLIDE 1: Title Slide
# ==============================================================================
def build_slide_1(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)

    # Main Center Card
    add_card(slide, 0.8, 0.8, 11.733, 5.9, bg_color=CARD_BG, border_color=BORDER_COL)

    tx_box = slide.shapes.add_textbox(Inches(1.2), Inches(1.1), Inches(10.9), Inches(5.3))
    tf = tx_box.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "DECENTRATHON 6.0 — SOUP KERNEL CHALLENGE"
    p0.font.name = "Arial"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_CYAN

    p1 = tf.add_paragraph()
    p1.text = "Qwen2.5 Fine-Tuning: Triton Kernels & VRAM Optimization"
    p1.font.name = "Arial"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_WHITE
    p1.space_before = Pt(8)

    p2 = tf.add_paragraph()
    p2.text = "Kaggle Tesla T4 (15GB): VRAM 3.39x үнемдеу (14.35 GB -> 4.23 GB), 1.219x Speedup және 10/10 PASS"
    p2.font.name = "Arial"
    p2.font.size = Pt(14)
    p2.font.color.rgb = ACCENT_EMER
    p2.space_before = Pt(6)

    p3 = tf.add_paragraph()
    p3.text = "\nКоманда: LunarTeam"
    p3.font.name = "Arial"
    p3.font.size = Pt(15)
    p3.font.bold = True
    p3.font.color.rgb = TEXT_WHITE

    p4 = tf.add_paragraph()
    p4.text = (
        "• Бекболат (Role 1): Техникалық Лид, Капитан & GPU Kernel Архитекторы (Chunked CE & NF4 Dequant)\n"
        "• Тоғжан (Role 4): ML Research Engineer & QA Lead (Convergence Math, VRAM Roofline & Pitch)\n"
        "• Нұрдәулет (Role 2): Triton Element-Wise & Model Kernels Инженері (RMSNorm, RoPE, SwiGLU)\n"
        "• Дархан (Role 3): Batch Optimization, Harness & Тұрақтылық Инженері (Dynamic Padding, 79 Tests)"
    )
    p4.font.name = "Arial"
    p4.font.size = Pt(11.5)
    p4.font.color.rgb = TEXT_BODY
    p4.space_before = Pt(8)

    p5 = tf.add_paragraph()
    p5.text = (
        "\n🔗 GitHub: github.com/ninumaki/SoupLunar   |   "
        "🔗 Kaggle Notebook: kaggle.com/code/ruekdp/notebookd4dc02b276"
    )
    p5.font.name = "Arial"
    p5.font.size = Pt(11)
    p5.font.bold = True
    p5.font.color.rgb = ACCENT_CYAN
    p5.space_before = Pt(10)


# ==============================================================================
# SLIDE 2: Question 1 — What operations & Why?
# ==============================================================================
def build_slide_2(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_header(slide, "1-Сұрақ: Қай операцияларды жеделдеттік және неге?", "ПРОФАЙЛИНГ ЖӘНЕ КЕДЕРГІЛЕР (BOTTLENECKS)")

    # Left Card: Profiling & Bottlenecks
    add_card(slide, 0.8, 1.6, 5.7, 5.2)
    b1 = slide.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf1 = b1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "Базалық Орындалу Профилі (Soup 0.77.0):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN

    items_left = [
        ("44% уақыт: Ұсақ элементтік операциялар", "FP16 <-> FP32 casts, 57x RMSNorm, 28x SwiGLU, RoPE, LoRA қосулары."),
        ("36% уақыт: GEMM / MatMul", "Сызықтық қабаттардың матрицалық көбейтулері."),
        ("7% уақыт: Cross-Entropy Loss & Softmax", "151,936 сөздік өлшемі бойынша логиттер мен градиенттер (VRAM шыңы!)."),
        ("5% уақыт: NF4 салмақтарын декванттау", "4-биттік салмақтарды әр қадамда absmax блоктарымен ашу."),
        ("5% уақыт: Attention блогы", "Бос trailing pad токендерімен O(S^2) есептелуі."),
        ("VRAM Tapshylygy (OOM қаупі):", "Базалық жүйе қолжетімді 15,360 МиБ-тың 14,345 МиБ-ын (~93.4%) алады!")
    ]
    for title, desc in items_left:
        pt = tf1.add_paragraph()
        pt.text = f"• {title}: "
        pt.font.bold = True
        pt.font.size = Pt(11)
        pt.font.color.rgb = TEXT_WHITE
        pt.space_before = Pt(5)
        pd = tf1.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10)
        pd.font.color.rgb = TEXT_MUTED

    # Right Card: The Core Reason & Strategy
    add_card(slide, 6.8, 1.6, 5.7, 5.2)
    b2 = slide.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf2 = b2.text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "Неге осы операцияларды таңдадық?"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_EMER

    reasons = [
        ("Memory-Bandwidth Bound мәселесі", "Tesla T4 картасының жад өткізу қабілеті небәрі 320 ГБ/с. Элементтік операциялардың арифметикалық қарқындылығы өте төмен (I << 10), уақыт HBM/DRAM-ға дерек тасуға кетеді."),
        ("SRAM On-Chip Fusion стратегиясы", "PyTorch бөлек CUDA кернелдерінің орнына, бірнеше амалды GPU-дың шапшаң ішкі жадында (SRAM/Registers) бір ғана I/O өтуімен орындау."),
        ("1.24 GB Логиттер Тензорын Жою", "152k сөздік логиттерін жадта сақтамай, Online Softmax (LogSumExp) арқылы кернел ішінде есептеу VRAM шыңын түбегейлі шешеді."),
        ("Pad-Token Truncation арқылы -35% FLOPs", "Датасет ұзындығы ~220 токен. Батч ішіндегі артық паддингті кесу арқылы Attention көлемін 35%-ға қысқарту.")
    ]
    for title, desc in reasons:
        pt = tf2.add_paragraph()
        pt.text = f"✔ {title}:"
        pt.font.bold = True
        pt.font.size = Pt(11)
        pt.font.color.rgb = ACCENT_CYAN
        pt.space_before = Pt(8)
        pd = tf2.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = TEXT_BODY


# ==============================================================================
# SLIDE 3: Question 2 — How are kernels built?
# ==============================================================================
def build_slide_3(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_header(slide, "2-Сұрақ: Кернелдер қалай құрылған?", "АРХИТЕКТУРА ЖӘНЕ ТРИТОН КЕРНЕЛДЕРІ (@TRITON.JIT)")

    cards = [
        ("Fused Chunked Cross-Entropy", "solution/fused_cross_entropy.py",
         "• Online Softmax (LogSumExp) алгоритмі арқылы C=128 өлшемді чанктермен есептеледі.\n"
         "• [2048 x 151936] FP16 логиттер (1.24 GB) HBM-де сақталмайды -> 77.8 MB буфер (93.7% үнемдеу!).\n"
         "• Streaming Shannon Entropy H(p) = LSE(z) - mean(z) және dlogits бір өтуде анықталады."),
        ("Fused RMSNorm + Fused Q/K RoPE", "solution/fused_rmsnorm.py & fused_rope.py",
         "• RMSNorm: PyTorch-тың 5 бөлек CUDA ядросының орнына 1 SRAM өтуі; дисперсия FP32 регистрлерінде.\n"
         "• QLoRA-да норма салмақтары қатып тұрғандықтан (requires_grad=False) dW редукциясы өткізіледі.\n"
         "• RoPE: Q (12 head) және K (2 head) бір кернелде, rotate_half/cat аралық тензорларынсыз бұрылады."),
        ("Byte-Packed NF4 Dequantization", "solution/nf4_dequant.py",
         "• 16-мәндік NF4 кестесі Triton регистрінде (Register LUT), double-quant absmax қолдауымен.\n"
         "• 4-биттік салмақтарды SRAM ішінде FP16-ға ашып, bitsandbytes dequantize_4bit орнын басады."),
        ("INT8 SwiGLU + Dynamic Padding + K-Bit MP", "solution/swiglu.py, dynamic_padding.py, apply()",
         "• INT8 SwiGLU: forward дәл FP16; backward үшін (gate, up) group-128 INT8 түрінде сақталады.\n"
         "• Dynamic Padding: батч ішіндегі артық pad-токендер кесіледі, батч реті өзгермейді.\n"
         "• K-Bit Mixed Precision: тек нормалар FP32-де, embed/lm_head FP16-да қалады.")
    ]

    coords = [(0.8, 1.6), (6.8, 1.6), (0.8, 4.3), (6.8, 4.3)]
    for (title, filename, desc), (x, y) in zip(cards, coords):
        add_card(slide, x, y, 5.7, 2.5)
        tb = slide.shapes.add_textbox(Inches(x + 0.2), Inches(y + 0.15), Inches(5.3), Inches(2.2))
        tf = tb.text_frame
        tf.word_wrap = True

        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.bold = True
        p0.font.size = Pt(13)
        p0.font.color.rgb = ACCENT_CYAN

        p_file = tf.add_paragraph()
        p_file.text = f"Файл: {filename}"
        p_file.font.size = Pt(9.5)
        p_file.font.color.rgb = ACCENT_EMER

        p_body = tf.add_paragraph()
        p_body.text = desc
        p_body.font.size = Pt(10)
        p_body.font.color.rgb = TEXT_BODY
        p_body.space_before = Pt(3)


# ==============================================================================
# SLIDE 4: Question 3 — Official Results Benchmark
# ==============================================================================
def build_slide_4(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_header(slide, "3-Сұрақ: Базамен салыстырғандағы нәтижелер", "РЕСМИ KAGGLE TESLA T4 БЕНЧМАРКІ (RUNS/RESULT.JSON)")

    # Benchmark Table
    rows = [
        ["Метрика / Көрсеткіш", "Базалық Soup 0.77.0", "LunarTeam Шешімі", "Өзгеріс / Ratio", "Қорытынды"],
        ["Пиктік бейнежад (Peak VRAM)", "14,345 МиБ (14.35 GB)", "4,231 МиБ (4.23 GB)", "Memory 3.390x", "🔥 10,114 МиБ (10.1 GB) үнемделді!"],
        ["Толық оқыту уақыты (soup train)", "79.38 с", "65.10 с", "Speedup 1.219x", "⚡ 14.3 секундқа жеделдеді!"],
        ["Өз Triton кернелдерінің үлесі", "0.0%", "13.3%", "Талап: >= 5.0%", "✅ Check 4 PASS (2.6 есе асты)"],
        ["1-қадамдағы Loss ауытқуы (ΔL1)", "2.05893", "2.05870", "Delta = 0.011%", "✅ Check 6 PASS (<= 1.0% шегі)"],
        ["64 қадамдық орташа Loss айырмасы", "0.00%", "0.060%", "Delta = 0.060%", "✅ Check 6 PASS (<= 3.0% шегі)"],
        ["Streaming Shannon Entropy айырмасы", "0.00%", "0.222%", "Delta = 0.222%", "✅ Check 5 PASS (<= 3.0% шегі)"],
        ["Eval Loss (Жасырын қатарлар)", "2.04970", "2.04956", "Delta = -0.007%", "💎 Check 7 PASS (Сапа жақсарды)"],
        ["Triton кернелдерінің unit-тесттері", "—", "21 passed in 18.02s", "100% жасыл", "✅ PyTorch эталонымен сәйкес"],
        ["Soup регрессиялық тесттері", "79 passed", "79 passed in 0.38s", "100% сәйкес", "✅ Check 10 PASS (79/79)"],
        ["Барлық ережелер (check_rules.py)", "—", "10 / 10 PASS", "ALL PASS: TRUE", "🏆 20/20 ұпай кепілдендірілді"]
    ]

    table_shape = slide.shapes.add_table(len(rows), 5, Inches(0.8), Inches(1.6), Inches(11.733), Inches(4.5))
    table = table_shape.table

    col_widths = [Inches(3.2), Inches(2.2), Inches(2.2), Inches(1.8), Inches(2.333)]
    for i, w in enumerate(col_widths):
        table.columns[i].width = w

    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = cell_text = val
            cell.fill.solid()
            if r_idx == 0:
                cell.fill.fore_color.rgb = RGBColor(14, 116, 144)  # Cyan header
            elif r_idx % 2 == 1:
                cell.fill.fore_color.rgb = CARD_BG
            else:
                cell.fill.fore_color.rgb = RGBColor(24, 34, 50)

            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
                p.font.name = "Arial"
                p.font.size = Pt(10 if r_idx > 0 else 10.5)
                p.font.bold = (r_idx == 0 or c_idx in (2, 3))
                if r_idx == 0:
                    p.font.color.rgb = TEXT_WHITE
                elif c_idx == 3 and "3.390x" in cell_text:
                    p.font.color.rgb = ACCENT_EMER
                elif c_idx == 4 and "PASS" in cell_text:
                    p.font.color.rgb = ACCENT_EMER
                else:
                    p.font.color.rgb = TEXT_BODY

    # Bottom Callout Note
    note_box = slide.shapes.add_textbox(Inches(0.8), Inches(6.25), Inches(11.733), Inches(0.8))
    tf_n = note_box.text_frame
    pn = tf_n.paragraphs[0]
    pn.text = "💡 Практикалық маңызы: Үнемделген 10.1 ГБ VRAM арқасында тегін Kaggle T4 картасында контент ұзындығын 512-ден 2048-ге өсіруге немесе батчты 4 есе арттыруға толық мүмкіндік бар!"
    pn.font.size = Pt(11)
    pn.font.bold = True
    pn.font.color.rgb = ACCENT_CYAN


# ==============================================================================
# SLIDE 5: Question 4 — Limitations & Roadmap
# ==============================================================================
def build_slide_5(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_header(slide, "4-Сұрақ: Шешімнің шектеулері және келесі қадамдар", "ИНЖЕНЕРЛІК ШЕКТЕУЛЕР ЖӘНЕ ROADMAP")

    # Left Card: Honest Limitations
    add_card(slide, 0.8, 1.6, 5.7, 5.2)
    b1 = slide.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf1 = b1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "Шешімнің Ашық Шектеулері (Limitations):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(239, 68, 68)  # Red alert

    limits = [
        ("Turing T4 Архитектурасы (sm_75):", "Tesla T4 картасында bfloat16 аппараттық нұсқаулықтары жоқ, тек FP16. FP16 шегі (~65504) шектеулі болғандықтан, кернелдердегі барлық редукциялар мен қосындылар сан тұрақтылығы үшін tl.float32 регистрлерінде орындалып, шығысы FP16-ға түсірілді."),
        ("Алғашқы 2 қадамдағы JIT-компиляция шығыны:", "run_task.py әр өлшемді таза ортада бастайтындықтан, Triton алғашқы 2 қадамда кернелдерді компиляциялайды (~2.5 с). Ұзақ оқытуларда (1000+ қадам) бұл overhead нөлге теңеседі."),
        ("Шағын батч өлшемі (Batch=4, Seq=512):", "M = 2048 қатары T4 GPU-дың барлық 40 Streaming Multiprocessor (SM) блоктарын толық толтырмайды. Жад үнемделген соң батчты көбейтуге болады.")
    ]
    for title, desc in limits:
        pt = tf1.add_paragraph()
        pt.text = f"⚠ {title}"
        pt.font.bold = True
        pt.font.size = Pt(11)
        pt.font.color.rgb = TEXT_WHITE
        pt.space_before = Pt(8)
        pd = tf1.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10)
        pd.font.color.rgb = TEXT_BODY

    # Right Card: Roadmap / Next steps
    add_card(slide, 6.8, 1.6, 5.7, 5.2)
    b2 = slide.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf2 = b2.text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "Келесі Даму Бағыттары (Roadmap):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_EMER

    roadmap = [
        ("Flash-Attention Style Triton Fused Block:", "Attention қабатының Q x K^T, Softmax және V көбейтулерін аралық тензорларды сақтамай, толық чип ішіндегі бірыңғай кернелге біріктіру."),
        ("W4A16 Fused GEMM (NF4 MatMul):", "NF4 салмақтарын деквантизациялау мен матрицалық көбейтуді (GEMM) бір ортақ кернелде қосып, сызықтық қабаттарды тағы 1.3x жеделдету."),
        ("Sequence Packing (Padding-Free):", "Батч ішіндегі қысқа жолдарды бір ұзын тізбекке біріктіріп (cu_seqlens арқылы), бос токендерді 0%-ға дейін жою.")
    ]
    for title, desc in roadmap:
        pt = tf2.add_paragraph()
        pt.text = f"🚀 {title}"
        pt.font.bold = True
        pt.font.size = Pt(11)
        pt.font.color.rgb = ACCENT_CYAN
        pt.space_before = Pt(8)
        pd = tf2.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = TEXT_BODY


# ==============================================================================
# SLIDE 6: Conclusion Slide
# ==============================================================================
def build_slide_6(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_header(slide, "Қорытынды және Байқауды Тапсыру (Conclusion)", "LUNARTEAM — ЖЕҢІСКЕ ДАЙЫН ШЕШІМ")

    # 3 Summary Cards
    cards = [
        ("🏆 10 / 10 PASS", "Толық Ережелік Сәйкестік",
         "• PyTest: 21/21 + Soup 79/79 passed\n"
         "• Kernel share: 13.3% (>= 5%)\n"
         "• Eval loss: -0.007% (сақталды)\n"
         "• 0 NaN/Inf барлық 64 қадамда"),
        ("⚡ 1.219x Speedup", "Оқытуды Жеделдету",
         "• База: 79.38 с -> Бізде: 65.10 с\n"
         "• 14.3 секунд таза уақыт үнемделді\n"
         "• Fused CE: 1.24 GB -> 78 MB\n"
         "• Dynamic unpadding: -35% FLOPs"),
        ("🔥 3.390x VRAM Ratio", "Жад Бойынша Чемпион",
         "• База: 14,345 MiB (14.35 GB)\n"
         "• LunarTeam: 4,231 MiB (4.23 GB)\n"
         "• Үнемделген: 10,114 MiB (10.1 GB!)\n"
         "• OOM қаупі толық жойылды")
    ]

    for idx, (head, sub, text) in enumerate(cards):
        x = 0.8 + idx * 4.0
        add_card(slide, x, 1.6, 3.733, 4.0)
        tb = slide.shapes.add_textbox(Inches(x + 0.2), Inches(1.8), Inches(3.333), Inches(3.6))
        tf = tb.text_frame
        tf.word_wrap = True

        p0 = tf.paragraphs[0]
        p0.text = head
        p0.font.bold = True
        p0.font.size = Pt(16)
        p0.font.color.rgb = ACCENT_CYAN

        p1 = tf.add_paragraph()
        p1.text = sub
        p1.font.bold = True
        p1.font.size = Pt(11)
        p1.font.color.rgb = ACCENT_EMER
        p1.space_before = Pt(4)

        p2 = tf.add_paragraph()
        p2.text = f"\n{text}"
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = TEXT_BODY

    # Bottom Links Card
    add_card(slide, 0.8, 5.8, 11.733, 1.0, bg_color=CARD_BG, border_color=ACCENT_CYAN)
    bl = slide.shapes.add_textbox(Inches(1.0), Inches(5.9), Inches(11.333), Inches(0.8))
    tfl = bl.text_frame
    tfl.word_wrap = True
    pl = tfl.paragraphs[0]
    pl.text = (
        "🔗 GitHub Репозиторий: https://github.com/ninumaki/SoupLunar (Ветка: main, финалдық коммит)\n"
        "🔗 Жария Kaggle Ноутбугы: https://www.kaggle.com/code/ruekdp/notebookd4dc02b276 (Run All -> 10/10 PASS)"
    )
    pl.font.size = Pt(10.5)
    pl.font.bold = True
    pl.font.color.rgb = TEXT_WHITE


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    build_slide_1(prs)
    build_slide_2(prs)
    build_slide_3(prs)
    build_slide_4(prs)
    build_slide_5(prs)
    build_slide_6(prs)

    prs.save(str(OUTPUT_FILE))
    print(f"Presentation saved successfully to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
