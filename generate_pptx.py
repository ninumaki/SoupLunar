"""Generate professional PowerPoint presentation for Decentrathon 6.0 Soup Kernel Challenge."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # Colors
    c_bg = RGBColor(15, 23, 42)        # Deep Navy Slate
    c_card = RGBColor(30, 41, 59)      # Slate Card
    c_cyan = RGBColor(6, 182, 212)     # Tech Cyan
    c_green = RGBColor(16, 185, 129)   # Emerald Accent
    c_white = RGBColor(248, 250, 252)  # Bright White
    c_gray = RGBColor(148, 163, 184)   # Subtle Gray

    def add_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = c_bg
        bg.line.fill.background()
        return bg

    def add_header(slide, title, category="SOUP KERNEL CHALLENGE | DECENTRATHON 6.0"):
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(1.2))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p1 = tf.paragraphs[0]
        p1.text = category.upper()
        p1.font.size = Pt(11)
        p1.font.bold = True
        p1.font.color.rgb = c_cyan

        p2 = tf.add_paragraph()
        p2.text = title
        p2.font.size = Pt(26)
        p2.font.bold = True
        p2.font.color.rgb = c_white

    # ==================== SLIDE 1: TITLE SLIDE ====================
    s1 = prs.slides.add_slide(blank_layout)
    add_bg(s1)

    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "DECENTRATHON 6.0 | SOUP KERNEL CHALLENGE"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = c_cyan

    p = tf.add_paragraph()
    p.text = "Triton Кернелдері арқылы Qwen2.5 Fine-Tuning Жеделдету мен Жадты Үнемдеу"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = c_white

    p = tf.add_paragraph()
    p.text = "\nKaggle Tesla T4 GPU-да 2.5 ГБ VRAM үнемдеу және оқыту қадамдарын 1.35x жеделдету"
    p.font.size = Pt(18)
    p.font.color.rgb = c_gray

    p = tf.add_paragraph()
    p.text = "\nКоманда: LunarTeam  |  Спикер & Автор: Тоғжан (Role 4: Құжаттама, Kaggle және Питч)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = c_green

    # ==================== SLIDE 2: QUESTION 1 ====================
    s2 = prs.slides.add_slide(blank_layout)
    add_bg(s2)
    add_header(s2, "1-Сұрақ: Қай операцияларды жеделдеттік және неге?")

    # Left Card: The Problem & Profile
    c1 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    c1.fill.solid()
    c1.fill.fore_color.rgb = c_card
    c1.line.color.rgb = c_cyan
    tf1 = c1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "📊 Негізгі Проблема (Profiling Analysis)"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = c_cyan

    items1 = [
        ("44% GPU уақыты", "Ұсақ элементтік операциялар мен тип түрлендіруге (casts) кетеді (RMSNorm, activations, LoRA adds)."),
        ("14.3 ГБ / 15 ГБ VRAM", "Базалық Soup жадтың 93%-ын толтырып, OOM (Out Of Memory) шегінде тұрды."),
        ("Memory Bandwidth Bound", "PyTorch әрбір Layer-де DRAM-ға 5+ рет қатынап, өткізу қабілетін тұншықтырады."),
        ("36% GEMM & 7% Softmax", "Матрица көбейту мен 152k сөздік шығыны.")
    ]
    for h, b in items1:
        p_h = tf1.add_paragraph()
        p_h.text = f"• {h}: "
        p_h.font.size = Pt(13)
        p_h.font.bold = True
        p_h.font.color.rgb = c_white
        p_b = tf1.add_paragraph()
        p_b.text = f"  {b}"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = c_gray

    # Right Card: The Solution Strategy
    c2 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
    c2.fill.solid()
    c2.fill.fore_color.rgb = c_card
    c2.line.color.rgb = c_green
    tf2 = c2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "⚡ Біздің Таңдауымыз бен Стратегиямыз"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = c_green

    items2 = [
        ("Fused RMSNorm (Өз Triton кернеліміз)", "Qwen2.5-те 28 блокта 57 қабат RMSNorm бар. Оларды бірыңғай SRAM кернеліне біріктірдік."),
        ("Fused SwiGLU Activation", "SiLU(gate) * up операциясын 1024 өлшемді блоктармен чип ішінде есептедік."),
        ("Zero Extra DRAM Footprint", "Аралық үлкен тензорларды HBM-ге жазбай, жад тапшылығын бірден шештік."),
        ("Exact Math & 100% Quality", "Математикалық дәлдік сақталды (Loss ауытқуы < 0.01%).")
    ]
    for h, b in items2:
        p_h = tf2.add_paragraph()
        p_h.text = f"✓ {h}: "
        p_h.font.size = Pt(13)
        p_h.font.bold = True
        p_h.font.color.rgb = c_white
        p_b = tf2.add_paragraph()
        p_b.text = f"  {b}"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = c_gray

    # ==================== SLIDE 3: QUESTION 2 ====================
    s3 = prs.slides.add_slide(blank_layout)
    add_bg(s3)
    add_header(s3, "2-Сұрақ: Кернел қалай құрылған?")

    # 3 columns for kernel deep-dive
    col_w = Inches(3.7)
    gap = Inches(0.3)
    left_m = Inches(0.8)

    # Box 1: Forward Tiling
    b1 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_m, Inches(1.8), col_w, Inches(5.0))
    b1.fill.solid()
    b1.fill.fore_color.rgb = c_card
    b1.line.color.rgb = c_cyan
    tf = b1.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "1. Forward SRAM Tiling"
    tf.paragraphs[0].font.size = Pt(16)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = c_cyan
    p = tf.add_paragraph()
    p.text = "\n• BLOCK_SIZE = 2048 (hidden_size = 1536 толық сияды)\n• Әр бағдарлама блогы 1 толық жолды өңдейді\n• 1 рет жүктеу (Load) -> FP32 Variance -> rsqrt(var + eps) -> Weight көбейту\n• Шығыс FP16 күйінде бірден жазылады (1 Store)\n• 5 бөлек CUDA kernel орнына тек 1 Triton kernel!"
    p.font.size = Pt(12)
    p.font.color.rgb = c_white

    # Box 2: Backward Optimization
    b2 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_m + col_w + gap, Inches(1.8), col_w, Inches(5.0))
    b2.fill.solid()
    b2.fill.fore_color.rgb = c_card
    b2.line.color.rgb = c_green
    tf = b2.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "2. Backward Autograd"
    tf.paragraphs[0].font.size = Pt(16)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = c_green
    p = tf.add_paragraph()
    p.text = "\n• Аралық FP32 активтендірулерді сақтамаймыз (Жад үнемдеу!)\n• Тек 1 өлшемді rrms векторын (N_rows) ғана сақтаймыз\n• Градиент dx тікелей формула бойынша регистрде есептеледі:\n  dx = rrms*(dy*w - sum_term)\n• dW редукциясы чипте жинақталады\n• Градиент қателігі 0!"
    p.font.size = Pt(12)
    p.font.color.rgb = c_white

    # Box 3: Seamless Integration
    b3 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_m + (col_w + gap)*2, Inches(1.8), col_w, Inches(5.0))
    b3.fill.solid()
    b3.fill.fore_color.rgb = c_card
    b3.line.color.rgb = c_cyan
    tf = b3.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = "3. apply() Патчингі"
    tf.paragraphs[0].font.size = Pt(16)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = c_cyan
    p = tf.add_paragraph()
    p.text = "\n• Unsloth тәрізді бір ғана solution.apply() шақыруымен қосылады\n• Qwen2RMSNorm.forward және Qwen2MLP.forward модулін қорғалған түрде ауыстырады\n• Тесттердегі CPU және fake модульдерге кедергі жасамайды\n• 79 Soup регрессиялық тестінен 100% өтеді!"
    p.font.size = Pt(12)
    p.font.color.rgb = c_white

    # ==================== SLIDE 4: QUESTION 3 (RESULTS) ====================
    s4 = prs.slides.add_slide(blank_layout)
    add_bg(s4)
    add_header(s4, "3-Сұрақ: Базамен салыстырғандағы нәтижелер")

    # Metrics highlight boxes
    m_data = [
        ("Speedup (s)", "1.23x", "Толық уақыт 81.4с -> 66.2с\n(Оқыту қадамы: 1.35x)", c_cyan),
        ("Peak Memory (m)", "1.21x", "14.3 ГБ -> 11.8 ГБ\nҮнемделді: 2.5 ГБ VRAM", c_green),
        ("Kernel Share", "7.84%", "Өз кернелдеріміздің үлесі\n(Талап: >= 5.0%)", c_cyan),
        ("Eval Loss", "2.050", "Базамен дәлме-дәл\n(Delta: -0.01% <= 0.5%)", c_green),
    ]

    box_w = Inches(2.7)
    for i, (label, val, sub, col) in enumerate(m_data):
        bx = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + i*2.95), Inches(1.8), box_w, Inches(2.1))
        bx.fill.solid()
        bx.fill.fore_color.rgb = c_card
        bx.line.color.rgb = col
        tf = bx.text_frame
        tf.word_wrap = True
        tf.paragraphs[0].text = label
        tf.paragraphs[0].font.size = Pt(13)
        tf.paragraphs[0].font.color.rgb = c_gray
        p_val = tf.add_paragraph()
        p_val.text = val
        p_val.font.size = Pt(28)
        p_val.font.bold = True
        p_val.font.color.rgb = col
        p_sub = tf.add_paragraph()
        p_sub.text = sub
        p_sub.font.size = Pt(10)
        p_sub.font.color.rgb = c_white

    # Detailed Comparison Table Card
    tbl_card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.2), Inches(11.7), Inches(2.6))
    tbl_card.fill.solid()
    tbl_card.fill.fore_color.rgb = c_card
    tbl_card.line.fill.background()
    tf_tbl = tbl_card.text_frame
    tf_tbl.word_wrap = True
    p = tf_tbl.paragraphs[0]
    p.text = "🎯 Хакатон Ережелері Тексерісі (Check Rules & Final Score):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = c_white

    p2 = tf_tbl.add_paragraph()
    p2.text = (
        "• 10/10 Harness Тексерісі: Барлығы сәтті өтті (PASS ✅) -> 20 / 20 ұпай толық кепілдендірілді!\n"
        "• Техникалық құжаттама (README.md & SUBMISSION.md): 3 бөлім де 5/5 -> 15 / 15 ұпай кепілдендірілді!\n"
        "• 2.5 ГБ VRAM үнемдеу арқылы тегін Kaggle T4-те контекст ұзындығын 512-ден 1024 токенге дейін өсіруге толық мүмкіндік бар!"
    )
    p2.font.size = Pt(13)
    p2.font.color.rgb = c_gray

    # ==================== SLIDE 5: QUESTION 4 ====================
    s5 = prs.slides.add_slide(blank_layout)
    add_bg(s5)
    add_header(s5, "4-Сұрақ: Шешімнің шектеулері және келесі қадамдар")

    # Left: Limitations
    c_lim = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    c_lim.fill.solid()
    c_lim.fill.fore_color.rgb = c_card
    c_lim.line.color.rgb = RGBColor(239, 68, 68)  # Red warning
    tf = c_lim.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "⚠️ Шешімнің Шектеулері (Limitations)"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = RGBColor(239, 68, 68)

    lims = [
        ("T4 Архитектурасы (No BF16)", "T4 картасында bfloat16 және FP8 Tensor Core жоқ. Есептеулер тек FP16 күйінде шектеледі."),
        ("JIT Компиляция Салмағы", "Алғашқы 2 қадамда Triton кернелдерді жинақтайды. 64 қадамдық қысқа оқытуда 2-3 секунд компиляцияға кетеді."),
        ("Кіші Батч (Batch 4, Seq 512)", "2048 жол үшін GPU-дың барлық SM блоктары 100% толық жүктелмей қалуы мүмкін."),
        ("NF4 GEMM тәуекелі", "NF4 декванттауды толық Triton-ға ауыстыру дәлдік ауытқуы салдарынан қосылмады.")
    ]
    for h, b in lims:
        p_h = tf.add_paragraph()
        p_h.text = f"• {h}:"
        p_h.font.size = Pt(13)
        p_h.font.bold = True
        p_h.font.color.rgb = c_white
        p_b = tf.add_paragraph()
        p_b.text = f"  {b}"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = c_gray

    # Right: Next Steps
    c_next = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))
    c_next.fill.solid()
    c_next.fill.fore_color.rgb = c_card
    c_next.line.color.rgb = c_cyan
    tf = c_next.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🚀 Келесі Қадамдар (Roadmap)"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = c_cyan

    nexts = [
        ("Fused Chunked Cross-Entropy", "152k сөздік үшін толық логиттер тензорын шығармай, шағын chunk-тармен шығынды Triton-да есептеу (тағы 2 ГБ үнемдеу!)."),
        ("Padding-free Sequence Packing", "Батч ішіндегі нөлдік pad-токендерді Triton арқылы алып тастап, тек нақты токендерді өңдеу."),
        ("Custom Fused LoRA Forward", "LoRA қабатындағы көбейту мен қосуды (x + W_lora*x) бір операцияға біріктіру."),
        ("Ұзақ оқытуға масштабтау", "1000+ қадамда JIT overhead жойылып, 1.35x+ жеделдету тұрақты сақталады.")
    ]
    for h, b in nexts:
        p_h = tf.add_paragraph()
        p_h.text = f"→ {h}:"
        p_h.font.size = Pt(13)
        p_h.font.bold = True
        p_h.font.color.rgb = c_white
        p_b = tf.add_paragraph()
        p_b.text = f"  {b}"
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = c_gray

    # ==================== SLIDE 6: CONCLUSION ====================
    s6 = prs.slides.add_slide(blank_layout)
    add_bg(s6)
    add_header(s6, "Қорытынды және Нәтижелер", "SOUP OPTIMIZERS | DECENTRATHON 6.0")

    c_fin = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(1.8), Inches(10.3), Inches(4.8))
    c_fin.fill.solid()
    c_fin.fill.fore_color.rgb = c_card
    c_fin.line.color.rgb = c_green
    tf = c_fin.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "🎉 Түйін: Тегін Kaggle T4-те Өнімді Оқыту Енді Қолжетімді!"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = c_green

    p2 = tf.add_paragraph()
    p2.text = (
        "\n1. Кепілдендірілген 15 Ұпай (Құжаттама): ТАЛАПТАРҒА ТОЛЫҚ САЙ README.md, SUBMISSION.md.\n"
        "2. Жұмыс істеуіне 20 Ұпай: 10/10 автоматты тексеруден және 79 Soup регрессиялық тестінен өтті.\n"
        "3. Жылдамдық & Жад Ұпайы: 1.23x жеделдету, 2.5 ГБ VRAM үнемдеу, өз кернелдеріміздің үлесі 7.84%.\n"
        "4. Дәлдік: Жасырын қазақша деректерде оқыту сапасы базамен толық сәйкес (Delta = -0.01%).\n\n"
        "🔗 Kaggle Public Run: kaggle.com/code/ruekdp/notebookd4dc02b276\n"
        "🔗 GitHub Repo: github.com/ninumaki/SoupLunar\n\n"
        "Назарларыңызға рақмет! Сұрақтарыңызға жауап беруге дайынбыз!"
    )
    p2.font.size = Pt(14)
    p2.font.color.rgb = c_white

    output_path = "/home/togzhan/Рабочий стол/децетратон/presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    create_deck()
