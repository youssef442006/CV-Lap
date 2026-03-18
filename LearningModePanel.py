from PyQt5.QtWidgets import (
    QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen

# ═══════════════════════════════════════════════════════════════
#  LESSON DATA  (unchanged)
# ═══════════════════════════════════════════════════════════════
LESSONS = [
  {
    "id": 1, "icon": "🔲", "title": "Convolution",
    "subtitle": "The engine behind every image filter",
    "color": "#00ffcc", "dark": "#003322",
    "concepts": ["Kernel / Filter", "Weighted sum", "Padding", "Kernel size effect"],
    "theory": """<div style='font-family:Consolas,monospace;font-size:10px;color:#ccc;line-height:170%;'>
<span style='color:#00ffcc;font-size:12px;font-weight:bold;'>What is Convolution?</span><br><br>
A kernel (small matrix) slides over every pixel.<br>
At each position: <b style='color:#ffcc44;'>multiply + sum</b> the overlapping values.<br><br>
<span style='color:#888;'>┌─────────────────────────────────┐</span><br>
<span style='color:#888;'>│  Image patch    Kernel          │</span><br>
<span style='color:#888;'>│  │10 20 30│  ×  │1/9 1/9 1/9│  │</span><br>
<span style='color:#888;'>│  │40 50 60│     │1/9 1/9 1/9│  │</span><br>
<span style='color:#888;'>│  │70 80 90│     │1/9 1/9 1/9│  │</span><br>
<span style='color:#888;'>│  → output = (10+20+...+90)/9 = 50  │</span><br>
<span style='color:#888;'>└─────────────────────────────────┘</span><br><br>
<span style='color:#ffaa00;font-weight:bold;'>The kernel controls the effect:</span><br>
<span style='color:#aaa;'>Sum = 1 →</span> <span style='color:#00ffcc;'>blur (brightness preserved)</span><br>
<span style='color:#aaa;'>Sum = 0 →</span> <span style='color:#ff6644;'>edge detection (dark output)</span><br>
<span style='color:#aaa;'>Negative values →</span> <span style='color:#ffcc44;'>sharpening / enhancement</span><br><br>
<span style='color:#ffaa00;font-weight:bold;'>Kernel size:</span><br>
3×3 → looks at 9 neighbors → subtle effect<br>
7×7 → looks at 49 neighbors → strong effect<br>
Larger = stronger effect <b>but slower</b>
</div>""",
    "steps": [
      ("Apply Gaussian Blur (size 3)", "Gaussian Blur", "Small kernel — subtle smoothing"),
      ("Apply Gaussian Blur (size 11)", "Gaussian Blur", "Larger kernel — strong blur"),
      ("Apply Sharpen", "Sharpen", "Negative weights boost edges"),
      ("Apply Emboss", "Emboss", "Asymmetric kernel = directional light"),
    ],
    "insight": "💡 Open the 🔲 Kernel tab while switching filters to see the matrix change!",
  },
  {
    "id": 2, "icon": "〰️", "title": "Frequency Filters",
    "subtitle": "Every image is a mix of frequencies",
    "color": "#44aaff", "dark": "#001a33",
    "concepts": ["Frequency domain", "Low-pass", "High-pass", "Fourier Transform"],
    "theory": """<div style='font-family:Consolas,monospace;font-size:10px;color:#ccc;line-height:170%;'>
<span style='color:#44aaff;font-size:12px;font-weight:bold;'>Spatial Frequencies</span><br><br>
Think of an image as a mix of waves of different frequencies:<br><br>
<span style='color:#44aaff;'>Low frequencies</span> = slow changes = sky, walls, flat areas<br>
<span style='color:#ff6644;'>High frequencies</span> = rapid changes = edges, texture, noise<br><br>
<span style='color:#ffaa00;font-weight:bold;'>Low Pass Filter (LPF):</span><br>
Keeps low freq → <b>blurs</b> the image, removes edges and noise.<br><br>
<span style='color:#ffaa00;font-weight:bold;'>High Pass Filter (HPF):</span><br>
Keeps high freq → <b>only edges remain</b>, flat areas → gray (128).<br><br>
<span style='color:#888;font-style:italic;'>Math: HPF = Original − LPF(Original)</span><br><br>
<span style='color:#ffaa00;font-weight:bold;'>Unsharp Masking (sharpening):</span><br>
<span style='color:#aaffcc;'>output = original + k × (original − blur)</span><br>
Add back the high frequencies you removed!
</div>""",
    "steps": [
      ("Apply Low Pass Filter", "Low Pass Filter", "Blurs — removes high frequencies"),
      ("Apply High Pass Filter", "High Pass Filter", "Edges only — flat areas → gray"),
      ("Apply Sharpen", "Sharpen", "Adds high freq back = unsharp mask"),
      ("Apply Gaussian Blur", "Gaussian Blur", "Compare with Low Pass Filter"),
    ],
    "insight": "💡 Pipeline: Low Pass → High Pass → they almost cancel each other out!",
  },
  {
    "id": 3, "icon": "📐", "title": "Edge Detection",
    "subtitle": "Finding where things change",
    "color": "#ffaa44", "dark": "#221500",
    "concepts": ["Gradient", "Sobel operator", "Canny pipeline", "Thresholds"],
    "theory": """<div style='font-family:Consolas,monospace;font-size:10px;color:#ccc;line-height:170%;'>
<span style='color:#ffaa44;font-size:12px;font-weight:bold;'>What is an Edge?</span><br><br>
An edge = a pixel where intensity changes <b>rapidly</b>.<br>
We detect it by computing the <b style='color:#ffcc44;'>gradient</b> (rate of change).<br><br>
<span style='color:#888;'>Pixel row: </span><span style='color:#44aaff;'>10 10 10 </span><span style='color:#ff6644;'>10 80 90 </span><span style='color:#44aaff;'>90 90</span><br>
<span style='color:#888;'>Gradient:  </span><span style='color:#44aaff;'>0  0  </span><span style='color:#ff6644;'>0  70 10 </span><span style='color:#44aaff;'>0  0  ← EDGE</span><br><br>
<span style='color:#ffaa00;font-weight:bold;'>Sobel:</span> Gx + Gy → G = √(Gx² + Gy²)<br>
<span style='color:#ffaa00;font-weight:bold;'>Laplacian:</span> ∇²f = 2nd derivative, all directions<br><br>
<span style='color:#ffaa00;font-weight:bold;'>Canny (best approach):</span><br>
1→ Blur  2→ Sobel  3→ Thin to 1px  4→ Double threshold  5→ Hysteresis
</div>""",
    "steps": [
      ("Apply Sobel Edges", "Sobel Edges", "Gradient magnitude — thick edges"),
      ("Apply Laplacian", "Laplacian", "2nd derivative — all directions at once"),
      ("Apply Canny Edges", "Canny Edges", "Full pipeline — clean 1px edges"),
      ("Blur first, then Canny", "Gaussian Blur", "Pre-blur → much cleaner result"),
    ],
    "insight": "💡 Lower Canny min-threshold = more edges detected. Higher = fewer but stronger.",
  },
  {
    "id": 4, "icon": "🎚️", "title": "Thresholding",
    "subtitle": "Separating objects from background",
    "color": "#ff88cc", "dark": "#220011",
    "concepts": ["Binary threshold", "Otsu's method", "Bimodal histogram", "Preprocessing"],
    "theory": """<div style='font-family:Consolas,monospace;font-size:10px;color:#ccc;line-height:170%;'>
<span style='color:#ff88cc;font-size:12px;font-weight:bold;'>Thresholding = Image → Binary</span><br><br>
<span style='color:#aaffcc;'>pixel > T → 255 (white) | pixel ≤ T → 0 (black)</span><br><br>
<span style='color:#888;'>Histogram:  background  text</span><br>
<span style='color:#888;'>count         ████         ███</span><br>
<span style='color:#888;'>0    ────────────────────── 255</span><br>
<span style='color:#888;'>                  ↑ optimal T here</span><br><br>
<span style='color:#ffaa00;font-weight:bold;'>Otsu's Method (automatic T):</span><br>
Tries every T (0–255), picks the one that<br>
<b>maximizes between-class variance</b>.<br>
Works best on bimodal histograms (two peaks).<br><br>
<b>Key insight:</b> Noise creates false crossings.<br>
<span style='color:#aaffcc;'>Always blur before thresholding!</span>
</div>""",
    "steps": [
      ("Apply Segmentation (Otsu)", "Segmentation (Otsu)", "Auto-threshold — no tuning needed"),
      ("Add Salt & Pepper noise first", "Salt and Pepper", "Noise corrupts thresholding"),
      ("Fix with Median Filter", "Median Filter", "Median removes S&P noise perfectly"),
      ("Gaussian Blur then Otsu", "Gaussian Blur", "Blur also helps clean the threshold"),
    ],
    "insight": "💡 Pipeline: Salt & Pepper → Median Filter → Otsu = noise-robust segmentation!",
  },
  {
    "id": 5, "icon": "🔮", "title": "Morphology",
    "subtitle": "Operating on shapes, not pixels",
    "color": "#aa88ff", "dark": "#110022",
    "concepts": ["Structuring element", "Erosion", "Dilation", "Opening / Closing"],
    "theory": """<div style='font-family:Consolas,monospace;font-size:10px;color:#ccc;line-height:170%;'>
<span style='color:#aa88ff;font-size:12px;font-weight:bold;'>Morphological Operations</span><br><br>
Works on <b>binary images</b> using a structuring element (SE).<br><br>
<span style='color:#ffaa00;font-weight:bold;'>Erosion</span> — shrinks white regions:<br>
<span style='color:#888;'>■■■■■■■  →  ─■■■■■─</span>  (removes noise dots)<br><br>
<span style='color:#ffaa00;font-weight:bold;'>Dilation</span> — expands white regions:<br>
<span style='color:#888;'>─■■■■■─  →  ■■■■■■■</span>  (fills small holes)<br><br>
<span style='color:#aa88ff;'>Opening</span>  = Erode → Dilate   removes noise<br>
<span style='color:#cc88ff;'>Closing</span>  = Dilate → Erode   fills holes<br>
<span style='color:#ffaa88;'>Gradient</span> = Dilate − Erode   object outline
</div>""",
    "steps": [
      ("Otsu to get binary image", "Segmentation (Otsu)", "Start with a binary mask"),
      ("Apply Erosion (iter=3)", "Erosion", "White regions shrink — noise disappears"),
      ("Apply Dilation (iter=3)", "Dilation", "Expands back — Opening complete"),
      ("Try Dilation then Erosion", "Dilation", "Reverse = Closing — fills holes"),
    ],
    "insight": "💡 Pipeline: Otsu → Erosion(3) → Dilation(3) = Opening = noise removed!",
  },
  {
    "id": 6, "icon": "🎲", "title": "Noise & Denoising",
    "subtitle": "Real cameras are never perfect",
    "color": "#ffcc44", "dark": "#221a00",
    "concepts": ["Gaussian noise", "Salt & pepper", "Median filter", "Bilateral filter"],
    "theory": """<div style='font-family:Consolas,monospace;font-size:10px;color:#ccc;line-height:170%;'>
<span style='color:#ffcc44;font-size:12px;font-weight:bold;'>Types of Noise</span><br><br>
<span style='color:#ffaa00;font-weight:bold;'>Gaussian Noise:</span> N(0,σ) added to each pixel.<br>
Caused by sensor heat. Fix: <span style='color:#aaffcc;'>Bilateral Filter</span><br><br>
<span style='color:#ffaa00;font-weight:bold;'>Salt & Pepper:</span> Random 0 or 255 pixels.<br>
Caused by dead pixels. Fix: <span style='color:#aaffcc;'>Median Filter</span><br><br>
<span style='color:#888;'>Why Median works for S&P:</span><br>
<span style='color:#888;'>Values: [10,12,255,11,9,13,10]</span><br>
<span style='color:#888;'>Mean = 45.7  ← wrong!</span><br>
<span style='color:#888;'>Median → </span><span style='color:#aaffcc;'>11 ✓  (outlier ignored)</span><br><br>
<b>Rule:</b> Median for S&P | Bilateral for Gaussian
</div>""",
    "steps": [
      ("Add Salt & Pepper noise", "Salt and Pepper", "Scattered black/white pixels"),
      ("Remove with Gaussian Blur", "Gaussian Blur", "Spreads noise — bad choice"),
      ("Remove with Median Filter", "Median Filter", "Noise gone, edges still sharp!"),
      ("Add Gaussian Noise, apply Bilateral", "Gaussian Noise", "Then apply Bilateral Filter"),
    ],
    "insight": "💡 Rule: Median for Salt&Pepper | Bilateral for Gaussian — never mix them up!",
  },
  {
    "id": 7, "icon": "🎨", "title": "Contrast Enhancement",
    "subtitle": "Making images more informative",
    "color": "#ff6688", "dark": "#220011",
    "concepts": ["Histogram", "Equalization", "CLAHE", "Gamma correction"],
    "theory": """<div style='font-family:Consolas,monospace;font-size:10px;color:#ccc;line-height:170%;'>
<span style='color:#ff6688;font-size:12px;font-weight:bold;'>Histogram & Contrast</span><br><br>
Poor contrast = histogram squeezed into a narrow range.<br><br>
<span style='color:#888;'>Dark:    █████</span><br>
<span style='color:#888;'>Enhanced: ██ █ ██ █ ██</span><br>
<span style='color:#888;'>          0────────255</span><br><br>
<span style='color:#ffaa00;font-weight:bold;'>Histogram EQ:</span> Stretches to full 0–255 range.<br>
Problem: over-enhances uniform areas.<br><br>
<span style='color:#ffaa00;font-weight:bold;'>CLAHE:</span> Local equalization in small tiles.<br>
Clip limit prevents noise amplification.<br>
Best for: faces, medical images, fog.<br><br>
<span style='color:#ffaa00;font-weight:bold;'>Gamma:</span> output = 255 × (input/255)^(1/γ)<br>
γ &lt; 1 → brightens  |  γ &gt; 1 → darkens
</div>""",
    "steps": [
      ("Apply Histogram Equalization", "Histogram Equalization", "Global stretch to 0–255"),
      ("Apply CLAHE", "CLAHE", "Local tiles — better on faces"),
      ("Apply Gamma Correction", "Gamma Correction", "γ<1 brightens shadows"),
      ("Apply Add (+) then CLAHE", "Add (+)", "Simulate overexposure, fix with CLAHE"),
    ],
    "insight": "💡 CLAHE is preferred over Histogram EQ for almost every real-world use case.",
  },
  {
    "id": 8, "icon": "🔑", "title": "Feature Detection",
    "subtitle": "Finding unique, repeatable points",
    "color": "#ffaa00", "dark": "#221400",
    "concepts": ["Keypoints", "Descriptors", "Scale invariance", "Matching"],
    "theory": """<div style='font-family:Consolas,monospace;font-size:10px;color:#ccc;line-height:170%;'>
<span style='color:#ffaa00;font-size:12px;font-weight:bold;'>Why Feature Detection?</span><br><br>
To match two images we need <b>distinctive points</b><br>
detectable regardless of scale, rotation, lighting.<br><br>
<span style='color:#888;'>Flat region → looks same everywhere (BAD)</span><br>
<span style='color:#888;'>Edge       → unique in one direction (OK)</span><br>
<span style='color:#aaffcc;'>Corner     → unique in ALL directions (BEST)</span><br><br>
<span style='color:#ffaa00;font-weight:bold;'>ORB</span> (fast, free): FAST corners + BRIEF 256-bit descriptor<br>
<span style='color:#ffaa00;font-weight:bold;'>SIFT</span> (robust): DoG scale-space + 128-D descriptor<br>
Invariant to: scale, rotation, illumination<br><br>
<b>Matching:</b> Lowe's ratio test: <span style='color:#aaffcc;'>d1 &lt; 0.7 × d2</span>
</div>""",
    "steps": [
      ("Apply ORB Features", "ORB Features", "Green dots = detected corners"),
      ("Apply SIFT Features", "SIFT Features", "Rich keypoints — scale + orientation"),
      ("Sharpen first, then ORB", "Sharpen", "Sharpening reveals more features"),
      ("Blur first, then ORB", "Gaussian Blur", "Blurring kills features — see the difference"),
    ],
    "insight": "💡 In Playground: SIFT Matching shows live feature matching between frames!",
  },
]


# ═══════════════════════════════════════════════════════════════
#  JOURNEY PATH WIDGET  —  the visual path map
# ═══════════════════════════════════════════════════════════════
class JourneyPath(QWidget):
    """Draws a connected path of lesson nodes."""
    node_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current   = 0
        self._completed = set()
        self.setFixedHeight(60)
        self.setStyleSheet("background:transparent;")
        self._btns = []
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(0)

        for i, l in enumerate(LESSONS):
            # connector line
            if i > 0:
                line = QLabel("──────")
                line.setObjectName(f"line_{i}")
                line.setAlignment(Qt.AlignCenter)
                line.setStyleSheet("color:#1a1a2a;font-size:10px;letter-spacing:-2px;")
                lay.addWidget(line, 1)

            btn = QPushButton(f"{i+1}")
            btn.setFixedSize(36, 36)
            btn.setToolTip(l["title"])
            btn.clicked.connect(lambda _, idx=i: self.node_clicked.emit(idx))
            lay.addWidget(btn)
            self._btns.append(btn)

        self._refresh()

    def update_state(self, current, completed):
        self._current   = current
        self._completed = completed
        self._refresh()

    def _refresh(self):
        for i, btn in enumerate(self._btns):
            color = LESSONS[i]["color"]
            dark  = LESSONS[i]["dark"]
            if i in self._completed:
                btn.setText("✓")
                btn.setStyleSheet(
                    "background:#003322;color:#00ff88;font-size:12px;font-weight:bold;"
                    "border:2px solid #00ff8866;border-radius:18px;")
            elif i == self._current:
                btn.setText(f"{i+1}")
                btn.setStyleSheet(
                    f"background:{dark};color:{color};font-size:11px;font-weight:bold;"
                    f"border:2px solid {color};border-radius:18px;")
            else:
                btn.setText(f"{i+1}")
                btn.setStyleSheet(
                    "background:#0a0a14;color:#2a2a44;font-size:11px;font-weight:bold;"
                    "border:2px solid #1a1a2a;border-radius:18px;")

        lay = self.layout()
        for j in range(lay.count()):
            item = lay.itemAt(j)
            if item and item.widget():
                w = item.widget()
                if w.objectName().startswith("line_"):
                    idx = int(w.objectName().split("_")[1])
                    if (idx-1) in self._completed:
                        w.setStyleSheet(f"color:#00ff8844;font-size:10px;letter-spacing:-2px;")
                    elif idx-1 == self._current:
                        color = LESSONS[idx-1]["color"]
                        w.setStyleSheet(f"color:{color}44;font-size:10px;letter-spacing:-2px;")
                    else:
                        w.setStyleSheet("color:#1a1a2a;font-size:10px;letter-spacing:-2px;")


# ═══════════════════════════════════════════════════════════════
#  LEARNING MODE PANEL  —  Guided Journey
# ═══════════════════════════════════════════════════════════════
class LearningModePanel(QWidget):
    apply_op_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current   = 0
        self._completed = set()
        self._step_done = set()   
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        path_container = QWidget()
        path_container.setStyleSheet(
            "background:#070710;border-bottom:1px solid #1a1a2a;")
        path_container.setFixedHeight(56)
        pc_lay = QVBoxLayout(path_container)
        pc_lay.setContentsMargins(0, 10, 0, 0)

        self._journey = JourneyPath()
        self._journey.node_clicked.connect(self._go)
        pc_lay.addWidget(self._journey)
        root.addWidget(path_container)

        # ── Progress bar ──────────────────────────────────────
        prog_row = QWidget()
        prog_row.setFixedHeight(28)
        prog_row.setStyleSheet("background:#060610;border-bottom:1px solid #111;")
        pr_lay = QHBoxLayout(prog_row); pr_lay.setContentsMargins(12,0,12,0)
        self._prog_lbl = QLabel("START YOUR JOURNEY")
        self._prog_lbl.setStyleSheet("color:#1a1a3a;font-size:9px;font-weight:bold;letter-spacing:2px;")
        self._steps_lbl = QLabel("")
        self._steps_lbl.setStyleSheet("color:#333;font-size:9px;")
        pr_lay.addWidget(self._prog_lbl); pr_lay.addStretch()
        pr_lay.addWidget(self._steps_lbl)
        root.addWidget(prog_row)

        # ── Main content ──────────────────────────────────────
        self._content_scroll = QScrollArea()
        self._content_scroll.setWidgetResizable(True)
        self._content_scroll.setStyleSheet(
            "QScrollArea{border:none;background:#0d0d16;}"
            "QScrollBar:vertical{background:#0a0a12;width:6px;border-radius:3px;}"
            "QScrollBar::handle:vertical{background:#222244;border-radius:3px;}")
        self._content_widget = QWidget()
        self._content_widget.setStyleSheet("background:#0d0d16;")
        self._content_lay = QVBoxLayout(self._content_widget)
        self._content_lay.setContentsMargins(12, 4, 12, 12)
        self._content_lay.setSpacing(10)
        self._content_scroll.setWidget(self._content_widget)
        root.addWidget(self._content_scroll, 1)

        # ── Bottom nav bar ────────────────────────────────────
        bar = QWidget()
        bar.setStyleSheet("background:#080810;border-top:1px solid #1a1a2a;")
        bar.setFixedHeight(48)
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(12,0,12,0); bar_lay.setSpacing(8)

        self._btn_prev = QPushButton("◀  Back")
        self._btn_next = QPushButton("Continue  ▶")
        for b in (self._btn_prev, self._btn_next):
            b.setFixedHeight(32)
            b.setStyleSheet(
                "background:#0d0d1a;color:#44ccff;font-size:10px;font-weight:bold;"
                "border:1px solid #224466;border-radius:5px;padding:0 14px;")
        self._btn_prev.clicked.connect(lambda: self._go(self._current - 1))
        self._btn_next.clicked.connect(self._advance)

        self._checkpoint_btn = QPushButton("⬡  Clear Checkpoint")
        self._checkpoint_btn.setFixedHeight(32)
        self._checkpoint_btn.setStyleSheet(
            "background:#0a0a12;color:#333;font-size:9px;font-weight:bold;"
            "border:1px solid #222;border-radius:5px;padding:0 12px;")
        self._checkpoint_btn.clicked.connect(self._clear_checkpoint)

        bar_lay.addWidget(self._btn_prev)
        bar_lay.addStretch()
        bar_lay.addWidget(self._checkpoint_btn)
        bar_lay.addStretch()
        bar_lay.addWidget(self._btn_next)
        root.addWidget(bar)

        self._load(0)

    # ── Navigate ──────────────────────────────────────────────
    def _go(self, idx):
        if 0 <= idx < len(LESSONS):
            self._step_done = set()
            self._load(idx)

    def _advance(self):
        n = len(LESSONS[self._current]["steps"])
        if len(self._step_done) >= n:
            self._clear_checkpoint()
        else:
            nxt = self._current + 1
            if nxt < len(LESSONS):
                self._go(nxt)

    def _clear_checkpoint(self):
        self._completed.add(self._current)
        self._journey.update_state(self._current, self._completed)
        self._update_progress_bar()
        self._refresh_checkpoint_btn()
        nxt = self._current + 1
        if nxt < len(LESSONS):
            self._go(nxt)

    # ── Load lesson ───────────────────────────────────────────
    def _load(self, idx):
        self._current = idx
        lesson = LESSONS[idx]
        color  = lesson["color"]
        dark   = lesson["dark"]

        self._journey.update_state(idx, self._completed)
        self._update_progress_bar()

        while self._content_lay.count():
            item = self._content_lay.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        # ── 1. Mission header ─────────────────────────────────
        mission = QWidget()
        mission.setStyleSheet(
            f"background:{dark};border:1px solid {color}33;border-radius:8px;")
        m_lay = QVBoxLayout(mission); m_lay.setContentsMargins(14,10,14,10); m_lay.setSpacing(4)

        cp_row = QHBoxLayout(); cp_row.setSpacing(8)
        cp_lbl = QLabel(f"CHECKPOINT  {idx+1} / {len(LESSONS)}")
        cp_lbl.setStyleSheet(
            f"color:{color}66;font-size:8px;letter-spacing:3px;font-weight:bold;")

        status = "✓  CLEARED" if idx in self._completed else ""
        status_lbl = QLabel(status)
        status_lbl.setStyleSheet("color:#00ff88;font-size:8px;font-weight:bold;letter-spacing:2px;")
        cp_row.addWidget(cp_lbl); cp_row.addStretch(); cp_row.addWidget(status_lbl)
        m_lay.addLayout(cp_row)

        t = QLabel(f"{lesson['icon']}  {lesson['title']}")
        t.setStyleSheet(f"color:{color};font-size:15px;font-weight:bold;")
        s = QLabel(lesson["subtitle"])
        s.setStyleSheet("color:#666;font-size:10px;")
        m_lay.addWidget(t); m_lay.addWidget(s)

        pills = QHBoxLayout(); pills.setSpacing(5)
        for c in lesson["concepts"]:
            pill = QLabel(c)
            pill.setStyleSheet(
                f"background:{color}18;color:#ffffff;font-size:9px;"
                f"border:1px solid {color}44;border-radius:10px;padding:2px 8px;")
            pills.addWidget(pill)
        pills.addStretch()
        m_lay.addLayout(pills)
        self._content_lay.addWidget(mission)

        # ── 2. Theory ─────────────────────────────────────────
        th_hdr = self._section_hdr("📖  Theory", color)
        self._content_lay.addWidget(th_hdr)
        theory_box = QLabel(lesson["theory"])
        theory_box.setWordWrap(True)
        theory_box.setTextFormat(Qt.RichText)
        theory_box.setAlignment(Qt.AlignTop)
        theory_box.setStyleSheet(
            f"background:#080810;border:1px solid {color}22;"
            "border-radius:6px;padding:12px;")
        self._content_lay.addWidget(theory_box)

        # ── 3. Journey steps ──────────────────────────────────
        steps_hdr_row = QHBoxLayout()
        st_hdr = self._section_hdr("🗺️  Your Path Through This Lesson", "#ffcc44")
        n_done = len(self._step_done)
        n_total = len(lesson["steps"])
        st_count = QLabel(f"{n_done} / {n_total} completed")
        st_count.setObjectName("steps_count")
        st_count.setStyleSheet(
            f"color:{'#00ff88' if n_done == n_total else '#444'};"
            "font-size:9px;font-weight:bold;")
        steps_hdr_row.addWidget(st_hdr); steps_hdr_row.addStretch()
        steps_hdr_row.addWidget(st_count)
        steps_container = QWidget(); steps_container.setLayout(steps_hdr_row)
        self._content_lay.addWidget(steps_container)

        instr = QLabel("Follow the steps in order. Each one builds on the previous.")
        instr.setStyleSheet("color:#333;font-size:9px;font-style:italic;")
        self._content_lay.addWidget(instr)

        self._step_widgets = []
        for i, (label, op, explanation) in enumerate(lesson["steps"]):
            sw = self._make_step(i, label, op, explanation, color)
            self._content_lay.addWidget(sw)
            self._step_widgets.append(sw)

        # ── 4. Insight ────────────────────────────────────────
        insight = QLabel(lesson["insight"])
        insight.setWordWrap(True)
        insight.setStyleSheet(
            f"background:#0a100a;color:#aaffcc;font-size:10px;"
            f"border-left:3px solid {color};border-radius:4px;padding:10px 12px;")
        self._content_lay.addWidget(insight)

        # ── 5. Checkpoint gate ────────────────────────────────
        if idx in self._completed:
            gate = QLabel("✓  Checkpoint cleared — journey continues")
            gate.setAlignment(Qt.AlignCenter)
            gate.setStyleSheet(
                "background:#001a0a;color:#00ff88;font-size:10px;font-weight:bold;"
                "border:1px solid #00ff8844;border-radius:6px;padding:10px;")
        else:
            gate = QLabel("Complete all steps above to clear this checkpoint →")
            gate.setAlignment(Qt.AlignCenter)
            gate.setStyleSheet(
                "background:#0a0a14;color:#2a2a44;font-size:10px;"
                "border:1px dashed #1a1a2a;border-radius:6px;padding:10px;")
        self._content_lay.addWidget(gate)
        self._content_lay.addStretch()

        # Update bottom bar
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < len(LESSONS) - 1)
        self._refresh_checkpoint_btn()
        self._content_scroll.verticalScrollBar().setValue(0)

    # ── Step card ─────────────────────────────────────────────
    def _make_step(self, i, label, op, explanation, color):
        done = i in self._step_done
        card = QWidget()
        card.setStyleSheet(
            f"background:{'#080d08' if done else '#0c0c18'};"
            f"border:1px solid {'#00ff8822' if done else '#1a1a2a'};"
            "border-radius:6px;")
        lay = QHBoxLayout(card); lay.setContentsMargins(10,8,10,8); lay.setSpacing(10)

        num = QLabel("✓" if done else str(i+1))
        num.setFixedSize(26, 26); num.setAlignment(Qt.AlignCenter)
        if done:
            num.setStyleSheet(
                "background:#003322;color:#00ff88;font-size:12px;"
                "font-weight:bold;border-radius:13px;")
        else:
            num.setStyleSheet(
                f"background:#1a1a2a;color:#555;font-size:10px;"
                "font-weight:bold;border-radius:13px;")

        txt_lay = QVBoxLayout(); txt_lay.setSpacing(2)
        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"color:{'#446644' if done else '#aaa'};"
            f"font-size:10px;font-weight:bold;"
            f"{'text-decoration:line-through;' if done else ''}")
        expl = QLabel(explanation)
        expl.setStyleSheet(f"color:{'#2a4a2a' if done else '#555'};font-size:9px;")
        txt_lay.addWidget(lbl); txt_lay.addWidget(expl)

        if done:
            btn = QLabel("✓ Done")
            btn.setFixedSize(64, 28); btn.setAlignment(Qt.AlignCenter)
            btn.setStyleSheet(
                "background:#003322;color:#00ff88;font-size:9px;"
                "border:1px solid #00ff8844;border-radius:5px;")
        else:
            btn = QPushButton("▶  Do it")
            btn.setFixedSize(64, 28)
            btn.setStyleSheet(
                f"background:{LESSONS[self._current]['dark']};"
                f"color:{color};font-size:9px;font-weight:bold;"
                f"border:1px solid {color}55;border-radius:5px;")
            btn.clicked.connect(lambda _, o=op, idx=i: self._do_step(idx, o))

        lay.addWidget(num); lay.addLayout(txt_lay, 1); lay.addWidget(btn)
        return card

    def _do_step(self, step_idx, op):
        self._step_done.add(step_idx)
        self.apply_op_requested.emit(op)

        lesson = LESSONS[self._current]
        color  = lesson["color"]

        for i, sw in enumerate(self._step_widgets):
            pass
        self._refresh_steps_area()

    def _refresh_steps_area(self):
        lesson = LESSONS[self._current]
        color  = lesson["color"]

        for sw in self._step_widgets:
            self._content_lay.removeWidget(sw)
            sw.deleteLater()

        self._step_widgets = []

        for i in range(self._content_lay.count()):
            item = self._content_lay.itemAt(i)
            if item and item.widget():
                w = item.widget()
                if hasattr(w, '_is_steps_marker'):
                    insert_at = i + 1
                    break
        else:
            insert_at = max(0, self._content_lay.count() - 3)

        for i, (label, op, explanation) in enumerate(lesson["steps"]):
            sw = self._make_step(i, label, op, explanation, color)
            self._content_lay.insertWidget(insert_at + i, sw)
            self._step_widgets.append(sw)
            
        n_done  = len(self._step_done)
        n_total = len(lesson["steps"])

        self._refresh_checkpoint_btn()

        if n_done >= n_total and self._current not in self._completed:
            self._show_all_done_pulse()

    def _show_all_done_pulse(self):
        color = LESSONS[self._current]["color"]
        self._checkpoint_btn.setText("⬡  CLEAR CHECKPOINT")
        self._checkpoint_btn.setStyleSheet(
            f"background:{LESSONS[self._current]['dark']};"
            f"color:{color};font-size:9px;font-weight:bold;"
            f"border:2px solid {color};border-radius:5px;padding:0 12px;")

    def _refresh_checkpoint_btn(self):
        lesson  = LESSONS[self._current]
        color   = lesson["color"]
        n_done  = len(self._step_done)
        n_total = len(lesson["steps"])
        all_done = n_done >= n_total
        cleared  = self._current in self._completed

        self._steps_lbl_update(n_done, n_total)

        if cleared:
            self._checkpoint_btn.setText("✓  CLEARED")
            self._checkpoint_btn.setStyleSheet(
                "background:#003322;color:#00ff88;font-size:9px;font-weight:bold;"
                "border:1px solid #00ff8844;border-radius:5px;padding:0 12px;")
        elif all_done:
            self._checkpoint_btn.setText("⬡  CLEAR CHECKPOINT")
            self._checkpoint_btn.setStyleSheet(
                f"background:{lesson['dark']};color:{color};"
                f"font-size:9px;font-weight:bold;"
                f"border:2px solid {color};border-radius:5px;padding:0 12px;")
        else:
            remaining = n_total - n_done
            self._checkpoint_btn.setText(f"○  {remaining} step{'s' if remaining>1 else ''} left")
            self._checkpoint_btn.setStyleSheet(
                "background:#0a0a12;color:#2a2a44;font-size:9px;font-weight:bold;"
                "border:1px solid #1a1a2a;border-radius:5px;padding:0 12px;")

    def _steps_lbl_update(self, done, total):
        n_cleared = len(self._completed)
        color = LESSONS[self._current]["color"]
        if n_cleared == 0:
            self._prog_lbl.setText("START YOUR JOURNEY")
            self._prog_lbl.setStyleSheet(
                "color:#1a1a3a;font-size:9px;font-weight:bold;letter-spacing:2px;")
        else:
            self._prog_lbl.setText(
                f"CHECKPOINT  {n_cleared} / {len(LESSONS)}  CLEARED")
            self._prog_lbl.setStyleSheet(
                "color:#00aa66;font-size:9px;font-weight:bold;letter-spacing:2px;")
        self._steps_lbl.setText(f"{done}/{total} steps")

    def _update_progress_bar(self):
        n = len(self._completed)
        total = len(LESSONS)
        if n == 0:
            self._prog_lbl.setText("START YOUR JOURNEY")
            self._prog_lbl.setStyleSheet(
                "color:#1a1a3a;font-size:9px;font-weight:bold;letter-spacing:2px;")
        elif n == total:
            self._prog_lbl.setText("✓  JOURNEY COMPLETE")
            self._prog_lbl.setStyleSheet(
                "color:#00ffcc;font-size:9px;font-weight:bold;letter-spacing:2px;")
        else:
            self._prog_lbl.setText(f"CHECKPOINT  {n} / {total}  CLEARED")
            self._prog_lbl.setStyleSheet(
                "color:#00aa66;font-size:9px;font-weight:bold;letter-spacing:2px;")

    def _section_hdr(self, text, color):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{color};font-size:10px;font-weight:bold;"
            "letter-spacing:1px;padding-top:4px;")
        return lbl