import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QComboBox,
    QSlider, QGroupBox, QFormLayout, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QGridLayout, QRadioButton, QSizePolicy,
    QDoubleSpinBox, QFileDialog, QMessageBox, 
    QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QBrush, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QTimer, QPoint, QRect
import datetime
import os

from CVPlaygroundProcessor import CVPlaygroundProcessor, CVPlaygroundPanel
from PipelineBuilderPanel import PipelineBuilderPanel
from ImageProcessor import ImageProcessor
from LearningModePanel import LearningModePanel

# ═══════════════════════════════════════════════════════════════
#  KERNELS
# ═══════════════════════════════════════════════════════════════
KERNELS = {
    "Gaussian Blur": {
        3: np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32)/16,
        5: np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],
                    [4,16,24,16,4],[1,4,6,4,1]], dtype=np.float32)/256,
    },
    "Median Filter": {
        3: np.ones((3,3),dtype=np.float32)/9,
        5: np.ones((5,5),dtype=np.float32)/25,
    },
    "Sharpen":   {"3x3": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)},
    "Emboss":    {"3x3": np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32)},
    "Laplacian": {
        "4-conn": np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32),
        "8-conn": np.array([[1,1,1],[1,-8,1],[1,1,1]], dtype=np.float32),
    },
    "Sobel Edges": {
        "Gx": np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32),
        "Gy": np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32),
    },
    "Canny Edges": {
        "Sobel Gx": np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32),
        "Sobel Gy": np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32),
    },
    "Low Pass Filter": {
        "Box 3x3": np.ones((3,3), dtype=np.float32)/9,
        "Box 5x5": np.ones((5,5), dtype=np.float32)/25,
    },
    "High Pass Filter": {
        "HPF": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32),
    },
    "SIFT Features": {
        "FAST ring": np.array([[0,1,1,1,0],[1,0,0,0,1],[1,0,1,0,1],
                                [1,0,0,0,1],[0,1,1,1,0]], dtype=np.float32)
    },
    "Erosion":  {"Struct 5x5": np.ones((5,5),dtype=np.float32)},
    "Dilation": {"Struct 5x5": np.ones((5,5),dtype=np.float32)},
    "Segmentation (Otsu)": {
        "Morph cross": np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.float32)
    },
    "ORB Features": {
        "FAST ring": np.array([[0,1,1,1,0],[1,0,0,0,1],[1,0,1,0,1],
                                [1,0,0,0,1],[0,1,1,1,0]], dtype=np.float32)
    },
}

FILTER_EXPLANATIONS = {
    "None": (
        "No operation applied.\n"
        "Frame passes through unchanged."
    ),
    "Gaussian Blur": (
        "📐 GAUSSIAN BLUR\n"
        "─────────────────────\n"
        "Blurs the image using a bell-curve weighted kernel.\n\n"
        "How it works:\n"
        "  1) Build an NxN kernel where center pixel has\n"
        "     highest weight, edges have lowest\n"
        "  2) Slide kernel over every pixel\n"
        "  3) Each output pixel = weighted average of neighbors\n\n"
        "Key rule: kernel values follow e^(-(x²+y²)/2σ²)\n"
        "Larger kernel OR larger σ → stronger blur\n\n"
        "✅ Best for: removing Gaussian noise before edge detection\n"
        "❌ Weakness: blurs edges along with noise"
    ),
    "Median Filter": (
        "📐 MEDIAN FILTER\n"
        "─────────────────────\n"
        "Replaces each pixel with the MEDIAN value\n"
        "of its NxN neighborhood.\n\n"
        "How it works:\n"
        "  1) Collect all pixel values in NxN window\n"
        "  2) Sort them: [12, 45, 67, 80, 95, 110, 200, 220, 255]\n"
        "  3) Pick the middle value (median) → output\n\n"
        "Key insight: extreme values (salt=255, pepper=0)\n"
        "get pushed to the ends of the sorted list\n"
        "and never become the median → noise eliminated!\n\n"
        "✅ Best for: salt & pepper noise removal\n"
        "✅ Preserves edges better than Gaussian\n"
        "❌ Slower than Gaussian for large kernels"
    ),
    "Bilateral Filter": (
        "📐 BILATERAL FILTER\n"
        "─────────────────────\n"
        "Smart blur that SKIPS pixels with different colors.\n\n"
        "How it works:\n"
        "  1) Spatial weight  → closer pixels = higher weight\n"
        "     (same as Gaussian)\n"
        "  2) Color weight    → similar color = higher weight\n"
        "     different color = near-zero weight\n"
        "  3) Final weight = spatial × color\n"
        "  4) Weighted average using combined weights\n\n"
        "Example: pixel on edge between dark/light region\n"
        "  → dark-side neighbors get high weight\n"
        "  → light-side neighbors get ~0 weight\n"
        "  → edge is preserved!\n\n"
        "Parameters:\n"
        "  Diameter    → neighborhood size\n"
        "  Sigma Color → how strict the color matching is\n\n"
        "✅ Best for: denoising while keeping sharp edges\n"
        "❌ Much slower than Gaussian"
    ),
    "Gaussian Noise": (
        "🎲 GAUSSIAN NOISE\n"
        "─────────────────────\n"
        "Adds random values drawn from a Normal distribution.\n\n"
        "σ = 10  → subtle noise\n"
        "σ = 50  → strong grainy effect\n\n"
        "✅ Use for: simulating real camera noise\n"
        "✅ Testing filter robustness"
    ),
    "Salt and Pepper": (
        "🎲 SALT & PEPPER NOISE\n"
        "─────────────────────\n"
        "Randomly forces pixels to pure black (0) or white (255).\n\n"
        "✅ Best removed by: Median Filter\n"
        "❌ Gaussian blur only spreads it around"
    ),
    "Canny Edges": (
        "📏 CANNY EDGE DETECTOR\n"
        "─────────────────────\n"
        "The gold standard multi-step edge detector.\n\n"
        "Steps: Gaussian blur → Sobel gradients →\n"
        "Non-max suppression → Double threshold → Hysteresis\n\n"
        "✅ Best edge detector for most use cases\n"
        "✅ Thin, clean, connected edge lines"
    ),
    "Sobel Edges": (
        "📏 SOBEL EDGE DETECTOR\n"
        "─────────────────────\n"
        "Detects edges by computing the image gradient.\n\n"
        "✅ Simple and fast\n"
        "✅ Gives edge direction info\n"
        "❌ Produces thicker edges than Canny"
    ),
    "Laplacian": (
        "📏 LAPLACIAN FILTER\n"
        "─────────────────────\n"
        "Second-order derivative edge detector.\n\n"
        "✅ Finds edges in ALL directions at once\n"
        "❌ Very sensitive to noise — blur first!"
    ),
    "Sharpen": (
        "✨ SHARPEN\n"
        "─────────────────────\n"
        "Makes edges crisper by boosting high-frequency detail.\n\n"
        "✅ Good for: text, fine details, product photos\n"
        "❌ Amplifies noise — apply denoising first"
    ),
    "Emboss": (
        "✨ EMBOSS\n"
        "─────────────────────\n"
        "Creates a 3D raised/carved look.\n\n"
        "✅ Use for: artistic effects, texture visualization"
    ),
    "Histogram Equalization": (
        "📊 HISTOGRAM EQUALIZATION\n"
        "─────────────────────\n"
        "Stretches the histogram to use the full 0–255 range.\n\n"
        "✅ Great for: dark/foggy images, medical imaging"
    ),
    "CLAHE": (
        "📊 CLAHE\n"
        "(Contrast Limited Adaptive Histogram Equalization)\n"
        "─────────────────────\n"
        "Like Histogram EQ but done locally in small tiles.\n\n"
        "✅ Best for: faces in harsh lighting\n"
        "✅ Medical imaging (X-ray, MRI)"
    ),
    "Gamma Correction": (
        "📊 GAMMA CORRECTION\n"
        "─────────────────────\n"
        "Power-law intensity transform.\n\n"
        "γ < 1 → brightens\n"
        "γ > 1 → darkens\n\n"
        "✅ Correcting under/over-exposed images"
    ),
    "Erosion": (
        "🔮 EROSION (Morphological)\n"
        "─────────────────────\n"
        "Shrinks bright/white regions.\n\n"
        "✅ Remove small white noise\n"
        "✅ Separate touching objects"
    ),
    "Dilation": (
        "🔮 DILATION (Morphological)\n"
        "─────────────────────\n"
        "Expands bright/white regions.\n\n"
        "✅ Fill small holes in objects\n"
        "✅ Connect nearby broken lines"
    ),
    "Segmentation (Otsu)": (
        "🎯 OTSU'S THRESHOLDING\n"
        "─────────────────────\n"
        "Automatically finds the BEST threshold.\n\n"
        "✅ No manual tuning needed\n"
        "❌ Fails with uneven lighting (use CLAHE first)"
    ),
    "ORB Features": (
        "🔑 ORB FEATURES\n"
        "(Oriented FAST + Rotated BRIEF)\n"
        "─────────────────────\n"
        "Fast, free alternative to SIFT.\n\n"
        "✅ Very fast — good for real-time applications\n"
        "✅ Free (SIFT/SURF are patented)"
    ),
    "SIFT Features": (
        "🔑 SIFT FEATURES\n"
        "(Scale-Invariant Feature Transform)\n"
        "─────────────────────\n"
        "Finds keypoints stable across scale, rotation, lighting.\n\n"
        "✅ Extremely robust to zoom, rotation, lighting\n"
        "❌ Slow — not ideal for real-time use"
    ),
    "Color Quantize": (
        "🎨 COLOR QUANTIZATION\n"
        "─────────────────────\n"
        "Reduces the image to only K distinct colors.\n\n"
        "✅ Image compression\n"
        "✅ Artistic posterization effect"
    ),
    "Low Pass Filter": (
        "〰️ LOW PASS FILTER\n"
        "─────────────────────\n"
        "Passes LOW frequency (smooth) content.\n\n"
        "✅ Noise reduction before processing"
    ),
    "High Pass Filter": (
        "〰️ HIGH PASS FILTER\n"
        "─────────────────────\n"
        "Passes HIGH frequency (edges, texture) content.\n\n"
        "✅ Sharpening, edge detection"
    ),
    "Add (+)": (
        "➕ ADD (Brightness Increase)\n"
        "─────────────────────\n"
        "Adds a constant value to every pixel.\n\n"
        "✅ Quick brightness adjustment"
    ),
    "Subtract (-)": (
        "➖ SUBTRACT (Brightness Decrease)\n"
        "─────────────────────\n"
        "Subtracts a constant from every pixel.\n\n"
        "✅ Darkening for artistic effect"
    ),
    "Multiply (×)": (
        "✖️ MULTIPLY (Contrast Scale)\n"
        "─────────────────────\n"
        "Multiplies every pixel by a factor.\n\n"
        "✅ Contrast enhancement"
    ),
    "Divide (÷)": (
        "➗ DIVIDE (Dynamic Range Compression)\n"
        "─────────────────────\n"
        "Divides every pixel by a factor.\n\n"
        "✅ Recover detail in overexposed images"
    ),
    "Custom Kernel": (
        "🧮 CUSTOM CONVOLUTION KERNEL\n"
        "─────────────────────\n"
        "Apply ANY kernel you define in the Custom K tab.\n\n"
        "✅ Full control over the convolution"
    ),
}

OPERATIONS_CONFIG = {
    "None": [],
    "Add (+)":      [{"name":"Value",           "min":0,   "max":255, "default":50}],
    "Subtract (-)": [{"name":"Value",           "min":0,   "max":255, "default":50}],
    "Multiply (×)": [{"name":"Factor (x10)",    "min":1,   "max":50,  "default":10}],
    "Divide (÷)":   [{"name":"Factor (x10)",    "min":1,   "max":50,  "default":10}],
    "Histogram Equalization": [],
    "CLAHE": [
        {"name":"Clip Limit (x10)","min":1,"max":100,"default":20},
        {"name":"Grid Size",       "min":2,"max":16, "default":8},
    ],
    "Gamma Correction":  [{"name":"Gamma (x10)",      "min":1,"max":50,  "default":10}],
    "Salt and Pepper":   [{"name":"Noise Prob (%)","min":1,"max":20,"default":5}],
    "Gaussian Noise":    [{"name":"Sigma",            "min":1,"max":100, "default":20}],
    "Median Filter":     [{"name":"Kernel Size (Odd)","min":1,"max":15,  "default":3}],
    "Gaussian Blur":     [{"name":"Kernel Size (Odd)","min":1,"max":15,  "default":5}],
    "Bilateral Filter":  [
        {"name":"Diameter",   "min":3, "max":15, "default":9},
        {"name":"Sigma Color","min":10,"max":150,"default":75},
    ],
    "Sharpen":   [{"name":"Strength (x10)","min":1,"max":30,"default":10}],
    "Emboss":    [],
    "Laplacian": [{"name":"Blend (%)","min":0,"max":100,"default":50}],
    "Sobel Edges":  [{"name":"Kernel Size","min":1,"max":7,"default":3}],
    "Canny Edges":  [
        {"name":"Min Threshold","min":0,"max":255,"default":100},
        {"name":"Max Threshold","min":0,"max":255,"default":200},
    ],
    "Low Pass Filter":  [{"name":"Kernel Size (Odd)","min":1,"max":21,"default":5}],
    "High Pass Filter": [{"name":"Strength (x10)","min":1,"max":30,"default":10}],
    "SIFT Features":    [{"name":"Max Features","min":50,"max":500,"default":200}],
    "Erosion":             [{"name":"Iterations", "min":1,"max":10,  "default":1}],
    "Dilation":            [{"name":"Iterations", "min":1,"max":10,  "default":1}],
    "ORB Features":        [{"name":"Max Features","min":50,"max":1000,"default":500}],
    "Segmentation (Otsu)": [],
    "Custom Kernel":       [],
}

# ═══════════════════════════════════════════════════════════════
#  VIDEO THREAD
# ═══════════════════════════════════════════════════════════════
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.mode1, self.params1 = "None", {}
        self.mode2, self.params2 = "None", {}
        self.custom_kernel1 = self.custom_kernel2 = None
        self.mutex = QMutex()
        self._fps  = 0.0

    @property
    def fps(self): return self._fps

    def update_settings(self, m1, p1, ck1=None):
        self.mutex.lock()
        self.mode1, self.params1 = m1, p1.copy()
        self.custom_kernel1 = ck1
        self.mutex.unlock()

    def run(self):
        cap = cv2.VideoCapture(0)
        t0 = time.time(); frames = 0
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                self.mutex.lock()
                m1, p1 = self.mode1, self.params1.copy()
                ck1 = self.custom_kernel1
                self.mutex.unlock()
                proc = ImageProcessor.apply(frame.copy(), m1, p1, ck1)
                self.change_pixmap_signal.emit(frame, proc, self._histogram(proc))
                frames += 1
                elapsed = time.time() - t0
                if elapsed >= 1.0:
                    self._fps = frames / elapsed
                    frames, t0 = 0, time.time()
        cap.release()

    def stop(self):
        self._run_flag = False; self.wait()

    def _histogram(self, img):
        out = np.zeros((160, 300, 3), dtype=np.uint8)
        for i, col in enumerate([(255,0,0),(0,255,0),(0,0,255)]):
            h = cv2.calcHist([img],[i],None,[256],[0,256])
            cv2.normalize(h, h, 0, 160, cv2.NORM_MINMAX); h = h.flatten()
            for x in range(1, 256):
                cv2.line(out,(x-1,160-int(h[x-1])),(x,160-int(h[x])),col,1)
        return out


# ═══════════════════════════════════════════════════════════════
#  MAGNIFIER LABEL  ← NEW
# ═══════════════════════════════════════════════════════════════
class MagnifierLabel(QLabel):
    """
    A QLabel that shows an image and draws a circular magnifier lens
    on mouse hover when enabled. The lens shows a zoomed-in patch
    from the source (original resolution) numpy array.
    """
    pixel_hovered  = pyqtSignal(int, int)
    pixel_clicked  = pyqtSignal(int, int)

    # Lens visual constants
    LENS_RADIUS   = 90          # pixels on screen
    LENS_ZOOM     = 3.0         # default zoom factor
    BORDER_COLOR  = (0, 255, 204)   # BGR cyan
    CROSS_COLOR   = (255, 255, 0)   # BGR yellow

    def __init__(self, parent=None, accent_color="#00ffcc"):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._fw, self._fh = 640, 480
        self._source_frame  = None   # numpy BGR – high-res source for the crop
        self._magnifier_on  = False
        self._mouse_pos     = QPoint(-999, -999)
        self._zoom          = self.LENS_ZOOM
        self._accent        = accent_color

    # ── public API ────────────────────────────────────────────
    def set_frame_size(self, w, h):
        self._fw, self._fh = w, h

    def set_source_frame(self, bgr_frame):
        """Supply the numpy frame used for the magnifier crop."""
        self._source_frame = bgr_frame

    def set_magnifier_enabled(self, enabled: bool):
        self._magnifier_on = enabled
        if not enabled:
            self.setCursor(Qt.ArrowCursor)
        else:
            self.setCursor(Qt.BlankCursor)
        self.update()

    def set_zoom(self, zoom: float):
        self._zoom = max(1.5, float(zoom))
        self.update()

    # ── coordinate helpers ────────────────────────────────────
    def _label_to_frame(self, px, py):
        lw, lh = self.width(), self.height()
        scale  = min(lw / self._fw, lh / self._fh)
        ox     = (lw - self._fw * scale) / 2
        oy     = (lh - self._fh * scale) / 2
        return int((px - ox) / scale), int((py - oy) / scale)

    def _frame_to_label(self, fx, fy):
        lw, lh = self.width(), self.height()
        scale  = min(lw / self._fw, lh / self._fh)
        ox     = (lw - self._fw * scale) / 2
        oy     = (lh - self._fh * scale) / 2
        return int(fx * scale + ox), int(fy * scale + oy)

    # ── Qt events ────────────────────────────────────────────
    def mouseMoveEvent(self, e):
        self._mouse_pos = e.pos()
        fx, fy = self._label_to_frame(e.x(), e.y())
        if 0 <= fx < self._fw and 0 <= fy < self._fh:
            self.pixel_hovered.emit(fx, fy)
        if self._magnifier_on:
            self.update()
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e):
        fx, fy = self._label_to_frame(e.x(), e.y())
        if 0 <= fx < self._fw and 0 <= fy < self._fh:
            self.pixel_clicked.emit(fx, fy)
        super().mousePressEvent(e)

    def leaveEvent(self, e):
        self._mouse_pos = QPoint(-999, -999)
        if self._magnifier_on:
            self.update()
        super().leaveEvent(e)

    # ── paint ─────────────────────────────────────────────────
    def paintEvent(self, e):
        super().paintEvent(e)          # draw the QLabel pixmap first

        if not self._magnifier_on:
            return
        if self._source_frame is None:
            return

        mx, my = self._mouse_pos.x(), self._mouse_pos.y()
        if mx < 0 or my < 0:
            return

        r      = self.LENS_RADIUS
        zoom   = self._zoom
        fx, fy = self._label_to_frame(mx, my)

        # ── compute crop region in frame coords ───────────────
        crop_half_w = int(r / zoom)
        crop_half_h = int(r / zoom)
        src_h, src_w = self._source_frame.shape[:2]

        x0 = max(0, fx - crop_half_w)
        y0 = max(0, fy - crop_half_h)
        x1 = min(src_w, fx + crop_half_w)
        y1 = min(src_h, fy + crop_half_h)

        crop = self._source_frame[y0:y1, x0:x1]
        if crop.size == 0:
            return

        # ── scale crop to lens diameter ───────────────────────
        lens_d = r * 2
        zoomed = cv2.resize(crop, (lens_d, lens_d), interpolation=cv2.INTER_LINEAR)
        zoomed_rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)

        # ── build circular QPixmap ────────────────────────────
        qi      = QImage(zoomed_rgb.data, lens_d, lens_d, 3 * lens_d, QImage.Format_RGB888).copy()
        lens_pm = QPixmap(lens_d, lens_d)
        lens_pm.fill(Qt.transparent)

        circle_painter = QPainter(lens_pm)
        circle_painter.setRenderHint(QPainter.Antialiasing)
        circle_painter.setClipRegion(
            __import__('PyQt5.QtGui', fromlist=['QRegion']).QRegion(0, 0, lens_d, lens_d, 
            __import__('PyQt5.QtGui', fromlist=['QRegion']).QRegion.Ellipse)
        )
        circle_painter.drawImage(0, 0, qi)
        circle_painter.end()

        # ── draw on label ─────────────────────────────────────
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Position lens: offset so it doesn't cover the cursor
        lx = mx - r
        ly = my - r

        # Clamp to widget bounds
        lx = max(0, min(lx, self.width()  - lens_d))
        ly = max(0, min(ly, self.height() - lens_d))

        # Draw lens background (pixmap already clipped to circle)
        painter.drawPixmap(lx, ly, lens_pm)

        # Outer border ring
        pen = QPen(QColor(self._accent))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawEllipse(lx, ly, lens_d - 1, lens_d - 1)

        # Inner thin ring
        pen2 = QPen(QColor(255, 255, 255, 80))
        pen2.setWidth(1)
        painter.setPen(pen2)
        painter.drawEllipse(lx + 4, ly + 4, lens_d - 9, lens_d - 9)

        # Crosshair at center
        cx, cy = lx + r, ly + r
        cross_color = QColor(255, 255, 0, 200)
        pen3 = QPen(cross_color)
        pen3.setWidth(1)
        painter.setPen(pen3)
        painter.drawLine(cx - 10, cy, cx + 10, cy)
        painter.drawLine(cx, cy - 10, cx, cy + 10)

        # Small dot at cursor (real position on label)
        pen4 = QPen(QColor(255, 100, 0))
        pen4.setWidth(2)
        painter.setPen(pen4)
        painter.drawEllipse(mx - 3, my - 3, 6, 6)

        # Zoom info label
        painter.setPen(QColor(self._accent))
        font = QFont("Consolas", 8)
        painter.setFont(font)
        painter.fillRect(lx, ly + lens_d + 2, lens_d, 16, QColor(0, 0, 0, 160))
        painter.drawText(lx, ly + lens_d + 14,
                         f"  {zoom:.1f}x  |  ({fx},{fy})")

        painter.end()


# ═══════════════════════════════════════════════════════════════
#  KERNEL VIEWER
# ═══════════════════════════════════════════════════════════════
class KernelViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4,4,4,4)
        self._layout.setSpacing(8)

    def set_mode(self, mode, params, custom_kernel=None):
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        if mode == "Custom Kernel":
            if custom_kernel is not None and custom_kernel.size > 0:
                self._layout.addWidget(self._matrix_widget("Custom Kernel", custom_kernel))
            else:
                lbl = QLabel("No kernel yet.\nGo to  🧮 Custom K  tab.")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("color:#555;font-style:italic;padding:20px;")
                self._layout.addWidget(lbl)
            self._layout.addStretch(); return
        data = KERNELS.get(mode)
        if not data:
            lbl = QLabel("No kernel matrix for this operation")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:#555;font-style:italic;padding:20px;")
            self._layout.addWidget(lbl); self._layout.addStretch(); return
        if mode == "Gaussian Blur":
            k = params.get("Kernel Size (Odd)", 5); k = k if k % 2 else k + 1
            size = k if k in data else (3 if k <= 3 else 5)
            mats = {f"Gaussian {size}x{size}": data.get(size, data[3])}
        elif mode == "Median Filter":
            k = params.get("Kernel Size (Odd)", 3); k = k if k % 2 else k + 1
            size = k if k in data else (3 if k <= 3 else 5)
            mats = {f"Uniform box {size}x{size}": data.get(size, data[3])}
        else:
            mats = {f"{k}": v for k, v in data.items()}
        for title, mat in mats.items():
            self._layout.addWidget(self._matrix_widget(title, mat))
        self._layout.addStretch()

    def _matrix_widget(self, title, matrix):
        group = QGroupBox(title)
        group.setStyleSheet(
            "QGroupBox{color:#00ffcc;font-weight:bold;border:1px solid #333;"
            "margin-top:8px;padding:6px;border-radius:4px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}")
        rows, cols = matrix.shape
        info = QLabel(
            f"Size:{rows}x{cols}  Sum:{matrix.sum():.4f}  "
            f"Mean:{matrix.mean():.4f}  Max|v|:{np.abs(matrix).max():.3f}")
        info.setStyleSheet("color:#aaa;font-size:10px;margin-bottom:4px;")
        grid = QGridLayout(); grid.setSpacing(3)
        max_abs = np.abs(matrix).max() or 1.0
        for r in range(rows):
            for c in range(cols):
                val = float(matrix[r, c])
                text = f"{val:.3f}" if matrix.dtype in (np.float32, np.float64) else str(int(val))
                cell = QLabel(text); cell.setAlignment(Qt.AlignCenter)
                t = int(180 * abs(val) / max_abs)
                if val > 0:   bg = f"#1a{t:02x}{min(t+80,255):02x}"
                elif val < 0: bg = f"#{min(t+60,255):02x}{t:02x}{t:02x}"
                else:         bg = "#1a1a1a"
                cell.setStyleSheet(
                    f"background:{bg};color:white;border:1px solid #444;"
                    f"padding:4px;font-size:10px;border-radius:3px;")
                cell.setMinimumSize(48, 30); grid.addWidget(cell, r, c)
        vbox = QVBoxLayout(); vbox.addWidget(info); vbox.addLayout(grid)
        group.setLayout(vbox); return group


# ═══════════════════════════════════════════════════════════════
#  CUSTOM KERNEL PANEL
# ═══════════════════════════════════════════════════════════════
class CustomKernelPanel(QWidget):
    kernel_changed = pyqtSignal(np.ndarray)
    _PRESETS = {
        "Identity":   (3, [[0,0,0],[0,1,0],[0,0,0]]),
        "Sharpen":    (3, [[0,-1,0],[-1,5,-1],[0,-1,0]]),
        "Blur (box)": (3, [[1,1,1],[1,1,1],[1,1,1]]),
        "Edge detect":(3, [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
        "Emboss":     (3, [[-2,-1,0],[-1,1,1],[0,1,2]]),
        "Laplacian":  (3, [[0,1,0],[1,-4,1],[0,1,0]]),
        "Sobel X":    (3, [[-1,0,1],[-2,0,2],[-1,0,1]]),
        "Sobel Y":    (3, [[-1,-2,-1],[0,0,0],[1,2,1]]),
        "Box 5x5":    (5, [[1]*5]*5),
    }
    def __init__(self, parent=None):
        super().__init__(parent)
        self._size = 3; self._cells = []; self._building = False
        main = QVBoxLayout(self)
        main.setContentsMargins(6,6,6,6); main.setSpacing(8)
        title = QLabel("🧮  Custom Kernel")
        title.setStyleSheet("color:#ff9944;font-size:12px;font-weight:bold;")
        main.addWidget(title)
        r1 = QHBoxLayout(); r1.addWidget(QLabel("Size:"))
        self.size_combo = QComboBox(); self.size_combo.addItems(["3x3","5x5","7x7"])
        self.size_combo.setFixedWidth(70)
        self.size_combo.currentIndexChanged.connect(self._on_size_changed)
        r1.addWidget(self.size_combo); r1.addSpacing(16); r1.addWidget(QLabel("Normalize:"))
        self.btn_norm = QPushButton("OFF"); self.btn_norm.setCheckable(True); self.btn_norm.setFixedWidth(46)
        self.btn_norm.setStyleSheet("background:#1a1a1a;color:#666;font-size:10px;border:1px solid #333;border-radius:3px;padding:3px;")
        self.btn_norm.toggled.connect(self._on_norm_toggled)
        r1.addWidget(self.btn_norm); r1.addStretch()
        main.addLayout(r1)
        r2 = QHBoxLayout(); r2.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox(); self.preset_combo.addItem("— choose —")
        self.preset_combo.addItems(list(self._PRESETS.keys())); self.preset_combo.setFixedWidth(150)
        self.preset_combo.currentTextChanged.connect(self._load_preset)
        r2.addWidget(self.preset_combo); r2.addStretch()
        main.addLayout(r2)
        self.grid_widget = QWidget()
        self.grid_widget.setStyleSheet("background:#111;border:1px solid #333;border-radius:4px;")
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(4); self.grid_layout.setContentsMargins(8,8,8,8)
        main.addWidget(self.grid_widget)
        self.info_lbl = QLabel("Sum: 1.000")
        self.info_lbl.setStyleSheet("color:#666;font-size:10px;")
        main.addWidget(self.info_lbl)
        br = QHBoxLayout()
        self.btn_apply = QPushButton("▶  Apply")
        self.btn_apply.setStyleSheet("background:#004488;color:#00ccff;font-size:11px;border:1px solid #0066aa;border-radius:4px;padding:6px;")
        self.btn_apply.clicked.connect(self._emit_kernel)
        self.btn_reset = QPushButton("⟳  Reset")
        self.btn_reset.setStyleSheet("background:#2a1a00;color:#aa6600;font-size:11px;border:1px solid #553300;border-radius:4px;padding:6px;")
        self.btn_reset.clicked.connect(self._reset_identity)
        br.addWidget(self.btn_apply); br.addWidget(self.btn_reset)
        main.addLayout(br); main.addStretch()
        self._build_grid(3)

    def _on_size_changed(self, idx): self._build_grid([3,5,7][idx])
    def _on_norm_toggled(self, ch):
        self.btn_norm.setText("ON" if ch else "OFF")
        self.btn_norm.setStyleSheet(
            f"background:{'#003300' if ch else '#1a1a1a'};color:{'#00ff88' if ch else '#666'};"
            "font-size:10px;border:1px solid #333;border-radius:3px;padding:3px;")
        self._emit_kernel()

    def _build_grid(self, size):
        self._building = True; self._size = size; self._cells.clear()
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        mid = size // 2
        for r in range(size):
            for c in range(size):
                sb = QDoubleSpinBox(); sb.setRange(-999.0, 999.0)
                sb.setDecimals(2); sb.setSingleStep(0.1)
                sb.setValue(1.0 if (r == mid and c == mid) else 0.0)
                sb.setFixedSize(62, 28)
                sb.setStyleSheet(
                    "QDoubleSpinBox{background:#1a1a1a;color:#00ffcc;border:1px solid #333;"
                    "border-radius:3px;font-size:11px;}"
                    "QDoubleSpinBox:focus{border:1px solid #00ffcc;}")
                sb.valueChanged.connect(self._on_cell_changed)
                self.grid_layout.addWidget(sb, r, c)
                self._cells.append(sb)
        self._building = False; self._update_info()

    def _on_cell_changed(self, _):
        if not self._building: self._update_info()

    def _update_info(self):
        k = self._get_raw(); s = k.sum(); mx = np.abs(k).max()
        hint = f"  (will div {s:.3f})" if self.btn_norm.isChecked() and s != 0 else ""
        self.info_lbl.setText(f"Sum:{s:.3f}   Max|v|:{mx:.3f}{hint}")

    def _get_raw(self):
        k = np.zeros((self._size, self._size), dtype=np.float32)
        for i, sb in enumerate(self._cells):
            k[i // self._size, i % self._size] = sb.value()
        return k

    def get_kernel(self):
        k = self._get_raw()
        if self.btn_norm.isChecked():
            s = k.sum()
            if s != 0: k = k / s
        return k

    def _emit_kernel(self): self.kernel_changed.emit(self.get_kernel())
    def _reset_identity(self):
        self._building = True; mid = self._size // 2
        for i, sb in enumerate(self._cells):
            sb.setValue(1.0 if (i // self._size == mid and i % self._size == mid) else 0.0)
        self._building = False; self._update_info(); self._emit_kernel()

    def _load_preset(self, name):
        if name not in self._PRESETS: return
        size, values = self._PRESETS[name]
        self.size_combo.blockSignals(True)
        self.size_combo.setCurrentIndex([3,5,7].index(size))
        self.size_combo.blockSignals(False)
        self._build_grid(size); self._building = True
        for i, sb in enumerate(self._cells):
            sb.setValue(float(values[i // size][i % size]))
        self._building = False; self._update_info(); self._emit_kernel()
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)


# ═══════════════════════════════════════════════════════════════
#  STATS VIEWER
# ═══════════════════════════════════════════════════════════════
class StatsViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4,4,4,4); layout.setSpacing(6)
        hdr = QLabel("Per-Channel Statistics (processed frame):")
        hdr.setStyleSheet("color:#00ffcc;font-size:11px;font-weight:bold;")
        layout.addWidget(hdr)
        self.table = QTableWidget(4, 5)
        self.table.setHorizontalHeaderLabels(["Channel","Mean","Std","Min","Max"])
        self.table.setVerticalHeaderLabels(["","","",""])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet(
            "QTableWidget{background:#111;color:white;gridline-color:#333;font-size:11px;}"
            "QHeaderView::section{background:#222;color:#00ffcc;border:1px solid #444;font-size:11px;}")
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setFixedHeight(130); layout.addWidget(self.table)
        bars_g = QGroupBox("Distribution Bars")
        bars_g.setStyleSheet("QGroupBox{color:#00ffcc;border:1px solid #333;margin-top:6px;padding:4px;}")
        bars_lay = QGridLayout(); self.bar_labels = {}
        for i, (name, color) in enumerate(zip(
                ["Blue","Green","Red","Gray"],
                ["#6688ff","#44cc66","#ff6655","#aaaaaa"])):
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color:{color};font-size:10px;"); lbl.setFixedWidth(38)
            bar = QLabel(); bar.setFixedHeight(16)
            bar.setStyleSheet("background:#111;border:1px solid #333;border-radius:2px;")
            val = QLabel("--")
            val.setStyleSheet(f"color:{color};font-size:10px;"); val.setFixedWidth(42)
            bars_lay.addWidget(lbl, i, 0); bars_lay.addWidget(bar, i, 1); bars_lay.addWidget(val, i, 2)
            self.bar_labels[name] = (bar, val, color)
        bars_g.setLayout(bars_lay); layout.addWidget(bars_g)
        self.extra = QLabel()
        self.extra.setStyleSheet(
            "color:#aaa;font-size:10px;border:1px solid #333;padding:6px;border-radius:4px;")
        self.extra.setWordWrap(True); layout.addWidget(self.extra); layout.addStretch()

    def update_frame(self, bgr):
        ch_data = [bgr[:,:,0], bgr[:,:,1], bgr[:,:,2],
                   cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)]
        colors = ["#6688ff","#44cc66","#ff6655","#aaaaaa"]
        for i, (ch, name) in enumerate(zip(ch_data, ["Blue","Green","Red","Gray"])):
            mean = ch.mean(); std = ch.std(); mn = int(ch.min()); mx = int(ch.max())
            for j, v in enumerate([name, f"{mean:.2f}", f"{std:.2f}", str(mn), str(mx)]):
                item = QTableWidgetItem(v); item.setTextAlignment(Qt.AlignCenter)
                if j == 0: item.setForeground(QColor(colors[i]))
                self.table.setItem(i, j, item)
            bar, val_lbl, color = self.bar_labels[name]; pct = int(mean / 255 * 100)
            bar.setStyleSheet(
                f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                f"stop:0 {color},stop:{pct/100:.2f} {color},"
                f"stop:{min(pct/100+0.01,1):.2f} #111,stop:1 #111);"
                "border:1px solid #333;border-radius:2px;")
            val_lbl.setText(f"{mean:.1f}")
        gray = ch_data[3].astype(np.float32); h, w = bgr.shape[:2]
        rms = float(np.sqrt(np.mean(gray**2)))
        snr = 20 * np.log10(gray.mean() / (gray.std() + 1e-6))
        self.extra.setText(
            f"Resolution:{w}x{h}  |  Pixels:{w*h:,}  |  Frame:{bgr.nbytes/1024:.1f}KB\n"
            f"RMS Lum:{rms:.1f}  |  Contrast:{gray.std():.1f}  |  SNR:{snr:.1f}dB")


# ═══════════════════════════════════════════════════════════════
#  CHANNELS VIEWER
# ═══════════════════════════════════════════════════════════════
class ChannelsViewer(QWidget):
    TW, TH = 185, 118
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4,4,4,4); layout.setSpacing(6)
        ctrl = QHBoxLayout(); ctrl.addWidget(QLabel("Color space:"))
        self.r_bgr = QRadioButton("BGR"); self.r_bgr.setChecked(True)
        self.r_hsv = QRadioButton("HSV"); self.r_lab = QRadioButton("LAB")
        for rb in (self.r_bgr, self.r_hsv, self.r_lab):
            rb.setStyleSheet("color:white;"); rb.toggled.connect(self._refresh)
            ctrl.addWidget(rb)
        ctrl.addStretch(); layout.addLayout(ctrl)
        grid = QGridLayout(); grid.setSpacing(6)
        self.mini_labels = []; self.mini_titles = []
        for idx, (r, c) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            title = QLabel("--"); title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("color:#00ffcc;font-size:10px;font-weight:bold;")
            lbl = QLabel(); lbl.setFixedSize(self.TW, self.TH)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("border:1px solid #444;background:#000;border-radius:3px;")
            grid.addWidget(title, r*2, c); grid.addWidget(lbl, r*2+1, c)
            self.mini_titles.append(title); self.mini_labels.append(lbl)
        self.mini_titles[3].setStyleSheet("color:#ffcc00;font-size:10px;font-weight:bold;")
        self.mini_labels[3].setStyleSheet("border:1px solid #ffcc00;background:#000;border-radius:3px;")
        layout.addLayout(grid); layout.addStretch(); self._frame = None

    def update_frame(self, bgr): self._frame = bgr; self._refresh()

    def _refresh(self):
        if self._frame is None: return
        bgr = self._frame
        if self.r_hsv.isChecked():
            conv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            names = ["H (Hue)","S (Sat)","V (Val)"]
            cmaps = [cv2.COLORMAP_HSV, cv2.COLORMAP_BONE, cv2.COLORMAP_MAGMA]
        elif self.r_lab.isChecked():
            conv = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            names = ["L* (Light)","a* (Gr-Rd)","b* (Bl-Ye)"]
            cmaps = [cv2.COLORMAP_BONE, cv2.COLORMAP_COOL, cv2.COLORMAP_OCEAN]
        else:
            conv = bgr; names = ["B (Blue)","G (Green)","R (Red)"]
            cmaps = [None, None, None]
        for i in range(3):
            ch = conv[:,:,i]
            if cmaps[i]: vis = cv2.applyColorMap(ch, cmaps[i])
            else:
                z = np.zeros_like(ch)
                vis = cv2.merge([ch,z,z] if i==0 else ([z,ch,z] if i==1 else [z,z,ch]))
            self._put(self.mini_labels[i], vis); self.mini_titles[i].setText(names[i])
        self._put(self.mini_labels[3], bgr); self.mini_titles[3].setText("Combined")

    def _put(self, lbl, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
        qi = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        lbl.setPixmap(
            QPixmap.fromImage(qi).scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ═══════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════
class CVLab(QMainWindow):
    # ── image display size (easy to change) ──────────────────
    VIEW_W = 585   # ← bigger than original 570
    VIEW_H = 480   # ← bigger than original 370

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Computer Vision Lab 2026")
        self.setGeometry(40, 30, 1920, 1050)
        self.thread              = None
        self._pixel_size_orig    = 1
        self._show_values_orig   = False
        self._pixel_size         = 1
        self._show_values        = False
        self.sliders1            = {}
        self._last_orig          = None
        self._last_proc          = None
        self._last_display_proc  = None
        self.fps_timer           = None
        self._custom_kernel      = None
        self._last_save_dir      = os.path.expanduser("~")
        self._playground_proc    = CVPlaygroundProcessor()
        self._pipeline_steps     = []
        self._perf_t0            = time.time()
        self._perf_frame_count   = 0
        self.init_ui()

    # ──────────────────── BUILD UI ────────────────────────────
    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#1a1a1a;color:white;}
            QGroupBox{color:#00ffcc;font-weight:bold;border:1px solid #333;
                      margin-top:10px;border-radius:4px;}
            QGroupBox::title{subcontrol-origin:margin;left:8px;}
            QLabel{color:white;}
            QPushButton{padding:7px;font-weight:bold;border-radius:5px;}
            QComboBox{background:#2a2a2a;color:white;padding:5px;
                      border:1px solid #444;border-radius:3px;}
            QComboBox QAbstractItemView{background:#2a2a2a;color:white;}
            QTabWidget::pane{border:1px solid #333;}
            QTabBar::tab{background:#222;color:#777;padding:5px 11px;font-size:11px;}
            QTabBar::tab:selected{background:#2a2a2a;color:#00ffcc;
                                  border-bottom:2px solid #00ffcc;}
            QSlider::groove:horizontal{border:1px solid #555;height:8px;
                                       background:#2a2a2a;border-radius:4px;}
            QSlider::handle:horizontal{background:#00ffcc;width:16px;
                                       margin:-4px 0;border-radius:8px;}
            QScrollArea{border:none;}
            QSpinBox,QDoubleSpinBox{background:#2a2a2a;color:white;
                                    border:1px solid #444;padding:2px;border-radius:3px;}
            QRadioButton,QCheckBox{color:white;}
        """)

        root = QWidget(); h_main = QHBoxLayout(root)
        h_main.setSpacing(8); h_main.setContentsMargins(8,8,8,8)

        # ═══════════════════ SIDEBAR ══════════════════════════
        sidebar = QWidget(); sidebar.setFixedWidth(300)
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(4,4,4,4); sb.setSpacing(6)

        g1 = QGroupBox("① Primary Filter"); v1 = QVBoxLayout()
        self.op1 = QComboBox()
        self.op1.addItems(list(OPERATIONS_CONFIG.keys()))
        self.op1.currentTextChanged.connect(
            lambda m: (self._build_params(m, self.form1, self.sliders1, primary=True),
                       self._update_filter_explain(m)))
        v1.addWidget(self.op1); self.form1 = QFormLayout()
        v1.addLayout(self.form1); g1.setLayout(v1)

        # Filter explanation box
        self.filter_explain_box = QGroupBox("💡  How This Filter Works")
        self.filter_explain_box.setStyleSheet(
            "QGroupBox{color:#ffcc44;font-weight:bold;border:1px solid #554400;"
            "margin-top:10px;border-radius:4px;background:#0d0b00;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}")
        self.filter_explain_box.setFixedHeight(280)
        self.filter_explain_lbl = QLabel("← Select a filter to see how it works")
        self.filter_explain_lbl.setStyleSheet("color:#888;font-size:11px;padding:4px;")
        self.filter_explain_lbl.setWordWrap(True)
        self.filter_explain_lbl.setAlignment(Qt.AlignTop)
        explain_scroll = QScrollArea()
        explain_scroll.setWidget(self.filter_explain_lbl)
        explain_scroll.setWidgetResizable(True)
        explain_scroll.setStyleSheet(
            "QScrollArea{border:none;background:#0d0b00;}"
            "QScrollBar:vertical{background:#1a1200;width:5px;border-radius:2px;}"
            "QScrollBar::handle:vertical{background:#554400;border-radius:2px;}")
        explain_lay = QVBoxLayout()
        explain_lay.setContentsMargins(4,4,4,4)
        explain_lay.addWidget(explain_scroll)
        self.filter_explain_box.setLayout(explain_lay)

        h_hdr = QLabel("HISTOGRAM (processed):")
        h_hdr.setStyleSheet("color:#00ffcc;font-size:12px;margin-top:4px;")
        self.hist_lbl = QLabel(); self.hist_lbl.setFixedSize(290, 130)
        self.hist_lbl.setStyleSheet("background:#000;border:1px solid #333;border-radius:3px;")

        self.info_lbl = QLabel("FPS: --  |  Res: --")
        self.info_lbl.setStyleSheet("color:#666;font-size:12px;")

        btn_row = QHBoxLayout()
        self.btn_cam = QPushButton("▶  START")
        self.btn_cam.setStyleSheet("background:#0078d7;font-size:13px;")
        self.btn_cam.clicked.connect(self.toggle_cam)
        self.btn_load_img = QPushButton("🖼  Load Image")
        self.btn_load_img.setStyleSheet(
            "background:#1a1a2e;color:#8888ff;font-size:11px;"
            "border:1px solid #334;border-radius:5px;padding:6px;")
        self.btn_load_img.clicked.connect(self.load_image)
        btn_row.addWidget(self.btn_cam, 1)

        # Save group
        save_group = QGroupBox("💾  Save Frame")
        save_group.setStyleSheet(
            "QGroupBox{color:#44ff88;font-weight:bold;border:1px solid #1a4d2e;"
            "margin-top:10px;border-radius:4px;background:#0d1f15;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}")
        sg = QVBoxLayout(); sg.setContentsMargins(8,6,8,8); sg.setSpacing(5)
        fmt_row = QHBoxLayout()
        fl = QLabel("Format:"); fl.setStyleSheet("color:#aaa;font-size:10px;"); fl.setFixedWidth(52)
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["PNG","JPEG","BMP","TIFF"])
        self.fmt_combo.setFixedWidth(80)
        self.fmt_combo.setStyleSheet("background:#1a2e20;color:#88ff99;font-size:11px;")
        self.jpeg_q_lbl = QLabel("Q:")
        self.jpeg_q_lbl.setStyleSheet("color:#aaa;font-size:10px;"); self.jpeg_q_lbl.setFixedWidth(18)
        self.jpeg_q_slider = QSlider(Qt.Horizontal)
        self.jpeg_q_slider.setRange(10, 100); self.jpeg_q_slider.setValue(95); self.jpeg_q_slider.setFixedWidth(70)
        self.jpeg_q_val_lbl = QLabel("95")
        self.jpeg_q_val_lbl.setFixedWidth(26)
        self.jpeg_q_val_lbl.setStyleSheet("color:#44ff88;font-size:10px;")
        self.jpeg_q_slider.valueChanged.connect(lambda v: self.jpeg_q_val_lbl.setText(str(v)))
        self.fmt_combo.currentTextChanged.connect(self._on_fmt_changed)
        for w in (self.jpeg_q_lbl, self.jpeg_q_slider, self.jpeg_q_val_lbl):
            w.setVisible(False)
        fmt_row.addWidget(fl); fmt_row.addWidget(self.fmt_combo)
        fmt_row.addWidget(self.jpeg_q_lbl); fmt_row.addWidget(self.jpeg_q_slider)
        fmt_row.addWidget(self.jpeg_q_val_lbl); fmt_row.addStretch()
        sg.addLayout(fmt_row)
        sbr = QHBoxLayout()
        self.btn_save_proc = QPushButton("💾  Save Processed")
        self.btn_save_proc.setStyleSheet(
            "background:#1a4d2e;color:#44ff88;font-size:11px;"
            "border:1px solid #2a7a46;border-radius:5px;padding:6px;")
        self.btn_save_proc.clicked.connect(self.save_processed)
        self.btn_save_orig = QPushButton("📷  Save Original")
        self.btn_save_orig.setStyleSheet(
            "background:#1a1a3a;color:#8888ff;font-size:11px;"
            "border:1px solid #333388;border-radius:5px;padding:6px;")
        self.btn_save_orig.clicked.connect(self.save_original)
        sbr.addWidget(self.btn_save_proc); sbr.addWidget(self.btn_save_orig)
        sg.addLayout(sbr)
        self.save_status_lbl = QLabel("  No file saved yet")
        self.save_status_lbl.setStyleSheet("color:#555;font-size:10px;font-style:italic;")
        sg.addWidget(self.save_status_lbl); save_group.setLayout(sg)

        btn_about = QPushButton("ℹ️  About CV Lab Pro")
        btn_about.setStyleSheet(
            "background:#0a0a16;color:#444;font-size:10px;"
            "border:1px solid #1a1a2a;border-radius:4px;padding:5px;")
        btn_about.clicked.connect(self._show_about)

        sb.addWidget(btn_about)
        sb.addLayout(btn_row); sb.addWidget(self.btn_load_img)
        sb.addWidget(g1); sb.addWidget(self.filter_explain_box); sb.addStretch()
        sb.addWidget(save_group); sb.addWidget(h_hdr)
        sb.addWidget(self.hist_lbl); sb.addWidget(self.info_lbl)

        # ═══════════════════ CENTER ═══════════════════════════
        center = QWidget(); c_lay = QVBoxLayout(center)
        c_lay.setContentsMargins(0,0,0,0); c_lay.setSpacing(4)

        cam_row = QHBoxLayout(); cam_row.setSpacing(8)

        # ─── Original view ────────────────────────────────────
        ov = QVBoxLayout()
        tl = QLabel("🎥  Original Feed")
        tl.setAlignment(Qt.AlignCenter)
        tl.setStyleSheet("color:#777;font-size:11px;margin-bottom:2px;")

        self.view_orig = MagnifierLabel(accent_color="#777777")
        self.view_orig.setAlignment(Qt.AlignCenter)
        self.view_orig.setFixedSize(self.VIEW_W, self.VIEW_H)
        self.view_orig.setStyleSheet("border:2px dashed #444;border-radius:6px;background:#0a0a0a;")
        ov.addWidget(tl); ov.addWidget(self.view_orig)

        # ─── Processed view ───────────────────────────────────
        pv = QVBoxLayout()
        proc_hdr = QHBoxLayout()
        pl = QLabel("✨  Processed  —  hover to inspect pixel values")
        pl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        pl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        pl.setStyleSheet("color:#00ffcc;font-size:11px;margin-bottom:2px;")

        self.view_proc = MagnifierLabel(accent_color="#00ffcc")
        self.view_proc.setAlignment(Qt.AlignCenter)
        self.view_proc.setFixedSize(self.VIEW_W, self.VIEW_H)
        self.view_proc.setStyleSheet("border:2px dashed #00ffcc44;border-radius:6px;background:#0a0a0a;")
        self.view_proc.pixel_hovered.connect(self._on_hover)
        proc_hdr.addWidget(pl); proc_hdr.addStretch()
        pv.addLayout(proc_hdr); pv.addWidget(self.view_proc)

        cam_row.addLayout(ov); cam_row.addLayout(pv)
        c_lay.addLayout(cam_row)

        self.hover_bar = QLabel("  Hover over processed image to inspect a pixel")
        self.hover_bar.setFixedHeight(26)
        self.hover_bar.setStyleSheet(
            "background:#111;color:#666;font-size:11px;"
            "border:1px solid #333;padding:2px 8px;border-radius:3px;")
        c_lay.addWidget(self.hover_bar)

        # ─── MAGNIFIER CONTROLS  (NEW) ───────────────────────
        mag_group = QGroupBox("🔍  Magnifier Lens")
        mag_group.setStyleSheet(
            "QGroupBox{color:#00ccff;font-weight:bold;border:1px solid #003344;"
            "margin-top:6px;border-radius:4px;background:#060d10;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}")
        mag_group.setFixedHeight(110)
        mag_lay = QVBoxLayout(); mag_lay.setContentsMargins(10,4,10,6); mag_lay.setSpacing(6)

        mag_hint = QLabel("Move mouse over an image — a circular lens zooms in on that region")
        mag_hint.setStyleSheet("color:#557788;font-size:10px;")
        mag_lay.addWidget(mag_hint)

        mag_ctrl = QHBoxLayout(); mag_ctrl.setSpacing(16)

        # Enable buttons for each view
        self.btn_mag_orig = QPushButton("🔍 Lens: Original  OFF")
        self.btn_mag_orig.setCheckable(True); self.btn_mag_orig.setChecked(False)
        self.btn_mag_orig.setFixedWidth(180)
        self.btn_mag_orig.setStyleSheet(self._mag_btn_style(False))
        self.btn_mag_orig.toggled.connect(self._on_mag_orig_toggled)

        self.btn_mag_proc = QPushButton("🔍 Lens: Processed  OFF")
        self.btn_mag_proc.setCheckable(True); self.btn_mag_proc.setChecked(False)
        self.btn_mag_proc.setFixedWidth(190)
        self.btn_mag_proc.setStyleSheet(self._mag_btn_style(False))
        self.btn_mag_proc.toggled.connect(self._on_mag_proc_toggled)

        # Zoom slider
        zoom_lbl = QLabel("Zoom:")
        zoom_lbl.setStyleSheet("color:#557788;font-size:10px;"); zoom_lbl.setFixedWidth(38)

        self.mag_zoom_slider = QSlider(Qt.Horizontal)
        self.mag_zoom_slider.setRange(15, 60); self.mag_zoom_slider.setValue(30)  # ×10
        self.mag_zoom_slider.setFixedWidth(120)
        self.mag_zoom_slider.setStyleSheet(
            "QSlider::groove:horizontal{border:1px solid #003344;height:6px;"
            "background:#060d10;border-radius:3px;}"
            "QSlider::handle:horizontal{background:#00ccff;width:14px;"
            "margin:-4px 0;border-radius:7px;}"
            "QSlider::sub-page:horizontal{background:#006688;border-radius:3px;}")
        self.mag_zoom_val = QLabel("3.0×")
        self.mag_zoom_val.setStyleSheet("color:#00ccff;font-size:11px;font-weight:bold;"); self.mag_zoom_val.setFixedWidth(38)

        def _on_zoom_changed(v):
            z = v / 10.0
            self.mag_zoom_val.setText(f"{z:.1f}×")
            self.view_orig.set_zoom(z)
            self.view_proc.set_zoom(z)

        self.mag_zoom_slider.valueChanged.connect(_on_zoom_changed)

        # Lens size slider
        size_lbl = QLabel("Size:")
        size_lbl.setStyleSheet("color:#557788;font-size:10px;"); size_lbl.setFixedWidth(32)
        self.mag_size_slider = QSlider(Qt.Horizontal)
        self.mag_size_slider.setRange(40, 160); self.mag_size_slider.setValue(90)
        self.mag_size_slider.setFixedWidth(100)
        self.mag_size_slider.setStyleSheet(
            "QSlider::groove:horizontal{border:1px solid #003344;height:6px;"
            "background:#060d10;border-radius:3px;}"
            "QSlider::handle:horizontal{background:#44aaff;width:14px;"
            "margin:-4px 0;border-radius:7px;}"
            "QSlider::sub-page:horizontal{background:#224466;border-radius:3px;}")
        self.mag_size_val = QLabel("90px")
        self.mag_size_val.setStyleSheet("color:#44aaff;font-size:11px;font-weight:bold;"); self.mag_size_val.setFixedWidth(38)

        def _on_size_changed(v):
            self.mag_size_val.setText(f"{v}px")
            MagnifierLabel.LENS_RADIUS = v
        self.mag_size_slider.valueChanged.connect(_on_size_changed)

        mag_ctrl.addWidget(self.btn_mag_orig)
        mag_ctrl.addWidget(self.btn_mag_proc)
        mag_ctrl.addWidget(zoom_lbl); mag_ctrl.addWidget(self.mag_zoom_slider); mag_ctrl.addWidget(self.mag_zoom_val)
        mag_ctrl.addWidget(size_lbl); mag_ctrl.addWidget(self.mag_size_slider); mag_ctrl.addWidget(self.mag_size_val)
        mag_ctrl.addStretch()
        mag_lay.addLayout(mag_ctrl)
        mag_group.setLayout(mag_lay)
        c_lay.addWidget(mag_group)

        # ─── Pixel Art ────────────────────────────────────────
        pix_group = QGroupBox("🎮  Pixel Art")
        pix_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        pix_group.setFixedHeight(135)
        pix_group.setStyleSheet(
            "QGroupBox{color:#ff9944;font-weight:bold;border:1px solid #553300;"
            "margin-top:6px;border-radius:4px;background:#1f1505;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}")
        pix_lay = QVBoxLayout(); pix_lay.setContentsMargins(10,4,10,6); pix_lay.setSpacing(6)
        _ss = (
            "QSlider::groove:horizontal{border:1px solid #553300;height:10px;"
            "background:#2a1500;border-radius:5px;}"
            "QSlider::handle:horizontal{background:#ff9944;width:18px;"
            "margin:-4px 0;border-radius:9px;}"
            "QSlider::sub-page:horizontal{background:#ff6600;border-radius:5px;}")
        _vs = ("color:#ff9944;font-size:11px;font-weight:bold;background:#2a1500;"
               "border:1px solid #553300;border-radius:3px;padding:2px;")

        pix_hint = QLabel("Increase pixel size to create a pixelated blocky image (≥20 to see values)")
        pix_hint.setStyleSheet("color:#888;font-size:10px;")
        pix_lay.addWidget(pix_hint)

        ol = QLabel("🎥 Original:"); ol.setStyleSheet("color:#777;font-size:11px;"); ol.setFixedWidth(82)
        self.pix_slider_orig = QSlider(Qt.Horizontal)
        self.pix_slider_orig.setRange(1, 40); self.pix_slider_orig.setValue(1)
        self.pix_slider_orig.setStyleSheet(_ss)
        self.pix_slider_orig.valueChanged.connect(self._on_pix_orig_changed)
        self.pix_val_lbl_orig = QLabel("1  (OFF)")
        self.pix_val_lbl_orig.setFixedWidth(72); self.pix_val_lbl_orig.setAlignment(Qt.AlignCenter)
        self.pix_val_lbl_orig.setStyleSheet(_vs)
        self.btn_values_orig = QPushButton("🔢  Values  OFF")
        self.btn_values_orig.setCheckable(True); self.btn_values_orig.setChecked(False)
        self.btn_values_orig.setEnabled(False); self.btn_values_orig.setFixedWidth(120)
        self.btn_values_orig.setStyleSheet(
            "background:#1a1a1a;color:#444;font-size:10px;border:1px solid #333;border-radius:4px;padding:4px;")
        self.btn_values_orig.toggled.connect(self._on_values_orig_toggled)
        ro = QHBoxLayout(); ro.setSpacing(10)
        ro.addWidget(ol); ro.addWidget(self.pix_slider_orig, 1)
        ro.addWidget(self.pix_val_lbl_orig); ro.addWidget(self.btn_values_orig)

        pl2 = QLabel("✨ Processed:"); pl2.setStyleSheet("color:#00ffcc;font-size:11px;"); pl2.setFixedWidth(82)
        self.pix_slider = QSlider(Qt.Horizontal)
        self.pix_slider.setRange(1, 40); self.pix_slider.setValue(1)
        self.pix_slider.setStyleSheet(_ss)
        self.pix_slider.valueChanged.connect(self._on_pix_changed)
        self.pix_val_lbl = QLabel("1  (OFF)")
        self.pix_val_lbl.setFixedWidth(72); self.pix_val_lbl.setAlignment(Qt.AlignCenter)
        self.pix_val_lbl.setStyleSheet(_vs)
        self.btn_values = QPushButton("🔢  Values  OFF")
        self.btn_values.setCheckable(True); self.btn_values.setChecked(False)
        self.btn_values.setEnabled(False); self.btn_values.setFixedWidth(120)
        self.btn_values.setStyleSheet(
            "background:#1a1a1a;color:#444;font-size:10px;border:1px solid #333;border-radius:4px;padding:4px;")
        self.btn_values.toggled.connect(self._on_values_toggled)
        rp = QHBoxLayout(); rp.setSpacing(10)
        rp.addWidget(pl2); rp.addWidget(self.pix_slider, 1)
        rp.addWidget(self.pix_val_lbl); rp.addWidget(self.btn_values)

        pix_lay.addLayout(ro); pix_lay.addLayout(rp)
        pix_group.setLayout(pix_lay); c_lay.addWidget(pix_group)

        # ═══════════════════ RIGHT PANEL ═════════════════════
        right = QWidget(); right.setFixedWidth(400)
        r_lay = QVBoxLayout(right); r_lay.setContentsMargins(4,4,4,4)
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_change)

        def _scroll(w):
            sa = QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(w); return sa

        self.kernel_viewer       = KernelViewer()
        self.stats_viewer        = StatsViewer()
        self.channels_viewer     = ChannelsViewer()
        self.custom_kernel_panel = CustomKernelPanel()
        self.custom_kernel_panel.kernel_changed.connect(self._on_custom_kernel_changed)
        self.playground_panel    = CVPlaygroundPanel()
        self.playground_panel.settings_changed.connect(self._on_playground_changed)
        self.playground_panel.btn_reset.clicked.connect(self._reset_playground_state)
        self.pipeline_panel = PipelineBuilderPanel()
        self.pipeline_panel.pipeline_changed.connect(self._on_pipeline_changed)
        self.learning_panel = LearningModePanel()
        self.learning_panel.apply_op_requested.connect(self._on_learning_apply)

        self.tabs.addTab(_scroll(self.kernel_viewer),       "🔲 Kernel")
        self.tabs.addTab(_scroll(self.stats_viewer),        "📊 Stats")
        self.tabs.addTab(_scroll(self.channels_viewer),     "🎨 Channels")
        self.tabs.addTab(_scroll(self.custom_kernel_panel), "🧮 Custom K")
        self.tabs.addTab(_scroll(self.playground_panel),    "🔟 Playground")
        self.tabs.addTab(_scroll(self.pipeline_panel),      "🔧 Pipeline")
        self.tabs.addTab(_scroll(self.learning_panel),      "🎓 Learning")
        self.tabs.tabBar().setTabTextColor(4, QColor("#ff44dd"))
        self.tabs.tabBar().setTabTextColor(5, QColor("#ffaa00"))
        self.tabs.tabBar().setTabTextColor(6, QColor("#44ccff"))

        r_lay.addWidget(self.tabs)
        h_main.addWidget(sidebar); h_main.addWidget(center, 1); h_main.addWidget(right)
        self.setCentralWidget(root)
        self._build_params(self.op1.currentText(), self.form1, self.sliders1, primary=True)

    # ── Magnifier button helpers ──────────────────────────────
    @staticmethod
    def _mag_btn_style(active: bool) -> str:
        if active:
            return ("background:#003344;color:#00ccff;font-size:10px;"
                    "border:2px solid #00ccff;border-radius:5px;padding:5px;")
        return ("background:#0a1a1a;color:#335566;font-size:10px;"
                "border:1px solid #1a3344;border-radius:5px;padding:5px;")

    def _on_mag_orig_toggled(self, checked: bool):
        self.btn_mag_orig.setText(f"🔍 Lens: Original  {'ON' if checked else 'OFF'}")
        self.btn_mag_orig.setStyleSheet(self._mag_btn_style(checked))
        self.view_orig.set_magnifier_enabled(checked)

    def _on_mag_proc_toggled(self, checked: bool):
        self.btn_mag_proc.setText(f"🔍 Lens: Processed  {'ON' if checked else 'OFF'}")
        self.btn_mag_proc.setStyleSheet(self._mag_btn_style(checked))
        self.view_proc.set_magnifier_enabled(checked)

    # ── Save ──────────────────────────────────────────────────
    def _on_fmt_changed(self, fmt):
        show = (fmt == "JPEG")
        for w in (self.jpeg_q_lbl, self.jpeg_q_slider, self.jpeg_q_val_lbl):
            w.setVisible(show)

    def _show_about(self):
        from PyQt5.QtWidgets import QDialog
        dlg = QDialog(self); dlg.setWindowTitle("About CV Lab Pro")
        dlg.setFixedSize(420, 400); dlg.setStyleSheet("background:#080810;")
        lay = QVBoxLayout(dlg); lay.setContentsMargins(20,20,20,20)
        title = QLabel("⚗️  CV Lab Pro  2026")
        title.setStyleSheet("color:#00ffcc;font-size:18px;font-weight:bold;")
        title.setAlignment(Qt.AlignCenter)
        sub = QLabel("Real-time Computer Vision Laboratory")
        sub.setStyleSheet("color:#555;font-size:11px;"); sub.setAlignment(Qt.AlignCenter)
        info = QLabel("Built with: Python · PyQt5 · OpenCV · NumPy · Ultralytics YOLOv8")
        info.setStyleSheet("color:#666;font-size:11px;padding:20px;"); info.setWordWrap(True)
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(
            "background:#0d0d1a;color:#44ccff;font-size:11px;"
            "border:1px solid #224466;border-radius:5px;padding:8px;")
        btn_close.clicked.connect(dlg.close)
        lay.addWidget(title); lay.addWidget(sub); lay.addWidget(info, 1); lay.addWidget(btn_close)
        dlg.exec_()

    def _ext_and_params(self):
        fmt = self.fmt_combo.currentText()
        ext = {"PNG":".png","JPEG":".jpg","BMP":".bmp","TIFF":".tif"}.get(fmt, ".png")
        params = ([cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q_slider.value()] if fmt == "JPEG"
                  else ([cv2.IMWRITE_PNG_COMPRESSION, 3] if fmt == "PNG" else []))
        return ext, params

    def _write_image(self, img, tag):
        if img is None:
            QMessageBox.warning(self, "No Frame", "No frame available."); return
        ext, params = self._ext_and_params()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"cvlab_{tag}_{ts}{ext}"
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save {tag}",
            os.path.join(self._last_save_dir, default_name),
            f"Image (*{ext})")
        if not path: return
        self._last_save_dir = os.path.dirname(path)
        ok = cv2.imwrite(path, img, params) if params else cv2.imwrite(path, img)
        if ok:
            self.save_status_lbl.setText(f"  ✓  {os.path.basename(path)}")
            self.save_status_lbl.setStyleSheet("color:#44ff88;font-size:10px;")
        else:
            QMessageBox.critical(self, "Save Failed", f"Could not write:\n{path}")
            self.save_status_lbl.setText("  ✗  Save failed")
            self.save_status_lbl.setStyleSheet("color:#ff4444;font-size:10px;")

    def save_processed(self):
        self._write_image(self._last_proc, "processed")

    def save_original(self):
        self._write_image(self._last_orig, "original")

    # ── Playground ────────────────────────────────────────────
    def _on_playground_changed(self, mode, params, enabled):
        if "_yolo_model_name" in params:
            self._playground_proc._yolo_model_name = params["_yolo_model_name"]
            self._playground_proc._yolo_model = None
        self._playground_proc.configure(mode, params, enabled)
        if self._last_proc is not None and not (self.thread and self.thread.isRunning()):
            self._recompute_display()

    def _reset_playground_state(self):
        self._playground_proc.reset_temporal()
        self.playground_panel.update_info("⟳  Temporal state reset")

    def _on_pipeline_changed(self, steps):
        self._pipeline_steps = steps
        if self._last_orig is not None:
            self._notify()

    def _on_learning_apply(self, op):
        idx = self.op1.findText(op)
        if idx >= 0:
            self.op1.setCurrentIndex(idx)

    # ── Custom kernel ─────────────────────────────────────────
    def _on_custom_kernel_changed(self, k):
        self._custom_kernel = k
        if self.op1.currentText() == "Custom Kernel":
            self.kernel_viewer.set_mode("Custom Kernel", {}, custom_kernel=k)
        self._notify()

    # ── Pixel sliders – Original ──────────────────────────────
    def _on_pix_orig_changed(self, val):
        self._pixel_size_orig = val
        if val <= 1:
            self.pix_val_lbl_orig.setText("1  (OFF)")
            self.pix_val_lbl_orig.setStyleSheet(
                "color:#555;font-size:11px;font-weight:bold;"
                "background:#1a1a1a;border:1px solid #333;border-radius:3px;padding:2px;")
            self.btn_values_orig.setEnabled(False)
            if self.btn_values_orig.isChecked(): self.btn_values_orig.setChecked(False)
        else:
            self.pix_val_lbl_orig.setText(f"{val}  px")
            self.pix_val_lbl_orig.setStyleSheet(
                "color:#ff9944;font-size:11px;font-weight:bold;"
                "background:#2a1500;border:1px solid #553300;border-radius:3px;padding:2px;")
            enabled = val >= 20
            self.btn_values_orig.setEnabled(enabled)
            if not enabled and self.btn_values_orig.isChecked():
                self.btn_values_orig.setChecked(False)
            self.btn_values_orig.setStyleSheet(
                "background:#2a1500;color:#ff9944;font-size:10px;border:1px solid #553300;border-radius:4px;padding:4px;"
                if enabled else
                "background:#1a1a1a;color:#444;font-size:10px;border:1px solid #333;border-radius:4px;padding:4px;")
        if self._last_orig is not None:
            self._render_orig(self._last_orig)

    def _on_values_orig_toggled(self, checked):
        self._show_values_orig = checked
        self.btn_values_orig.setText("🔢  Values  ON" if checked else "🔢  Values  OFF")
        self.btn_values_orig.setStyleSheet(
            "background:#2a1500;color:#ff9944;font-size:10px;border:1px solid #ff6600;border-radius:4px;padding:4px;"
            if checked else
            "background:#1a1a1a;color:#777;font-size:10px;border:1px solid #444;border-radius:4px;padding:4px;")
        if self._last_orig is not None:
            self._render_orig(self._last_orig)

    def _render_orig(self, orig):
        display = self._apply_pixelate(orig, self._pixel_size_orig)
        if self._show_values_orig and self._pixel_size_orig > 1:
            display = self._draw_values(display, self._pixel_size_orig)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB); h, w, ch = rgb.shape
        self.view_orig.setPixmap(
            QPixmap.fromImage(QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888))
            .scaled(self.view_orig.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.view_orig.set_frame_size(w, h)
        self.view_orig.set_source_frame(orig)

    # ── Pixel sliders – Processed ─────────────────────────────
    def _on_pix_changed(self, val):
        self._pixel_size = val
        if val <= 1:
            self.pix_val_lbl.setText("1  (OFF)")
            self.pix_val_lbl.setStyleSheet(
                "color:#555;font-size:11px;font-weight:bold;"
                "background:#1a1a1a;border:1px solid #333;border-radius:3px;padding:2px;")
            self.btn_values.setEnabled(False)
            if self.btn_values.isChecked(): self.btn_values.setChecked(False)
        else:
            self.pix_val_lbl.setText(f"{val}  px")
            self.pix_val_lbl.setStyleSheet(
                "color:#ff9944;font-size:11px;font-weight:bold;"
                "background:#2a1500;border:1px solid #553300;border-radius:3px;padding:2px;")
            enabled = val >= 20
            self.btn_values.setEnabled(enabled)
            if not enabled and self.btn_values.isChecked():
                self.btn_values.setChecked(False)
            self.btn_values.setStyleSheet(
                "background:#2a1500;color:#ff9944;font-size:10px;border:1px solid #553300;border-radius:4px;padding:4px;"
                if enabled else
                "background:#1a1a1a;color:#444;font-size:10px;border:1px solid #333;border-radius:4px;padding:4px;")
        if self._last_display_proc is not None:
            self._render_proc_display(self._last_display_proc)

    def _on_values_toggled(self, checked):
        self._show_values = checked
        self.btn_values.setText("🔢  Values  ON" if checked else "🔢  Values  OFF")
        self.btn_values.setStyleSheet(
            "background:#2a1500;color:#ff9944;font-size:10px;border:1px solid #ff6600;border-radius:4px;padding:4px;"
            if checked else
            "background:#1a1a1a;color:#777;font-size:10px;border:1px solid #444;border-radius:4px;padding:4px;")
        if self._last_display_proc is not None:
            self._render_proc_display(self._last_display_proc)

    def _render_proc_display(self, display_frame):
        display = self._apply_pixelate(display_frame, self._pixel_size)
        if self._show_values and self._pixel_size > 1:
            display = self._draw_values(display, self._pixel_size)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB); h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.view_proc.setPixmap(
            QPixmap.fromImage(qi).scaled(
                self.view_proc.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.view_proc.set_frame_size(w, h)
        self.view_proc.set_source_frame(display_frame)

    def _recompute_display(self):
        if self._last_proc is None or self._last_orig is None: return
        display, pg_info = self._playground_proc.apply(self._last_orig, self._last_proc)
        self._last_display_proc = display
        self.playground_panel.update_info(pg_info)
        self._render_proc_display(display)

    # ── Shared static helpers ─────────────────────────────────
    @staticmethod
    def _apply_pixelate(img, pixel_size):
        if pixel_size <= 1: return img
        h, w = img.shape[:2]; sw = max(1, w // pixel_size); sh = max(1, h // pixel_size)
        return cv2.resize(
            cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR),
            (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _draw_values(img, pixel_size):
        out = img.copy(); h, w = out.shape[:2]
        bw = max(1, w // pixel_size); bh = max(1, h // pixel_size)
        bw_px = w // bw; bh_px = h // bh
        fs = max(0.20, min(0.50, bw_px / 55.0)); th = 1
        for row in range(bh):
            for col in range(bw):
                x0 = col * bw_px; x1 = min(x0 + bw_px, w)
                y0 = row * bh_px; y1 = min(y0 + bh_px, h)
                blk = out[y0:y1, x0:x1]
                ab = int(blk[:,:,0].mean()); ag = int(blk[:,:,1].mean()); ar = int(blk[:,:,2].mean())
                gray = int(0.299*ar + 0.587*ag + 0.114*ab)
                tc = (0,0,0) if gray > 128 else (255,255,255)
                cx = x0 + (x1-x0)//2; cy = y0 + (y1-y0)//2; gs = str(gray)
                (tw, tth), _ = cv2.getTextSize(gs, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
                cv2.putText(out, gs, (cx-tw//2, cy+tth//2),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, tc, th, cv2.LINE_AA)
                if bw_px >= 24:
                    rs = f"{ar},{ag},{ab}"; fs2 = fs * 0.55
                    (tw2, _), _ = cv2.getTextSize(rs, cv2.FONT_HERSHEY_SIMPLEX, fs2, 1)
                    cv2.putText(out, rs, (cx-tw2//2, y1-3),
                                cv2.FONT_HERSHEY_SIMPLEX, fs2, tc, 1, cv2.LINE_AA)
                cv2.rectangle(out, (x0,y0), (x1-1, y1-1), (70,70,70), 1)
        return out

    @staticmethod
    def _make_histogram(img):
        out = np.zeros((160, 300, 3), dtype=np.uint8)
        for i, col in enumerate([(255,0,0),(0,255,0),(0,0,255)]):
            h = cv2.calcHist([img],[i],None,[256],[0,256])
            cv2.normalize(h, h, 0, 160, cv2.NORM_MINMAX); h = h.flatten()
            for x in range(1, 256):
                cv2.line(out,(x-1,160-int(h[x-1])),(x,160-int(h[x])),col,1)
        return out

    # ── Load image ────────────────────────────────────────────
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if not path: return
        img = cv2.imread(path)
        if img is None: return
        if self.thread and self.thread.isRunning():
            if self.fps_timer: self.fps_timer.stop()
            self.thread.stop(); self.thread = None
            self.btn_cam.setText("▶  START")
            self.btn_cam.setStyleSheet("background:#0078d7;font-size:13px;")
        self._last_orig = img
        self.btn_load_img.setText(f"🖼  {path.split('/')[-1][:22]}")
        self.info_lbl.setText(f"Image  |  {img.shape[1]}x{img.shape[0]}")
        self._render_orig(img); self._notify()

    # ── Param UI ──────────────────────────────────────────────
    def _build_params(self, mode, form, sliders, primary=False):
        while form.rowCount() > 0:
            form.removeRow(0)
        sliders.clear()
        cfg = OPERATIONS_CONFIG.get(mode, [])
        for p in cfg:
            nl = QLabel(p["name"])
            nl.setStyleSheet("color:#bb88ff;font-size:10px;font-weight:bold;")
            sl = QSlider(Qt.Horizontal)
            sl.setRange(p["min"], p["max"]); sl.setValue(p["default"])
            sl.setStyleSheet(
                "QSlider::groove:horizontal{border:1px solid #442255;height:6px;"
                "background:#1a0a2a;border-radius:3px;}"
                "QSlider::handle:horizontal{background:#ff44dd;width:12px;"
                "margin:-3px 0;border-radius:6px;}"
                "QSlider::sub-page:horizontal{background:#882288;border-radius:3px;}")
            vl = QLabel(str(p["default"]))
            vl.setFixedWidth(32); vl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            vl.setStyleSheet("color:#ff44dd;font-size:10px;font-weight:bold;")
            sl.valueChanged.connect(lambda v, l=vl: l.setText(str(v)))
            sl.valueChanged.connect(lambda _: self._notify())
            sl_row = QHBoxLayout()
            sl_row.setContentsMargins(0,0,0,0); sl_row.setSpacing(4)
            sl_row.addWidget(sl, 1); sl_row.addWidget(vl)
            wrapper = QWidget(); wrapper.setStyleSheet("background:transparent;")
            wl = QVBoxLayout(wrapper)
            wl.setContentsMargins(0,2,0,2); wl.setSpacing(2)
            wl.addWidget(nl); wl.addLayout(sl_row)
            form.addRow(wrapper)
            sliders[p["name"]] = sl
        if primary:
            self._update_filter_explain(mode)

    # ── Notify ────────────────────────────────────────────────
    def _notify(self):
        m1 = self.op1.currentText()
        p1 = {n: s.value() for n, s in self.sliders1.items()}
        ck1 = self._custom_kernel if m1 == "Custom Kernel" else None
        self.kernel_viewer.set_mode(m1, p1, custom_kernel=ck1)
        if self.thread and self.thread.isRunning():
            self.thread.update_settings(m1, p1, ck1)
        elif self._last_orig is not None:
            t_start = time.time()
            proc = ImageProcessor.apply(self._last_orig.copy(), m1, p1, ck1)
            for (pm, pp) in self._pipeline_steps:
                proc = ImageProcessor.apply(proc, pm, pp, None)
            proc_ms = (time.time() - t_start) * 1000
            self._last_proc = proc
            display, pg_info = self._playground_proc.apply(self._last_orig, proc)
            self._last_display_proc = display
            self.playground_panel.update_info(pg_info)
            self._render_proc_display(display)
            hist = self._make_histogram(proc)
            rgb_h = cv2.cvtColor(hist, cv2.COLOR_BGR2RGB)
            hh, hw = rgb_h.shape[:2]
            self.hist_lbl.setPixmap(QPixmap.fromImage(
                QImage(rgb_h.data, hw, hh, 3*hw, QImage.Format_RGB888)))
            idx = self.tabs.currentIndex()
            if idx == 1: self.stats_viewer.update_frame(proc)
            elif idx == 2: self.channels_viewer.update_frame(proc)
            self.pipeline_panel.update_perf(0.0, proc_ms)

    # ── Camera ────────────────────────────────────────────────
    def toggle_cam(self):
        if self.thread is None or not self.thread.isRunning():
            self.thread = VideoThread()
            self.thread.change_pixmap_signal.connect(self._update)
            m1 = self.op1.currentText()
            p1 = {n: s.value() for n, s in self.sliders1.items()}
            ck1 = self._custom_kernel if m1 == "Custom Kernel" else None
            self.thread.update_settings(m1, p1, ck1)
            self.thread.start()
            self.btn_cam.setText("⏹  STOP")
            self.btn_cam.setStyleSheet("background:#c00000;font-size:13px;")
            self.fps_timer = QTimer(self)
            self.fps_timer.timeout.connect(self._update_fps)
            self.fps_timer.start(1000)
        else:
            if self.fps_timer: self.fps_timer.stop()
            self.thread.stop(); self.thread = None
            self.btn_cam.setText("▶  START")
            self.btn_cam.setStyleSheet("background:#0078d7;font-size:13px;")

    def _update_fps(self):
        if self.thread and self.thread.isRunning():
            h, w = (self._last_orig.shape[:2]
                    if self._last_orig is not None else (0, 0))
            self.info_lbl.setText(f"FPS:{self.thread.fps:.1f}  |  Res:{w}x{h}")
            self.pipeline_panel.update_perf(self.thread.fps, 0.0)

    # ── Frame update (camera) ─────────────────────────────────
    def _update(self, orig, proc, hist):
        self._last_orig = orig
        for (pm, pp) in self._pipeline_steps:
            proc = ImageProcessor.apply(proc, pm, pp, None)
        self._last_proc = proc
        display, pg_info = self._playground_proc.apply(orig, proc)
        self._last_display_proc = display
        if self.tabs.currentIndex() == 4:
            self.playground_panel.update_info(pg_info)
        self._render_orig(orig)
        self._render_proc_display(display)
        hist = self._make_histogram(proc)
        rgb_h = cv2.cvtColor(hist, cv2.COLOR_BGR2RGB); hh, hw = rgb_h.shape[:2]
        self.hist_lbl.setPixmap(QPixmap.fromImage(
            QImage(rgb_h.data, hw, hh, 3*hw, QImage.Format_RGB888)))

    def _on_tab_change(self, idx):
        if self._last_proc is None: return
        if idx == 1: self.stats_viewer.update_frame(self._last_proc)
        elif idx == 2: self.channels_viewer.update_frame(self._last_proc)
        elif idx == 4 and self._last_display_proc is not None:
            self.playground_panel.update_info(self._playground_proc.info)

    # ── Hover ─────────────────────────────────────────────────
    def _on_hover(self, fx, fy):
        if self._last_proc is None: return
        h, w = self._last_proc.shape[:2]
        if not (0 <= fx < w and 0 <= fy < h): return
        b, g, r = [int(x) for x in self._last_proc[fy, fx]]
        gray = int(0.299*r + 0.587*g + 0.114*b)
        hsv = cv2.cvtColor(np.array([[[b,g,r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
        self.hover_bar.setText(
            f"  ({fx},{fy})    RGB({r},{g},{b})    Hex #{r:02X}{g:02X}{b:02X}"
            f"    Gray {gray}    HSV({int(hsv[0])},{int(hsv[1])},{int(hsv[2])})")
        self.hover_bar.setStyleSheet(
            f"background:rgb({r},{g},{b});color:{'black' if gray>128 else 'white'};"
            "font-size:11px;border:1px solid #333;padding:2px 8px;border-radius:3px;")

    def _update_filter_explain(self, mode):
        text = FILTER_EXPLANATIONS.get(mode, "No description available for this filter.")
        self.filter_explain_lbl.setText(text)
        color_map = {
            "📐":"#00ccff","📏":"#ff9944","✨":"#ffcc44","📊":"#44ff88",
            "🔮":"#cc44ff","🎯":"#ff4488","🔑":"#ffaa00","🎨":"#ff66cc",
            "〰️":"#44cccc","➕":"#88ff88","➖":"#ff8888","✖️":"#ffff44",
            "➗":"#ffaa44","🧮":"#ff9944","🎲":"#aaaaaa",
        }
        box_color = "#554400"
        for emoji, col in color_map.items():
            if emoji in text:
                box_color = col; break
        self.filter_explain_box.setStyleSheet(
            f"QGroupBox{{color:{box_color};font-weight:bold;border:1px solid {box_color}66;"
            f"margin-top:10px;border-radius:4px;background:#0d0b00;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;}}")
        self.filter_explain_lbl.setStyleSheet(
            f"color:#cccccc;font-size:12px;padding:6px;line-height:160%;")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    lab = CVLab(); lab.show()
    sys.exit(app.exec_())
