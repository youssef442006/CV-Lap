from PyQt5.QtWidgets import (QLabel, QPushButton, QSlider,
    QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QFormLayout,
    QGroupBox, QScrollArea, QFrame
)
from PyQt5.QtCore import pyqtSignal, QTimer, Qt

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ═══════════════════════════════════════════════════════════════
#  OPERATIONS CONFIG
# ═══════════════════════════════════════════════════════════════
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
    "SIFT Features":    [{"name":"Max Features","min":50,"max":2000,"default":500}],
    "Erosion":          [{"name":"Iterations","min":1,"max":10,"default":1}],
    "Dilation":         [{"name":"Iterations","min":1,"max":10,"default":1}],
    "ORB Features":     [{"name":"Max Features","min":50,"max":1000,"default":500}],
    "Segmentation (Otsu)": [],
    "Custom Kernel":    [],
}

# ═══════════════════════════════════════════════════════════════
#  MINI METER
# ═══════════════════════════════════════════════════════════════
class MiniMeter(QWidget):
    def __init__(self, label, color="#44ff88", parent=None):
        super().__init__(parent)
        self._color = color
        lay = QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)
        self._lbl = QLabel(label)
        self._lbl.setStyleSheet("color:#888;font-size:10px;"); self._lbl.setFixedWidth(62)
        self._bar = QLabel(); self._bar.setFixedHeight(10)
        self._bar.setStyleSheet("background:#1a1a1a;border:1px solid #333;border-radius:3px;")
        self._val = QLabel("—")
        self._val.setStyleSheet(f"color:{color};font-size:10px;font-weight:bold;")
        self._val.setFixedWidth(80)
        lay.addWidget(self._lbl); lay.addWidget(self._bar, 1); lay.addWidget(self._val)

    def set_value(self, pct, text, color=None):
        c = color or self._color
        p = max(0.0, min(1.0, pct))
        self._bar.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {c},stop:{p:.3f} {c},"
            f"stop:{min(p+0.005,1):.3f} #1a1a1a,stop:1 #1a1a1a);"
            "border:1px solid #333;border-radius:3px;")
        self._val.setText(text)
        self._val.setStyleSheet(f"color:{c};font-size:10px;font-weight:bold;")


# ═══════════════════════════════════════════════════════════════
#  PIPELINE BUILDER PANEL
# ═══════════════════════════════════════════════════════════════
class PipelineBuilderPanel(QWidget):
    pipeline_changed = pyqtSignal(list)

    ALL_OPS = [op for op in OPERATIONS_CONFIG.keys()
                if op not in ("None", "Custom Kernel")]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._steps        = []
        self._selected_idx = None
        self._last_fps     = 0.0
        self._last_proc    = 0.0
        self._fps_history  = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8,8,8,8); layout.setSpacing(8)

        # ── Title ─────────────────────────────────────────────
        title = QLabel("🔧  Pipeline Builder")
        title.setStyleSheet(
            "color:#ffaa00;font-size:13px;font-weight:bold;letter-spacing:1px;")
        layout.addWidget(title)

        # ── Performance Monitor ───────────────────────────────
        perf_box = QGroupBox("⚡  Performance Monitor")
        perf_box.setStyleSheet(
            "QGroupBox{color:#ffaa00;font-weight:bold;border:1px solid #3a2800;"
            "margin-top:10px;border-radius:6px;background:#0d0900;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;}")
        pb_lay = QVBoxLayout(); pb_lay.setContentsMargins(10,8,10,8); pb_lay.setSpacing(5)
        self._fps_meter  = MiniMeter("FPS",       "#44ff88")
        self._proc_meter = MiniMeter("Proc Time", "#44aaff")
        self._cpu_meter  = MiniMeter("CPU",       "#ffaa44")
        self._mem_meter  = MiniMeter("RAM",       "#cc88ff")
        if not HAS_PSUTIL:
            h = QLabel("💡 pip install psutil  for CPU/RAM")
            h.setStyleSheet("color:#554400;font-size:9px;font-style:italic;")
            pb_lay.addWidget(h)
        for m in (self._fps_meter, self._proc_meter,
                    self._cpu_meter, self._mem_meter):
            pb_lay.addWidget(m)
        perf_box.setLayout(pb_lay)
        layout.addWidget(perf_box)

        # ── Steps list header ─────────────────────────────────
        hdr = QHBoxLayout()
        sl = QLabel("🔗  Pipeline Steps")
        sl.setStyleSheet("color:#ffaa00;font-size:11px;font-weight:bold;")
        hdr.addWidget(sl); hdr.addStretch()
        self._count_lbl = QLabel("0 steps")
        self._count_lbl.setStyleSheet("color:#554400;font-size:10px;")
        hdr.addWidget(self._count_lbl)
        btn_clear = QPushButton("🗑 Clear")
        btn_clear.setStyleSheet(
            "background:#2a0800;color:#ff6644;font-size:10px;"
            "border:1px solid #441100;border-radius:4px;padding:3px 8px;")
        btn_clear.clicked.connect(self._clear_all)
        hdr.addWidget(btn_clear)
        layout.addLayout(hdr)

        # ── Steps scroll area ─────────────────────────────────
        self.steps_widget = QWidget()
        self.steps_widget.setStyleSheet("background:#0a0800;border-radius:4px;")
        self.steps_layout = QVBoxLayout(self.steps_widget)
        self.steps_layout.setContentsMargins(4,4,4,4); self.steps_layout.setSpacing(2)
        self._show_empty()

        scroll = QScrollArea(); scroll.setWidget(self.steps_widget)
        scroll.setWidgetResizable(True); scroll.setFixedHeight(160)
        scroll.setStyleSheet(
            "QScrollArea{border:1px solid #2a1800;border-radius:5px;}"
            "QScrollBar:vertical{background:#111;width:6px;border-radius:3px;}"
            "QScrollBar::handle:vertical{background:#443300;border-radius:3px;}")
        layout.addWidget(scroll)

        # ── Params panel (hidden until step clicked) ──────────
        self.params_box = QGroupBox("⚙️  Step Parameters")
        self.params_box.setStyleSheet(
            "QGroupBox{color:#ffcc44;font-weight:bold;border:1px solid #554400;"
            "margin-top:10px;border-radius:6px;background:#0d0b00;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;}")
        self.params_inner_lay = QVBoxLayout()
        self.params_inner_lay.setContentsMargins(10,8,10,8); self.params_inner_lay.setSpacing(5)
        # placeholder
        self._params_hint = QLabel("← Click a step above to edit its parameters")
        self._params_hint.setStyleSheet("color:#444;font-size:10px;font-style:italic;")
        self._params_hint.setAlignment(Qt.AlignCenter)
        self.params_inner_lay.addWidget(self._params_hint)
        self.params_box.setLayout(self.params_inner_lay)
        layout.addWidget(self.params_box)

        # ── Add Step ──────────────────────────────────────────
        add_box = QGroupBox("➕  Add Step")
        add_box.setStyleSheet(
            "QGroupBox{color:#ffaa00;font-weight:bold;border:1px solid #3a2800;"
            "margin-top:10px;border-radius:6px;background:#0a0800;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;}")
        add_lay = QHBoxLayout(); add_lay.setContentsMargins(8,6,8,8)
        self.add_combo = QComboBox()
        for op in self.ALL_OPS: self.add_combo.addItem(op)
        self.add_combo.setStyleSheet(
            "QComboBox{background:#1a1000;color:#ffcc88;font-size:10px;"
            "border:1px solid #443300;border-radius:4px;padding:4px;}"
            "QComboBox QAbstractItemView{background:#1a1000;color:#ffcc88;"
            "selection-background-color:#3a2000;}")
        btn_add = QPushButton("＋  Add")
        btn_add.setStyleSheet(
            "background:#2a1800;color:#ffaa00;font-size:11px;font-weight:bold;"
            "border:1px solid #ffaa00;border-radius:5px;padding:5px 14px;")
        btn_add.clicked.connect(self._add_step)
        add_lay.addWidget(self.add_combo, 1); add_lay.addWidget(btn_add)
        add_box.setLayout(add_lay)
        layout.addWidget(add_box)
        layout.addStretch()

        # ── Perf timer ────────────────────────────────────────
        self._perf_timer = QTimer(self)
        self._perf_timer.timeout.connect(self._update_perf)
        self._perf_timer.start(1000)
        if HAS_PSUTIL: psutil.cpu_percent(interval=None)

    # ── Perf ──────────────────────────────────────────────────
    def update_perf(self, fps, proc_ms):
        self._last_fps  = fps; self._last_proc = proc_ms
        self._fps_history.append(fps)
        if len(self._fps_history) > 60: self._fps_history.pop(0)

    def _update_perf(self):
        fps = self._last_fps; proc = self._last_proc
        fps_col = "#44ff88" if fps>24 else ("#ffaa00" if fps>12 else "#ff4444")
        self._fps_meter.set_value(min(fps/60,1), f"{fps:.1f} fps", fps_col)
        proc_col = "#44aaff" if proc<50 else ("#ffaa00" if proc<100 else "#ff4444")
        self._proc_meter.set_value(min(proc/200,1), f"{proc:.1f} ms", proc_col)
        if HAS_PSUTIL:
            cpu = psutil.cpu_percent(interval=None)
            cpu_col = "#44ff88" if cpu<60 else ("#ffaa00" if cpu<85 else "#ff4444")
            self._cpu_meter.set_value(cpu/100, f"{cpu:.1f}%", cpu_col)
            vm = psutil.virtual_memory()
            ram_col = "#cc88ff" if vm.percent<70 else ("#ffaa00" if vm.percent<90 else "#ff4444")
            self._mem_meter.set_value(vm.percent/100,
                f"{vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB", ram_col)
        else:
            self._cpu_meter.set_value(0, "N/A", "#444")
            self._mem_meter.set_value(0, "N/A", "#444")

    # ── Steps ─────────────────────────────────────────────────
    def _show_empty(self):
        lbl = QLabel("   No steps yet  —  add operations below")
        lbl.setStyleSheet("color:#333;font-size:10px;font-style:italic;padding:8px 0;")
        self.steps_layout.addWidget(lbl)

    def _add_step(self):
        mode = self.add_combo.currentText()
        if not mode: return
        cfg = OPERATIONS_CONFIG.get(mode, [])
        self._steps.append({"mode": mode,
                            "params": {p["name"]: p["default"] for p in cfg}})
        self._rebuild_steps_ui()
        self._emit()

    def _clear_all(self):
        self._steps.clear(); self._selected_idx = None
        self._rebuild_steps_ui()
        self._clear_params_panel()
        self._emit()

    def _rebuild_steps_ui(self):
        while self.steps_layout.count():
            item = self.steps_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self._count_lbl.setText(
            f"{len(self._steps)} step{'s' if len(self._steps)!=1 else ''}")
        if not self._steps:
            self._show_empty(); return

        for i, step in enumerate(self._steps):
            is_sel = (i == self._selected_idx)
            row = QWidget()
            row.setStyleSheet(
                f"background:{'#2a1c00' if is_sel else ('#181000' if i%2==0 else '#100c00')};"
                f"border:{'1px solid #ffaa00' if is_sel else '1px solid transparent'};"
                "border-radius:4px;margin:1px 0;")
            rl = QHBoxLayout(row); rl.setContentsMargins(6,3,6,3); rl.setSpacing(4)

            num = QLabel(f"{i+1}")
            num.setStyleSheet(
                f"background:{'#ffaa00' if is_sel else '#3a2000'};"
                f"color:{'#000' if is_sel else '#ffaa00'};"
                "font-size:9px;font-weight:bold;border-radius:8px;padding:1px 5px;")
            num.setFixedWidth(22)

            name_l = QLabel(step["mode"])
            name_l.setStyleSheet(
                f"color:{'#ffee88' if is_sel else '#ffcc88'};font-size:10px;"
                f"{'font-weight:bold;' if is_sel else ''}")

            cfg = OPERATIONS_CONFIG.get(step["mode"], [])
            if cfg:
                p_lbl = QLabel(f"({len(cfg)})")
                p_lbl.setStyleSheet("color:#886600;font-size:9px;")
                rl.addWidget(p_lbl)

            def _btn(text, sty):
                b = QPushButton(text); b.setFixedSize(22,22)
                b.setStyleSheet(sty); return b

            btn_sel = _btn("✎",
                "background:#2a1c00;color:#ffcc44;font-size:10px;"
                "border:1px solid #554400;border-radius:4px;")
            btn_up  = _btn("▲",
                "background:#1a1000;color:#ffaa44;font-size:9px;"
                "border:1px solid #332200;border-radius:4px;")
            btn_dn  = _btn("▼",
                "background:#1a1000;color:#ffaa44;font-size:9px;"
                "border:1px solid #332200;border-radius:4px;")
            btn_del = _btn("✕",
                "background:#2a0000;color:#ff6644;font-size:9px;"
                "border:1px solid #440000;border-radius:4px;")

            btn_sel.clicked.connect(lambda _, idx=i: self._select_step(idx))
            btn_up.clicked.connect(lambda _, idx=i: self._move_step(idx,-1))
            btn_dn.clicked.connect(lambda _, idx=i: self._move_step(idx, 1))
            btn_del.clicked.connect(lambda _, idx=i: self._del_step(idx))

            rl.addWidget(num); rl.addWidget(name_l, 1)
            rl.addWidget(btn_sel)
            rl.addWidget(btn_up); rl.addWidget(btn_dn); rl.addWidget(btn_del)
            self.steps_layout.addWidget(row)

        self.steps_layout.addStretch()

    def _del_step(self, idx):
        if 0 <= idx < len(self._steps):
            self._steps.pop(idx)
            if self._selected_idx == idx:
                self._selected_idx = None
                self._clear_params_panel()
            elif self._selected_idx and self._selected_idx > idx:
                self._selected_idx -= 1
            self._rebuild_steps_ui(); self._emit()

    def _move_step(self, idx, d):
        ni = idx + d
        if 0 <= ni < len(self._steps):
            self._steps[idx], self._steps[ni] = self._steps[ni], self._steps[idx]
            if self._selected_idx == idx:   self._selected_idx = ni
            elif self._selected_idx == ni:  self._selected_idx = idx
            self._rebuild_steps_ui(); self._emit()

    # ── Params panel ──────────────────────────────────────────
    def _select_step(self, idx):
        if self._selected_idx == idx:
            self._selected_idx = None
            self._clear_params_panel()
        else:
            self._selected_idx = idx
            self._build_params_panel(idx)
        self._rebuild_steps_ui()

    def _clear_params_panel(self):
        while self.params_inner_lay.count():
            item = self.params_inner_lay.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self._params_hint = QLabel("← Click ✎ on a step to edit its parameters")
        self._params_hint.setStyleSheet("color:#444;font-size:10px;font-style:italic;")
        self._params_hint.setAlignment(Qt.AlignCenter)
        self.params_inner_lay.addWidget(self._params_hint)
        self.params_box.setTitle("⚙️  Step Parameters")

    def _build_params_panel(self, idx):
        step = self._steps[idx]
        mode = step["mode"]
        cfg  = OPERATIONS_CONFIG.get(mode, [])

        while self.params_inner_lay.count():
            item = self.params_inner_lay.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        self.params_box.setTitle(f"⚙️  {mode}")

        if not cfg:
            lbl = QLabel("No adjustable parameters for this operation")
            lbl.setStyleSheet("color:#555;font-size:10px;font-style:italic;")
            lbl.setAlignment(Qt.AlignCenter)
            self.params_inner_lay.addWidget(lbl)
            return

        for p in cfg:
            row = QWidget(); row.setStyleSheet("background:transparent;")
            rl  = QHBoxLayout(row); rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)

            name_lbl = QLabel(p["name"]+":")
            name_lbl.setStyleSheet("color:#aaa;font-size:10px;min-width:120px;")

            sl = QSlider(Qt.Horizontal)
            sl.setRange(p["min"], p["max"])
            current_val = step["params"].get(p["name"], p["default"])
            sl.setValue(current_val)
            sl.setStyleSheet(
                "QSlider::groove:horizontal{border:1px solid #554400;height:8px;"
                "background:#1a1000;border-radius:4px;}"
                "QSlider::handle:horizontal{background:#ffaa00;width:14px;"
                "margin:-3px 0;border-radius:7px;}"
                "QSlider::sub-page:horizontal{background:#554400;border-radius:4px;}")

            val_lbl = QLabel(str(current_val))
            val_lbl.setFixedWidth(38); val_lbl.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
            val_lbl.setStyleSheet("color:#ffcc44;font-size:10px;font-weight:bold;")

            def _on_change(v, l=val_lbl, pname=p["name"], sidx=idx):
                l.setText(str(v))
                self._steps[sidx]["params"][pname] = v
                self._emit()

            sl.valueChanged.connect(_on_change)
            rl.addWidget(name_lbl); rl.addWidget(sl, 1); rl.addWidget(val_lbl)
            self.params_inner_lay.addWidget(row)

        # Reset button
        btn_reset = QPushButton("⟳  Reset to defaults")
        btn_reset.setStyleSheet(
            "background:#1a0f00;color:#aa6600;font-size:10px;"
            "border:1px solid #443300;border-radius:4px;padding:4px;margin-top:4px;")
        btn_reset.clicked.connect(lambda: self._reset_step_params(idx))
        self.params_inner_lay.addWidget(btn_reset)

    def _reset_step_params(self, idx):
        step = self._steps[idx]
        cfg  = OPERATIONS_CONFIG.get(step["mode"], [])
        for p in cfg:
            step["params"][p["name"]] = p["default"]
        self._build_params_panel(idx)
        self._emit()

    # ── Emit ──────────────────────────────────────────────────
    def _emit(self):
        self.pipeline_changed.emit(
            [(s["mode"], s["params"]) for s in self._steps])

    def get_pipeline(self):
        return [(s["mode"], s["params"]) for s in self._steps]