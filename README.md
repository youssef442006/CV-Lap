<div align="center">

# ⚗️ CV Lab Pro 2026

### Real-time Computer Vision Laboratory

![Python](https://img.shields.io/badge/Python-3.8+-ffcc44?style=flat-square&logo=python&logoColor=black)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-44aaff?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-ff6644?style=flat-square&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-ff44dd?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-00ffcc?style=flat-square)

*An interactive desktop laboratory for learning and experimenting with computer vision — live camera, real-time filters, AI detection, and guided lessons in one place.*

</div>

---

## 📸 Overview

CV Lab Pro is a PyQt5 desktop application built for students, developers, and researchers who want to **see** how computer vision algorithms work — not just read about them.

Load an image or plug in your webcam, apply any of **25 filters**, chain them into a **pipeline**, overlay **15 real-time CV algorithms**, and follow **8 guided lessons** that teach you the theory while you experiment.

---

## ✨ Features at a Glance

| Module | What it does |
|--------|-------------|
| **① Primary Filter** | 25 image operations with live sliders + filter explanation |
| **🔲 Kernel Viewer** | Color-coded matrix, stats, frequency response heatmap |
| **📊 Stats Viewer** | Per-channel Mean/Std/Min/Max with progress bars |
| **🎨 Channels Viewer** | B, G, R channels as separate grayscale previews |
| **🧮 Custom Kernel** | Draw your own 3×3 / 5×5 / 7×7 convolution kernel |
| **🔟 CV Playground** | 15 real-time algorithms on your camera feed |
| **🔧 Pipeline Builder** | Chain filters sequentially with per-step param editing |
| **🎓 Learning Mode** | 8 guided checkpoints — learn by doing, not reading |

---

## 🖥️ Interface

```
┌─────────────────────────────────────────────────────────────┐
│  Original Frame          │  Processed Frame                  │
│  (camera / image)        │  (after filters + pipeline)       │
│                          │                                   │
│  [pixel values overlay]  │  [pixel values overlay]          │
├──────────────────────────┴───────────────────────────────────┤
│  HISTOGRAM (RGB)                                              │
├───────────────────────────────────────────────────────────────┤
│  ▶ START  │ 🖼 Load  │  ① Primary Filter  │  💾 Save         │
├───────────────────────────────────────────────────────────────┤
│  🔲Kernel │ 📊Stats │ 🎨Channels │ 🧮CustomK │ 🔟Playground │
│           │         🔧Pipeline   │   🎓Learning              │
└───────────────────────────────────────────────────────────────┘
```

---

## 🔧 Installation

### Requirements

```bash
pip install PyQt5 opencv-python numpy
pip install psutil          # for CPU/RAM monitor in Pipeline tab
pip install ultralytics     # for YOLOv8 (optional — graceful fallback)
```

### Run

```bash
python main.py
```

> **First YOLO run** will auto-download `yolov8n.pt` (~6MB) on demand.

---

## 📁 File Structure

```
cv-lab-pro/
├── main.py                    # Main window, UI layout, VideoThread
├── ImageProcessor.py          # All 25 filter implementations
├── CVPlaygroundProcessor.py   # 15 CV algorithms + Playground UI
├── PipelineBuilderPanel.py    # Pipeline builder + performance monitor
├── LearningModePanel.py       # 8 guided lessons + journey system
```

> ~4,100 lines total across 5 modules.

---

## ⚗️ Primary Filter — 25 Operations

### Blur & Smooth
| Filter | Description |
|--------|-------------|
| Gaussian Blur | Weighted average kernel — classic smoothing |
| Median Filter | Replaces pixel with neighborhood median — perfect for salt & pepper |
| Bilateral Filter | Smooths flat areas, **preserves edges** — best general denoiser |
| Low Pass Filter | Keeps low frequencies — removes edges and texture |

### Enhance & Sharpen
| Filter | Description |
|--------|-------------|
| Sharpen | Negative kernel weights boost high-frequency details |
| High Pass Filter | Keeps only edges — flat areas become neutral gray |
| Emboss | Asymmetric kernel creates 3D lighting illusion |

### Edge Detection
| Filter | Description |
|--------|-------------|
| Sobel Edges | Gradient magnitude in X and Y — thick edges |
| Laplacian | 2nd derivative — detects all edge orientations at once |
| Canny Edges | Full pipeline: blur → Sobel → non-max suppression → hysteresis |

### Contrast & Color
| Filter | Description |
|--------|-------------|
| Histogram Equalization | Stretches histogram to fill 0–255 range globally |
| CLAHE | Local histogram EQ in tiles — better for faces and uneven lighting |
| Gamma Correction | `output = 255 × (input/255)^(1/γ)` — exposure control |

### Arithmetic
| Filter | Description |
|--------|-------------|
| Add (+) | Uniform brightness increase |
| Subtract (−) | Uniform brightness decrease |
| Multiply (×) | Contrast scaling |
| Divide (÷) | Compresses dynamic range |

### Morphology
| Filter | Description |
|--------|-------------|
| Erosion | Shrinks white regions — removes noise dots |
| Dilation | Expands white regions — fills holes |

### Features & Segmentation
| Filter | Description |
|--------|-------------|
| ORB Features | Fast corner detection + 256-bit BRIEF descriptor |
| SIFT Features | Scale-invariant keypoints + 128-D descriptor |
| Segmentation (Otsu) | Auto-threshold via between-class variance maximization |
| Custom Kernel | Design your own convolution matrix |

### Noise
| Filter | Description |
|--------|-------------|
| Gaussian Noise | Adds N(0,σ) random values — simulates sensor noise |
| Salt and Pepper | Random pixels set to 0 or 255 — simulates transmission errors |

---

## 🔟 CV Playground — 15 Algorithms

### 👤 Detection
| Algorithm | Description |
|-----------|-------------|
| **Face Recognition** | Haar cascade + LBP descriptor. Shows face ID and texture histogram |
| **Eye Detection** | Scans top half of frame only — Haar cascade, frame skip every 2 |
| **Smile Detection** | Two-stage: detect face first, then search lower 55% of face ROI |

### 🌊 Motion & Temporal
| Algorithm | Description |
|-----------|-------------|
| **Background Subtraction** | MOG2 for live camera · GrabCut + KMeans for static images |
| **Dense Optical Flow** | Farneback — estimates (vx,vy) for every pixel, HSV color encoding |
| **Sparse Optical Flow (LK)** | Lucas-Kanade — tracks Shi-Tomasi corner points with trail visualization |
| **Motion Detection** | Frame differencing → threshold → contour bounding boxes |

### 📐 Structural
| Algorithm | Description |
|-----------|-------------|
| **Contour Detection** | Binary threshold → findContours → color-coded boundaries + centroid dots |
| **Corner Detection (Harris)** | Structure tensor R = det(M) − k·trace(M)² — anti-flicker smoothing |

### 🔬 Analysis
| Algorithm | Description |
|-----------|-------------|
| **Depth Map (approx)** | Laplacian response as depth proxy — MAGMA colormap |
| **Saliency (Spectral)** | Spectral Residual method (Hou & Zhang 2007) — finds attention peak |

### 🔑 Features
| Algorithm | Description |
|-----------|-------------|
| **SIFT Matching** | Capture reference frame → track keypoints live with Lowe's ratio test |

### 🎨 Segmentation
| Algorithm | Description |
|-----------|-------------|
| **Watershed** | Distance transform → marker-based flooding — boundaries in red |
| **KMeans Segmentation** | Groups pixels by color similarity — K=2 to 12 clusters |

### 🤖 AI
| Algorithm | Description |
|-----------|-------------|
| **YOLO Object Detection** | YOLOv8 via Ultralytics — 80 COCO classes, 3 model sizes |

**YOLO Models:**
```
yolov8n  (nano)   ~6MB   — fastest, real-time on CPU
yolov8s  (small)  ~22MB  — balanced speed/accuracy
yolov8m  (medium) ~50MB  — most accurate
```

---

## 🔧 Pipeline Builder

Chain any number of filters and apply them sequentially to every frame.

```
Primary Filter → [Pipeline Step 1] → [Step 2] → [Step 3] → Playground Overlay
```

**Features:**
- Add / remove / reorder steps (▲▼)
- Click **✎** on any step to edit its parameters inline with sliders
- **⟳ Reset** to defaults per step
- Live performance monitor: FPS · Processing time · CPU % · RAM

---

## 🎓 Learning Mode — Guided Journey

8 checkpoints that teach computer vision from scratch **by doing**.

```
1 ──── 2 ──── 3 ──── 4 ──── 5 ──── 6 ──── 7 ──── 8
✓      ✓     (you)
```

| # | Checkpoint | What you learn |
|---|-----------|----------------|
| 🔲 1 | Convolution | Kernels, weighted sums, why kernel sum matters |
| 〰️ 2 | Frequency Filters | LPF, HPF, unsharp masking, Fourier concept |
| 📐 3 | Edge Detection | Gradients, Sobel, Laplacian, full Canny pipeline |
| 🎚️ 4 | Thresholding | Binary threshold, Otsu's method, preprocessing |
| 🔮 5 | Morphology | Erosion, Dilation, Opening, Closing, Gradient |
| 🎲 6 | Noise & Denoising | Gaussian vs Salt&Pepper, Median vs Bilateral |
| 🎨 7 | Contrast Enhancement | Histogram EQ, CLAHE, Gamma correction |
| 🔑 8 | Feature Detection | ORB, SIFT, keypoints, descriptors, matching |

Each checkpoint has:
- **Theory** with ASCII diagrams and formulas
- **4 interactive steps** — click "▶ Do it" and the filter applies immediately
- **Insight box** with practical tips
- **Checkpoint system** — complete all steps to advance

---

## 💡 Pro Tips

```
✔  Always Gaussian Blur before Canny or Otsu — cleaner results every time
✔  Use Median Filter for salt & pepper noise — Gaussian will spread it
✔  Use Bilateral Filter for Gaussian noise — preserves edges unlike blur
✔  Pipeline + Playground stack: pipeline runs first, overlay on top
✔  Open 🔲 Kernel tab while switching filters — watch the matrix change
✔  SIFT Matching: switch to it and let it detect keypoints first, 
   then it starts matching automatically against the initial frame
✔  Harris corners use 70% new + 30% old cached points — no flickering
✔  KMeans K=2 → FG/BG split · K=4 → scene segmentation · K=8 → fine colors
✔  Histogram in the sidebar always reflects post-pipeline processed frame
```

---

## 📦 What You'll Understand After Using This

- How convolution kernels work and why kernel sum matters
- The difference between spatial and frequency domain filtering
- Why Canny produces cleaner edges than raw Sobel
- How Otsu automatically finds the optimal binary threshold
- The math behind Erosion, Dilation, Opening, and Closing
- Why Median beats Gaussian blur for salt-and-pepper noise
- How optical flow tracks motion pixel-by-pixel between frames
- What makes SIFT invariant to scale, rotation, and illumination
- How YOLO detects 80 object classes in a single neural network forward pass

---

## ⚠️ Known Limitations

- **Smile detection** works best with frontal face, good lighting, and visible teeth — OpenCV's smile cascade is inherently unreliable
- **Face Recognition** uses LBP texture description only — not a real identity database. Labels are positional (Person, ID-1, ID-2...)
- **Depth Map** is a focus-gradient approximation — not real metric depth (requires stereo cameras for that)
- **Playground Processor** preferre to use it on live video than photos
- 

---

## 🛠️ Built With

| Library | Version | Role |
|---------|---------|------|
| Python | 3.8+ | Core language |
| PyQt5 | 5.15+ | Desktop UI framework |
| OpenCV | 4.x | All CV algorithms and filters |
| NumPy | any | Array operations |
| psutil | any | Cross-platform CPU/RAM monitoring |
| ultralytics | latest | YOLOv8 object detection |

---

<div align="center">
<sub>CV Lab Pro 2026 · Built for learners, by learners</sub>
</div>
