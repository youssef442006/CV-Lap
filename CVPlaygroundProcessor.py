import cv2
import numpy as np
from PyQt5.QtWidgets import (QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QComboBox,
    QSlider, QFormLayout, QCheckBox, QFrame,QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
import time

try:
    from ultralytics import YOLO as _YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

def _draw_label(img, text, pos, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = pos
    cv2.rectangle(img, (x-2, y-th-4), (x+tw+2, y+2), (0,0,0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════
#  CV PLAYGROUND PROCESSOR
# ═══════════════════════════════════════════════════════════════
class CVPlaygroundProcessor:

    def __init__(self):
        self.enabled   = False
        self._mode     = "None"
        self._params   = {}
        self._info     = "—"
        self._prev_gray  = None
        self._prev_frame = None
        self._tracks     = []
        self._bg_sub     = None
        self._bg_sub_cfg = None
        self._frame_cnt  = 0
        self._cascades   = {}
        self._hog        = None
        self._load_detectors()
        self._bg_static_hash = None
        self._bg_static_mask = None
        self._tracker       = None
        self._tracker_type  = None
        self._tracker_bbox  = None
        self._tracker_init_frame = None
        self._sift_ref_frame = None
        self._sift_ref_kp    = None
        self._sift_ref_desc  = None
        self._last_sift_out  = None
        # ── YOLOv8 ────────────────────
        self._yolo_model      = None
        self._yolo_model_name = "yolov8n.pt"
        self._yolo_last_dets  = []
        self._yolo_infer_ms   = 0.0

    def _load_detectors(self):
        base = cv2.data.haarcascades
        cascade_files = {
            "face":  "haarcascade_frontalface_default.xml",
            "eye":   "haarcascade_eye.xml",
            "smile": "haarcascade_smile.xml",
        }
        for k, f in cascade_files.items():
            try:
                c = cv2.CascadeClassifier(base + f)
                if not c.empty():
                    self._cascades[k] = c
            except Exception:
                pass
        try:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except Exception:
            pass

    def configure(self, mode, params, enabled):
        if mode != self._mode:
            self._reset_temporal()
        self._mode    = mode
        self._params  = params.copy()
        self.enabled  = enabled

    def reset_temporal(self):
        self._reset_temporal()

    def _reset_temporal(self):
        self._prev_gray  = None
        self._prev_frame = None
        self._tracks     = []
        self._bg_sub     = None
        self._tracker    = None
        self._tracker_bbox = None
        self._tracker_init_frame = None
        self._sift_ref_frame = None
        self._sift_ref_kp    = None
        self._sift_ref_desc  = None
        self._last_sift_out  = None
        self._bg_sub_cfg = None
        self._frame_cnt  = 0

    @property
    def info(self): return self._info

    def apply(self, orig, proc):
        if not self.enabled or self._mode == "None":
            self._info = "—  Playground disabled"
            return proc, self._info
        out = proc.copy()
        self._info = "—"
        try:
            m, p = self._mode, self._params
            if   m == "Face Recognition":         out = self._face_recognition(orig, out, p)
            elif m == "Eye Detection":            out = self._eye(orig, out, p)
            elif m == "Smile Detection":          out = self._smile(orig, out, p)
            elif m == "Background Subtraction":   out = self._bg_subtract(orig, out, p)
            elif m == "Dense Optical Flow":       out = self._dense_flow(orig, out, p)
            elif m == "Sparse Optical Flow (LK)": out = self._sparse_flow(orig, out, p)
            elif m == "Motion Detection":         out = self._motion(orig, out, p)
            elif m == "Contour Detection":        out = self._contours(orig, out, p)
            elif m == "Corner Detection (Harris)":out = self._harris(orig, out, p)
            elif m == "Depth Map (Stereo approx)":out = self._depth_approx(orig, out, p)
            elif m == "Saliency (Spectral)":      out = self._saliency(orig, out, p)
            elif m == "SIFT Matching":            out = self._sift_matching(orig, out, p)
            elif m == "Watershed Segmentation":   out = self._watershed(orig, out, p)
            elif m == "KMeans Segmentation":      out = self._kmeans_seg(orig, out, p)
            elif m == "YOLO Object Detection":    out = self._yolo_detection(orig, out, p)
        except Exception as e:
            self._info = f"⚠ {str(e)[:90]}"
        self._prev_frame = orig.copy()
        self._frame_cnt += 1
        return out, self._info

    @staticmethod
    def _overlay_rect(img, x, y, x2, y2, color, alpha=0.25):
        sub = img[y:y2, x:x2]
        colored = np.full_like(sub, color[::-1] if len(color)==3 else color)
        img[y:y2, x:x2] = cv2.addWeighted(sub, 1-alpha, colored, alpha, 0)

    def _eye(self, orig, out, p):
        if "eye" not in self._cascades:
            self._info = "⚠ Eye cascade not found"; return out

        self._eye_skip = getattr(self, "_eye_skip", 0) + 1
        if self._eye_skip % 2 != 0:
            return out

        h = orig.shape[0]
        roi = orig[:h//2, :]
        gray_roi = cv2.cvtColor(
            cv2.resize(roi, (roi.shape[1]//2, roi.shape[0]//2)),
            cv2.COLOR_BGR2GRAY)

        sf = p.get("Scale (x100)", 110) / 100.0
        mn = p.get("Min Neighbors", 5)
        eyes = self._cascades["eye"].detectMultiScale(gray_roi, scaleFactor=sf, minNeighbors=mn)

        for (x, y, w, h2) in eyes:
            cx, cy = x*2+w, y*2+h2
            r = max(w,h2)
            cv2.circle(out, (cx, cy), r, (255,140,0), 2)
            cv2.circle(out, (cx, cy), r//3, (255,200,50), -1)
            cv2.circle(out, (cx-2, cy-2), 2, (255,255,255), -1)

        self._info = f"👁 Eyes: {len(eyes)}"
        return out

    def _smile(self, orig, out, p):
        if "smile" not in self._cascades or "face" not in self._cascades:
            self._info = "⚠ Cascades not found"; return out

        self._smile_skip = getattr(self, "_smile_skip", 0) + 1
        self._smile_last = getattr(self, "_smile_last", [])

        if self._smile_skip % 3 != 0:
            for (x1,y1,x2,y2,is_smile) in self._smile_last:
                cv2.rectangle(out,(x1,y1),(x2,y2),
                            (0,255,255) if is_smile else (180,180,180),
                            2 if is_smile else 1)
            self._info = f"😊 Smiles: {sum(1 for *_,s in self._smile_last if s)}"
            return out

        scale = 0.5
        h, w  = orig.shape[:2]
        small = cv2.resize(orig, (int(w*scale), int(h*scale)))

        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

        faces = self._cascades["face"].detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(25,25))

        self._smile_last = []
        total_smiles = 0

        for (fx, fy, fw, fh) in faces:
            fx1,fy1 = int(fx/scale), int(fy/scale)
            fx2,fy2 = int((fx+fw)/scale), int((fy+fh)/scale)
            cv2.rectangle(out,(fx1,fy1),(fx2,fy2),(180,180,180),1)
            self._smile_last.append((fx1,fy1,fx2,fy2,False))
            roi_y = int(fh * 0.55)
            roi   = gray[fy+roi_y : fy+fh, fx : fx+fw]
            if roi.size == 0: continue

            roi = cv2.equalizeHist(roi)

            mn = p.get("Min Neighbors", 20)
            sf = p.get("Scale (x100)", 118) / 100.0

            smiles = self._cascades["smile"].detectMultiScale(
                roi, scaleFactor=sf, minNeighbors=mn,
                minSize=(int(fw*0.3), int(fh*0.1)), 
                maxSize=(fw, int(fh*0.5)))         

            for (sx, sy, sw2, sh2) in smiles:
                x1 = int((fx+sx)/scale)
                y1 = int((fy+roi_y+sy)/scale)
                x2 = int((fx+sx+sw2)/scale)
                y2 = int((fy+roi_y+sy+sh2)/scale)
                cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,255),2)
                self._smile_last.append((x1,y1,x2,y2,True))
                total_smiles += 1

        self._info = f"😊 Smiles: {total_smiles}" + ("  😄" if total_smiles else "")
        return out
    def _bg_subtract(self, orig, out, p):
        is_live = self._frame_cnt > 30
        if is_live:
            hist    = p.get("History", 500)
            vt      = p.get("Var Threshold", 16)
            lr      = p.get("Learn Rate (x1k)", 10) / 1000.0
            cfg_key = (hist, vt)
            if self._bg_sub is None or self._bg_sub_cfg != cfg_key:
                self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                    history=hist, varThreshold=vt, detectShadows=True)
                self._bg_sub_cfg = cfg_key
            mask = self._bg_sub.apply(orig, learningRate=lr)
            colored = np.zeros_like(out)
            colored[mask == 255] = (0, 220, 255)
            colored[mask == 127] = (40, 40, 100)
            out = cv2.addWeighted(out, 0.6, colored, 0.4, 0)
            fg = 100.0 * np.sum(mask==255) / mask.size
            sh = 100.0 * np.sum(mask==127) / mask.size
            self._info = f"🎭 FG:{fg:.1f}%  Shadow:{sh:.1f}%  BG:{100-fg-sh:.1f}%  [MOG2]"
            return out
        img_key = hash(orig[::orig.shape[0]//10, ::orig.shape[1]//10].tobytes())
        if img_key == self._bg_static_hash and self._bg_static_mask is not None:
            fm = self._bg_static_mask
        else:
            h, w = orig.shape[:2]
            scale = 0.3
            sh2, sw = int(h*scale), int(w*scale)
            small = cv2.resize(orig, (sw, sh2))
            lab  = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
            flat = lab.reshape(-1, 3).astype(np.float32)
            _, labels, _ = cv2.kmeans(flat, 5, None,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                3, cv2.KMEANS_PP_CENTERS)
            seg = labels.reshape(sh2, sw).astype(np.uint8)
            strip = seg[int(sh2*0.15):int(sh2*0.85), int(sw*0.25):int(sw*0.75)]
            counts = np.bincount(strip.flatten(), minlength=5)
            fg_clusters = set(np.argsort(counts)[-3:])
            pr_fg = np.isin(seg, list(fg_clusters)).astype(np.uint8)
            gc_mask = np.full((sh2, sw), cv2.GC_PR_BGD, dtype=np.uint8)
            gc_mask[pr_fg==1] = cv2.GC_PR_FGD
            b = max(2, int(min(sh2,sw)*0.04))
            gc_mask[:b,:]=gc_mask[-b:,:]=gc_mask[:,:b]=gc_mask[:,-b:]=cv2.GC_BGD
            bgd=np.zeros((1,65),np.float64); fgd=np.zeros((1,65),np.float64)
            cv2.grabCut(small, gc_mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
            fm = np.where((gc_mask==cv2.GC_BGD)|(gc_mask==cv2.GC_PR_BGD),0,255).astype(np.uint8)
            fm = cv2.resize(fm, (w,h), interpolation=cv2.INTER_NEAREST)
            k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            fm = cv2.morphologyEx(fm, cv2.MORPH_CLOSE, k5, iterations=2)
            self._bg_static_hash = img_key
            self._bg_static_mask = fm
        gray_bg = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        gray_bg = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2BGR).astype(np.float32)
        gray_bg *= 0.25
        mask3  = fm[:, :, np.newaxis] / 255.0
        result = orig.astype(np.float32) * mask3 + gray_bg * (1 - mask3)
        result = np.clip(result, 0, 255).astype(np.uint8)
        cnts,_ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, cnts, -1, (0,220,255), 2)
        fg = 100.0 * np.sum(fm==255) / fm.size
        self._info = f"🎭 FG:{fg:.1f}%  BG:{100-fg:.1f}%  [GrabCut — cached ✓]"
        return result

    def _dense_flow(self, orig, out, p):
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        levels = min(p.get("Pyr Levels", 3), 2)
        ws = p.get("Win Size", 15)
        ws = ws if ws % 2 else ws + 1
        ws = min(ws, 9)
        blend = p.get("Blend (%)", 65) / 100.0
        scale = 0.5
        small_gray = cv2.resize(gray, None, fx=scale, fy=scale)
        if self._prev_gray is None or self._prev_gray.shape != small_gray.shape:
            self._prev_gray = small_gray
            self._info = "⏳ Dense flow: warming up (needs 2 frames)…"
            return out
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, small_gray, None,
            pyr_scale=0.5, levels=levels, winsize=ws,
            iterations=1, poly_n=5, poly_sigma=1.2, flags=0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros((small_gray.shape[0], small_gray.shape[1], 3), dtype=np.uint8)
        hsv[...,1] = 255
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_bgr = cv2.resize(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),
                            (orig.shape[1], orig.shape[0]))
        out = cv2.addWeighted(out, 1.0 - blend, flow_bgr, blend, 0)
        self._prev_gray = small_gray
        self._info = f"🌊 Dense Flow  |  Mean: {float(mag.mean()):.2f} px/f  Max: {float(mag.max()):.1f} px/f"
        return out

    def _sparse_flow(self, orig, out, p):
        gray      = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        max_c     = p.get("Max Corners", 100)
        ws        = p.get("Win Size", 15)
        trail_len = p.get("Trail Len", 20)
        lk = dict(winSize=(ws,ws), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03))
        feat = dict(maxCorners=max_c, qualityLevel=0.3, minDistance=7, blockSize=7)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape or not self._tracks:
            pts = cv2.goodFeaturesToTrack(gray, mask=None, **feat)
            if pts is not None:
                self._tracks = [[(int(pt[0][0]), int(pt[0][1]))] for pt in pts]
            self._prev_gray = gray
            self._info = f"✨ Sparse Flow: seeded {len(self._tracks)} points"
            return out
        if self._tracks:
            p0  = np.float32([[t[-1]] for t in self._tracks])
            p1, st, _  = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None, **lk)
            p0r, st2,_ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, p1, None, **lk)
            good  = (st.reshape(-1)==1) & (abs(p0-p0r).reshape(-1,2).max(-1)<1)
            new_t = []
            for tr, (x,y), ok in zip(self._tracks, p1.reshape(-1,2), good):
                if not ok: continue
                tr.append((int(x), int(y)))
                if len(tr) > trail_len: tr = tr[-trail_len:]
                new_t.append(tr)
                cv2.circle(out, (int(x),int(y)), 4, (0,255,80), -1)
            self._tracks = new_t
            for tr in self._tracks:
                for i in range(1, len(tr)):
                    a = i/len(tr)
                    cv2.line(out, tr[i-1], tr[i],
                            (int(80*a), int(255*a), int(200*(1-a))), 1)
            if self._frame_cnt % 30 == 0:
                mask_new = np.ones_like(gray)
                for tr in self._tracks:
                    cv2.circle(mask_new, tr[-1], 5, 0, -1)
                new_pts = cv2.goodFeaturesToTrack(gray, mask=mask_new, **feat)
                if new_pts is not None:
                    for pt in new_pts[:max(0, max_c-len(self._tracks))]:
                        self._tracks.append([(int(pt[0][0]),int(pt[0][1]))])
        self._prev_gray = gray
        self._info = f"✨ Sparse Flow  |  Active tracks: {len(self._tracks)}"
        return out

    def _motion(self, orig, out, p):
        thresh_v  = p.get("Threshold", 25)
        min_area  = p.get("Min Area", 500)
        blur_sz   = p.get("Blur Size", 5); blur_sz = blur_sz if blur_sz%2 else blur_sz+1
        if self._prev_frame is None or self._prev_frame.shape != orig.shape:
            self._info = "⏳ Motion: waiting for next frame…"
            return out
        g1 = cv2.GaussianBlur(cv2.cvtColor(self._prev_frame,cv2.COLOR_BGR2GRAY),(blur_sz,blur_sz),0)
        g2 = cv2.GaussianBlur(cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY),(blur_sz,blur_sz),0)
        diff = cv2.absdiff(g1, g2)
        _, thresh = cv2.threshold(diff, thresh_v, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, np.ones((5,5),np.uint8), iterations=2)
        contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mc = 0
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area: continue
            x,y,w,h = cv2.boundingRect(cnt)
            self._overlay_rect(out, x,y,x+w,y+h, (0,80,255), 0.15)
            cv2.rectangle(out,(x,y),(x+w,y+h),(0,100,255),2)
            mc += 1
        heat = np.zeros_like(out)
        heat[thresh>0] = (0,60,200)
        out = cv2.addWeighted(out,0.88,heat,0.12,0)
        pct = 100.0*np.sum(thresh>0)/thresh.size
        self._info = f"🏃 Motion regions: {mc}  |  Changed: {pct:.1f}%"
        return out

    def _contours(self, orig, out, p):
        tv = p.get("Thresh Binary", 127)
        ma = p.get("Min Area", 1000)
        th = p.get("Line Thick", 2)
        scale = 0.5
        h, w  = orig.shape[:2]
        small = cv2.resize(orig, (int(w*scale), int(h*scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cols = [(0,255,128),(0,128,255),(255,128,0),(255,0,128),(200,200,0),(0,200,200)]
        drawn, areas = 0, []
        ma_scaled = ma * (scale ** 2)
        for i, cnt in enumerate(cnts):
            a = cv2.contourArea(cnt)
            if a < ma_scaled: continue
            cnt_full = (cnt / scale).astype(np.int32)
            c = cols[i % len(cols)]
            cv2.drawContours(out, [cnt_full], -1, c, th)
            M = cv2.moments(cnt_full)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(out, (cx, cy), 4, c, -1)
            areas.append(a / scale**2); drawn += 1
        avg = np.mean(areas) if areas else 0
        self._info = f"📐 Contours: {drawn}  |  Avg: {avg:.0f} px²  Total: {len(cnts)}"
        return out

    def _harris(self, orig, out, p):
        bs = p.get("Block Size", 2)
        k  = p.get("k (x100)", 4) / 100.0
        tp = p.get("Threshold %", 10) / 100.0
        self._harris_skip = getattr(self, "_harris_skip", 0) + 1
        self._harris_pts  = getattr(self, "_harris_pts",  np.empty((0,2), int))
        if self._harris_skip % 3 == 0:
            scale = 0.4
            h, w  = orig.shape[:2]
            small = cv2.resize(orig, (int(w*scale), int(h*scale)))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            dst   = cv2.cornerHarris(gray, blockSize=bs, ksize=3, k=k)
            mx    = dst.max()
            if mx > 0:
                ys, xs = np.where(dst > tp * mx)
                new_pts = np.column_stack([
                    (xs / scale).astype(int),
                    (ys / scale).astype(int)
                ])
                if len(self._harris_pts) > 0:
                    keep_old = int(len(self._harris_pts) * 0.3)
                    self._harris_pts = np.vstack([new_pts, self._harris_pts[:keep_old]])
                else:
                    self._harris_pts = new_pts
        for x, y in self._harris_pts:
            if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                cv2.circle(out, (int(x), int(y)), 3, (0, 255, 100), -1)
        self._info = f"🔶 Harris corners: {len(self._harris_pts)}"
        return out

    def _depth_approx(self, orig, out, p):
        bs = p.get("Block Size", 11); bs = bs if bs%2 else bs+1
        nd = p.get("Num Disp x16", 4) * 16
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray,(bs,bs),0)
        laplacian = cv2.convertScaleAbs(cv2.Laplacian(blurred,cv2.CV_64F))
        depth_norm = cv2.normalize(laplacian,None,0,255,cv2.NORM_MINMAX)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        out = cv2.addWeighted(out,0.4,depth_color,0.6,0)
        self._info = "🌄 Depth approx (focus-gradient based — use stereo for real depth)"
        return out

    def _saliency(self, orig, out, p):
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        gray_small = cv2.resize(gray,(64,64))
        fft  = np.fft.fft2(gray_small)
        log_amp = np.log(np.abs(fft)+1e-8)
        phase   = np.angle(fft)
        spectral_res = log_amp - cv2.GaussianBlur(log_amp,(9,9),0)
        sal  = np.abs(np.fft.ifft2(np.exp(spectral_res+1j*phase)))**2
        sal  = cv2.resize(sal.astype(np.float32), (orig.shape[1],orig.shape[0]))
        sal  = cv2.GaussianBlur(sal,(11,11),0)
        sal_n = cv2.normalize(sal,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        sal_color = cv2.applyColorMap(sal_n, cv2.COLORMAP_JET)
        out  = cv2.addWeighted(out,0.5,sal_color,0.5,0)
        max_y,max_x = np.unravel_index(np.argmax(sal),sal.shape)
        cv2.circle(out,(max_x,max_y),20,(255,255,255),2)
        cv2.putText(out,"SALIENT",(max_x+22,max_y+6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        self._info = f"🔎 Spectral saliency  |  Peak: ({max_x},{max_y})"
        return out

    def _sift_matching(self, orig, out, p):
        max_feat = p.get("Max Features", 300)
        ratio    = p.get("Match Ratio x10", 7) / 10.0
        h, w     = orig.shape[:2]
        if not hasattr(self, "_sift") or self._sift is None:
            try:
                self._sift = cv2.SIFT_create(nfeatures=max_feat)
                self._bf   = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            except Exception:
                self._info = "⚠ SIFT not available in this OpenCV build"
                return out
        if not hasattr(self, "_sift_ref_desc"):
            self._sift_ref_desc  = None
            self._sift_ref_kp    = None
            self._sift_ref_frame = None
            self._last_sift_out  = None
        if self._frame_cnt % 2 != 0 and self._last_sift_out is not None:
            return self._last_sift_out
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        kp2, desc2 = self._sift.detectAndCompute(gray, None)
        if self._sift_ref_desc is None:
            result = out.copy()
            cv2.drawKeypoints(result, kp2 or [], result,
                            color=(0, 200, 255),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            _draw_label(result, f"Waiting for reference... KP:{len(kp2) if kp2 else 0}",
                        (8, 28), (0, 200, 255))
            self._info = "📸 SIFT: Press 'Capture Reference' in panel"
            self._last_sift_out = result
            return result
        if desc2 is None or len(desc2) < 4:
            self._info = "⚠ SIFT: Not enough features in current frame"
            return self._last_sift_out or out
        try:
            matches = self._bf.knnMatch(self._sift_ref_desc, desc2, k=2)
        except Exception as e:
            self._info = f"⚠ Match error: {e}"
            return self._last_sift_out or out
        good = [m for m, n in (pair for pair in matches if len(pair) == 2)
                if m.distance < ratio * n.distance]
        matched_img = cv2.drawMatches(
            self._sift_ref_frame, self._sift_ref_kp,
            orig, kp2, good[:60], None,
            matchColor=(0, 255, 128), singlePointColor=(0, 150, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        mh = matched_img.shape[0]; mw = matched_img.shape[1]
        scale_f = min(w / mw, h / mh)
        new_w = int(mw * scale_f); new_h = int(mh * scale_f)
        matched_small = cv2.resize(matched_img, (new_w, new_h))
        result = np.zeros_like(orig)
        y0 = (h - new_h) // 2; x0 = (w - new_w) // 2
        result[y0:y0+new_h, x0:x0+new_w] = matched_small
        _draw_label(result, f"REF  KP:{len(self._sift_ref_kp)}", (x0+6,y0+20), (0,200,255))
        _draw_label(result, f"LIVE KP:{len(kp2)}", (x0+new_w//2+6,y0+20), (255,150,0))
        _draw_label(result, f"GOOD MATCHES: {len(good)}", (x0+new_w//2-60,y0+new_h-10), (0,255,128))
        self._info = (f"🔗 SIFT | Ref:{len(self._sift_ref_kp)} Live:{len(kp2)} Matches:{len(good)}")
        self._last_sift_out = result
        return result

    def _watershed(self, orig, out, p):
        dist_t = p.get("Dist Threshold %", 50) / 100.0
        dil_it = p.get("Dilation Iter", 3)
        gray  = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=dil_it)
        dist    = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, dist_t * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers_ws = cv2.watershed(orig, markers)
        n_seg = max(0, markers_ws.max() - 1)
        out2 = out.copy()
        np.random.seed(42)
        colors = np.random.randint(80,255,(n_seg+2,3),dtype=np.uint8)
        for lbl in range(2, markers_ws.max()+1):
            mask = markers_ws == lbl
            out2[mask] = cv2.addWeighted(
                out2, 0.5, np.full_like(out2, colors[lbl-1].tolist()), 0.5, 0)[mask]
        out2[markers_ws == -1] = [0, 0, 255]
        self._info = f"💧 Watershed  |  Segments: {n_seg}  |  dist_t:{dist_t:.2f}"
        return out2

    def _kmeans_seg(self, orig, out, p):
        K = p.get("K Clusters", 4)
        iters = p.get("Max Iter", 10)
        scale = 0.5
        small = cv2.resize(orig, (0,0), fx=scale, fy=scale)
        data = small.reshape((-1,3)).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 1.0)
        _, labels, centers = cv2.kmeans(data, K, None, crit, 1, cv2.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        seg = centers[labels.flatten()].reshape(small.shape)
        seg = cv2.resize(seg, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
        lmap = labels.reshape(small.shape[:2])
        lmap = cv2.resize(lmap.astype(np.uint8), (orig.shape[1], orig.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        np.random.seed(7)
        pal = np.random.randint(100,255,(K,3),dtype=np.uint8)
        overlay = pal[lmap]
        out2 = cv2.addWeighted(seg.astype(np.uint8), 0.65, overlay, 0.35, 0)
        for i,c in enumerate(centers):
            cv2.putText(out2, f"C{i}:({c[2]},{c[1]},{c[0]})",
                        (4,18+i*16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, pal[i].tolist(), 1)
        self._info = f"🎨 KMeans Seg | K={K}"
        return out2

    def _face_recognition(self, orig, out, p):
        if "face" not in self._cascades:
            self._info = "⚠ Face cascade not available"; return out

        self._face_skip = getattr(self, "_face_skip", 0) + 1
        self._face_last = getattr(self, "_face_last", [])

        if self._face_skip % 3 != 0:
            for (x,y,w,h,label,conf) in self._face_last:
                cv2.rectangle(out,(x,y),(x+w,y+h),(50,255,180),2)
                self._overlay_rect(out,x,y,x+w,y+h,(0,180,120),0.12)
                cv2.putText(out,label,(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.55,(50,255,180),2)
            self._info = f"🧑 Faces: {len(self._face_last)}"
            return out

        scale  = 0.5
        h0, w0 = orig.shape[:2]
        small  = cv2.resize(orig, (int(w0*scale), int(h0*scale)))

        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

        faces = self._cascades["face"].detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE)

        self._face_last = []
        for i, (fx,fy,fw,fh) in enumerate(faces):
            x,y,w,h = int(fx/scale),int(fy/scale),int(fw/scale),int(fh/scale)
            label   = f"ID-{i+1}" if len(faces)>1 else "Person"
            cv2.rectangle(out,(x,y),(x+w,y+h),(50,255,180),2)
            self._overlay_rect(out,x,y,x+w,y+h,(0,180,120),0.12)
            cv2.putText(out,label,(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.55,(50,255,180),2)
            self._face_last.append((x,y,w,h,label,0))

        self._info = f"🧑 Faces: {len(faces)}"
        return out
    @staticmethod
    def _lbp_histogram(gray):
        lbp = np.zeros_like(gray)
        for dy in range(-1,2):
            for dx in range(-1,2):
                if dx==0 and dy==0: continue
                neighbor = np.roll(np.roll(gray,dy,axis=0),dx,axis=1)
                lbp += (neighbor >= gray).astype(np.uint8)
        hist,_ = np.histogram(lbp.flatten(), bins=32, range=(0,256))
        return hist.astype(np.float32)

    def _yolo_detection(self, orig, out, p):
        h, w = orig.shape[:2]
        if not HAS_YOLO:
            cv2.putText(out, "pip install ultralytics",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            self._info = "⚠ Run:  pip install ultralytics"
            return out

        conf_thresh = p.get("Conf Thresh x100", 35) / 100.0
        nms_thresh  = p.get("NMS Thresh x100",  40) / 100.0
        max_det     = int(p.get("Max Detections", 30))
        skip        = max(1, int(p.get("Skip Frames", 2)))

        model_name = getattr(self, "_yolo_model_name", "yolov8n.pt")
        if not hasattr(self, "_yolo_model") or self._yolo_model is None:
            try:
                self._yolo_model = _YOLO(model_name)
                self._yolo_infer_ms  = 0
                self._yolo_last_dets = []
            except Exception as e:
                cv2.putText(out, f"YOLO load failed: {str(e)[:40]}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                self._info = f"⚠ YOLO load failed: {e}"
                return out

        if self._frame_cnt % skip != 0:
            detections = getattr(self, "_yolo_last_dets", [])
        else:
            t0 = time.time()
            try:
                results = self._yolo_model.predict(
                    source=orig, conf=conf_thresh, iou=nms_thresh,
                    max_det=max_det, verbose=False, stream=False)
            except Exception as e:
                self._info = f"⚠ YOLO inference: {e}"; return out
            self._yolo_infer_ms = (time.time() - t0) * 1000
            detections = []
            for r in results:
                if r.boxes is None: continue
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    cls_id   = int(box.cls[0].item())
                    conf_val = float(box.conf[0].item())
                    name     = r.names.get(cls_id, str(cls_id))
                    hue      = int((cls_id * 137) % 180)
                    bgr      = cv2.cvtColor(np.array([[[hue,210,220]]],dtype=np.uint8),cv2.COLOR_HSV2BGR)
                    color    = tuple(int(c) for c in bgr[0,0])
                    detections.append((name, x1, y1, x2, y2, conf_val, color))
            self._yolo_last_dets = detections

        counts = {}
        for name, x1, y1, x2, y2, conf, color in detections:
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w-1,x2), min(h-1,y2)
            sub = out[y1:y2, x1:x2]
            if sub.size:
                out[y1:y2,x1:x2] = cv2.addWeighted(sub,0.8,np.full_like(sub,color),0.2,0)
            cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
            bar_w = int((x2-x1)*conf)
            cv2.rectangle(out,(x1,y2-4),(x1+bar_w,y2),color,-1)
            label = f"{name}  {conf*100:.0f}%"
            (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.48,1)
            ly = max(0, y1-th-8)
            cv2.rectangle(out,(x1,ly),(x1+tw+6,y1),color,-1)
            cv2.putText(out,label,(x1+3,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,0,0),1,cv2.LINE_AA)
            counts[name] = counts.get(name,0)+1

        infer_ms = getattr(self,"_yolo_infer_ms",0)
        fps_est  = 1000.0/infer_ms if infer_ms>0 else 0
        model_tag = getattr(self,"_yolo_model_name","yolov8n.pt")
        hud = f"YOLOv8 [{model_tag}]  {infer_ms:.0f}ms ({fps_est:.1f}fps)  objects:{len(detections)}"
        (sw,sh),_ = cv2.getTextSize(hud,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
        cv2.rectangle(out,(0,h-sh-10),(sw+8,h),(0,0,0),-1)
        cv2.putText(out,hud,(4,h-5),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,220,255),1,cv2.LINE_AA)
        summary = "  ".join(f"{k}:{v}" for k,v in sorted(counts.items(),key=lambda x:-x[1])[:5])
        self._info = f"🤖 YOLOv8  |  {len(detections)} objects  |  {infer_ms:.0f}ms  |  {summary}"
        return out


# ═══════════════════════════════════════════════════════════════
#  PLAYGROUND CONFIG
# ═══════════════════════════════════════════════════════════════
PLAYGROUND_CONFIG = {
    "None": [],
    "Eye Detection": [
        {"name":"Scale (x100)", "min":101,"max":150,"default":110},
        {"name":"Min Neighbors","min":1,  "max":10, "default":5},
    ],
    "Smile Detection": [
        {"name":"Scale (x100)", "min":101,"max":150,"default":118},
        {"name":"Min Neighbors","min":10, "max":60, "default":20},
    ],
    "Background Subtraction": [
        {"name":"History",          "min":10, "max":1000,"default":500},
        {"name":"Var Threshold",    "min":4,  "max":100, "default":16},
        {"name":"Learn Rate (x1k)", "min":1,  "max":100, "default":10},
    ],
    "Dense Optical Flow": [
        {"name":"Pyr Levels",  "min":1,"max":5, "default":3},
        {"name":"Win Size",    "min":5,"max":31,"default":15},
        {"name":"Blend (%)",   "min":0,"max":100,"default":65},
    ],
    "Sparse Optical Flow (LK)": [
        {"name":"Max Corners","min":10,"max":500,"default":100},
        {"name":"Win Size",   "min":5, "max":31, "default":15},
        {"name":"Trail Len",  "min":5, "max":60, "default":20},
    ],
    "Motion Detection": [
        {"name":"Threshold",  "min":5,  "max":100,  "default":25},
        {"name":"Min Area",   "min":100,"max":10000,"default":500},
        {"name":"Blur Size",  "min":1,  "max":21,   "default":5},
    ],
    "Contour Detection": [
        {"name":"Thresh Binary","min":0,  "max":255,  "default":127},
        {"name":"Min Area",     "min":100,"max":50000,"default":1000},
        {"name":"Line Thick",   "min":1,  "max":5,    "default":2},
    ],
    "Corner Detection (Harris)": [
        {"name":"Block Size",  "min":2,"max":10, "default":2},
        {"name":"k (x100)",    "min":2,"max":20, "default":4},
        {"name":"Threshold %", "min":10,"max":100,"default":10},
    ],
    "Depth Map (Stereo approx)": [
        {"name":"Block Size",   "min":5, "max":31,"default":11},
        {"name":"Num Disp x16","min":1, "max":16, "default":4},
    ],
    "Saliency (Spectral)": [],
    "SIFT Matching": [
        {"name":"Max Features",   "min":50,"max":400,"default":200},
        {"name":"Match Ratio x10","min":5, "max":9,   "default":7},
    ],
    "Watershed Segmentation": [
        {"name":"Dist Threshold %","min":40,"max":100,"default":50},
        {"name":"Dilation Iter",   "min":1, "max":10,"default":3},
    ],
    "KMeans Segmentation": [
        {"name":"K Clusters","min":2,"max":8,"default":4},
        {"name":"Max Iter",  "min":5,"max":40,"default":20},
    ],
    "Face Recognition": [],
    "YOLO Object Detection": [
        {"name":"Conf Thresh x100", "min":10,"max":90, "default":35},
        {"name":"NMS Thresh x100",  "min":10,"max":70, "default":40},
        {"name":"Max Detections",   "min":5, "max":100,"default":30},
        {"name":"Skip Frames",      "min":1, "max":8,  "default":2},
    ],
}
PLAYGROUND_EXPLANATIONS = {
    "None": "",

    "Face Recognition": (
        "🧑 LBP Face Recognition\n"
        "──────────────────────\n"
        "Stage 1 — Detection (Haar Cascade):\n"
        "  Scans image at multiple scales using a sliding\n"
        "  window of pre-trained rectangle features.\n"
        "  Uses Adaboost to select best 200 features\n"
        "  from 160,000+ candidates.\n"
        "  Cascade: early rejection → very fast.\n\n"
        "Stage 2 — Description (LBP):\n"
        "  For each pixel, compare to 8 neighbors clockwise.\n"
        "  neighbor >= center → 1, else → 0\n"
        "  Result: 8-bit code per pixel (0–255)\n"
        "  Encode entire face as histogram of 32 bins.\n\n"
        "Stage 3 — Confidence:\n"
        "  std(histogram) = how much variation in texture\n"
        "  High std → complex face texture → confident\n"
        "  Low std  → flat/smooth region → uncertain\n\n"
        "⚡ No GPU needed. Runs on CPU in real-time.\n"
        "⚠ Not a real 'recognition' — no identity DB.\n"
        "   To add real recognition: train LBPH recognizer\n"
        "   with cv2.face.LBPHFaceRecognizer_create()"
    ),

    "Eye Detection": (
        "👁 Haar Cascade Eye Detector\n"
        "──────────────────────\n"
        "Same Viola-Jones framework as face detection\n"
        "but trained on eye region patches.\n\n"
        "Optimization applied here:\n"
        "  → Only scans TOP HALF of the frame\n"
        "  → Eyes are biologically always in upper body\n"
        "  → Cuts computation by ~50%\n"
        "  → Also reduces false positives from mouth/chin\n\n"
        "Integral Image trick:\n"
        "  Rectangle sum at ANY size = O(1) lookup\n"
        "  Makes sliding window viable in real-time\n\n"
        "Params:\n"
        "  Scale (x100): step between pyramid levels\n"
        "    1.05 = fine (slow), 1.3 = coarse (fast)\n"
        "  Min Neighbors: confirmations before accepting\n"
        "    Low → more detections + more false positives\n"
        "    High → fewer, more reliable detections\n\n"
        "Visual: orange circle = iris, dot = highlight\n"
        "Frame skip: every 2nd frame for performance"
    ),

    "Smile Detection": (
        "😊 Haar Cascade Smile Detector\n"
        "──────────────────────\n"
        "Two-stage cascade pipeline:\n\n"
        "Stage 1 — Face detection (full image):\n"
        "  scaleFactor=1.3, minNeighbors=5\n"
        "  Returns (x,y,w,h) for each face\n\n"
        "Stage 2 — Smile detection (face ROI only):\n"
        "  Only searches INSIDE each face bounding box\n"
        "  → Reduces search area by ~95%\n"
        "  → Eliminates false positives from elsewhere\n\n"
        "Why smile detection is harder than face:\n"
        "  Smiles vary enormously (size, teeth, lips)\n"
        "  Requires high Min Neighbors (20+) to avoid\n"
        "  detecting every horizontal line as a smile\n\n"
        "Params:\n"
        "  Scale (x100): pyramid scale factor\n"
        "  Min Neighbors: strictness (higher = fewer FP)\n\n"
        "Frame skip: every 3rd frame for performance\n"
        "Resize: 50% before detection for speed"
    ),

    "Background Subtraction": (
        "🎭 Background Subtraction\n"
        "──────────────────────\n"
        "Automatically switches mode based on frame count:\n\n"
        "📹 LIVE MODE (frame_cnt > 30) → MOG2:\n"
        "  Gaussian Mixture Model per pixel:\n"
        "  Each pixel modeled as K=5 Gaussians\n"
        "  Background = stable Gaussians over time\n"
        "  Foreground = pixels deviating from model\n"
        "  Shadow = brighter pixels (value ↑, hue same)\n\n"
        "  History: how many frames to model background\n"
        "    Short history → adapts fast to changes\n"
        "    Long history  → stable, ignores brief motion\n"
        "  Var Threshold: σ² threshold for FG/BG decision\n"
        "  Learn Rate: 0=static model, 1=instant update\n\n"
        "🖼 STATIC MODE (single image) → GrabCut:\n"
        "  1) KMeans(K=5) in LAB space to cluster colors\n"
        "  2) Center strip → guess foreground clusters\n"
        "  3) GrabCut(5 iters) refines boundary\n"
        "  4) Cache result by image hash → compute once\n\n"
        "  Visualization:\n"
        "  FG → natural color  |  BG → grayscale × 0.25"
    ),

    "Dense Optical Flow": (
        "🌊 Dense Optical Flow — Farneback Algorithm\n"
        "──────────────────────\n"
        "Computes a flow vector (vx, vy) for EVERY pixel.\n\n"
        "Core idea — polynomial expansion:\n"
        "  Approximate neighborhood of each pixel as\n"
        "  a 2D polynomial:  f(x) ≈ xᵀAx + bᵀx + c\n"
        "  Track how polynomial changes between frames\n"
        "  → solve for (vx, vy) at each pixel\n\n"
        "Pyramid (Pyr Levels):\n"
        "  Downscale image → compute coarse flow first\n"
        "  → propagate to finer scales\n"
        "  More levels = handles large motions better\n\n"
        "Output encoding (HSV):\n"
        "  H (hue)   = direction of motion (0°–360°)\n"
        "  S         = always 255 (full saturation)\n"
        "  V (value) = speed (normalized magnitude)\n\n"
        "  Red  → moving right\n"
        "  Cyan → moving left\n"
        "  Green→ moving down\n"
        "  Purple→ moving up\n\n"
        "Optimization: runs on 50% scaled image"
    ),

    "Sparse Optical Flow (LK)": (
        "✨ Lucas-Kanade Sparse Optical Flow\n"
        "──────────────────────\n"
        "Tracks only SPECIFIC keypoints (not every pixel).\n\n"
        "Step 1 — Feature seeding (Shi-Tomasi):\n"
        "  Find corners where gradient is large in\n"
        "  BOTH x AND y directions simultaneously\n"
        "  qualityLevel=0.3, minDistance=7px between pts\n\n"
        "Step 2 — LK tracking:\n"
        "  Assumes: pixels in small window move together\n"
        "  Solves: Σ [Ix² IxIy] [vx]   [-IxIt]\n"
        "           [IxIy Iy²] [vy] = [-IyIt]\n"
        "  Window size controls tracking accuracy\n\n"
        "Step 3 — Bidirectional verification:\n"
        "  Track forward: frame t → frame t+1\n"
        "  Track backward: frame t+1 → frame t\n"
        "  If error > 1px → reject track (unreliable)\n\n"
        "Step 4 — Trail visualization:\n"
        "  Stores last N positions per point\n"
        "  Color gradient: old=green/dim → new=cyan/bright\n\n"
        "Refresh: every 30 frames adds new points to\n"
        "  replace lost tracks (keeps count stable)"
    ),

    "Motion Detection": (
        "🏃 Frame Differencing Motion Detection\n"
        "──────────────────────\n"
        "Classical temporal differencing approach:\n\n"
        "Step 1 — Preprocessing:\n"
        "  Convert both frames to grayscale\n"
        "  Apply Gaussian blur (Blur Size param)\n"
        "  Why blur? Removes pixel-level noise that\n"
        "  would create fake 'motion' detections\n\n"
        "Step 2 — Differencing:\n"
        "  diff = |frame_t - frame_{t-1}|\n"
        "  Each pixel: how much has it changed?\n\n"
        "Step 3 — Thresholding:\n"
        "  diff > Threshold → motion pixel (255)\n"
        "  diff ≤ Threshold → background pixel (0)\n"
        "  Threshold controls sensitivity:\n"
        "    Low (5)  → detects tiny movements + noise\n"
        "    High(50) → only detects large movements\n\n"
        "Step 4 — Morphological dilation:\n"
        "  Expands motion regions to connect gaps\n"
        "  Fills holes caused by uniform-color objects\n\n"
        "Step 5 — Contour bounding boxes:\n"
        "  findContours on binary mask\n"
        "  Filter by Min Area → remove small noise\n"
        "  Draw blue bounding boxes on motion regions\n\n"
        "Blue heat overlay = raw changed pixels (12% alpha)"
    ),

    "Contour Detection": (
        "📐 Contour Detection\n"
        "──────────────────────\n"
        "Finds boundaries between objects and background.\n\n"
        "Step 1 — Grayscale + Threshold:\n"
        "  Convert BGR → Gray\n"
        "  Binary threshold: pixel > Thresh → 255, else 0\n"
        "  Creates a black/white silhouette image\n"
        "  Lower threshold → more white regions → more contours\n\n"
        "Step 2 — findContours (RETR_EXTERNAL):\n"
        "  Traces boundary pixels of each white region\n"
        "  EXTERNAL: only outermost contours (no nested)\n"
        "  APPROX_SIMPLE: compresses lines to endpoints\n"
        "  e.g., rectangle → 4 points (not 800)\n\n"
        "Step 3 — Filtering + Drawing:\n"
        "  Skip contours smaller than Min Area (noise)\n"
        "  Each surviving contour gets a unique color\n"
        "  Color cycles through 6 predefined colors\n\n"
        "Step 4 — Centroid (Image Moments):\n"
        "  M = cv2.moments(contour)\n"
        "  cx = M['m10'] / M['m00']  (x center of mass)\n"
        "  cy = M['m01'] / M['m00']  (y center of mass)\n"
        "  Dot drawn at centroid position\n\n"
        "Optimization: runs on 25% scaled image,\n"
        "then maps contours back to full resolution"
    ),

    "Corner Detection (Harris)": (
        "🔶 Harris Corner Detector\n"
        "──────────────────────\n"
        "Finds 'interesting' points stable for tracking.\n\n"
        "What is a corner?\n"
        "  Flat region: gradient ≈ 0 in all directions\n"
        "  Edge: gradient large in ONE direction\n"
        "  Corner: gradient large in ALL directions\n\n"
        "Math — Structure tensor M:\n"
        "  M = Σ_window [Ix²   IxIy]\n"
        "               [IxIy  Iy² ]\n"
        "  where Ix, Iy = image gradients\n\n"
        "Corner response R:\n"
        "  R = det(M) - k × trace(M)²\n"
        "  det(M) = λ₁ × λ₂    (product of eigenvalues)\n"
        "  trace(M) = λ₁ + λ₂  (sum of eigenvalues)\n"
        "  k ≈ 0.04 (Harris-Stephens empirical constant)\n\n"
        "  λ₁,λ₂ both large → R large → CORNER ✅\n"
        "  one large, one small → R negative → edge\n"
        "  both small → R ≈ 0 → flat region\n\n"
        "Anti-flicker smoothing:\n"
        "  New points = 70% new + 30% old cached points\n"
        "  Prevents sudden disappearance of corners\n\n"
        "Optimization: 40% scale, every 3rd frame"
    ),

    "Depth Map (Stereo approx)": (
        "🌄 Single-Image Depth Approximation\n"
        "──────────────────────\n"
        "⚠ TRUE depth requires stereo cameras or LiDAR.\n"
        "This uses a focus-based heuristic instead.\n\n"
        "Physics principle — Depth of Field:\n"
        "  Objects in focus appear SHARP (high frequency)\n"
        "  Objects out of focus appear BLURRY (low freq)\n"
        "  Camera focus point = sharpest region = closest\n\n"
        "Algorithm:\n"
        "  1) Gaussian blur → smooth out fine noise\n"
        "  2) Laplacian (∇²f = ∂²f/∂x² + ∂²f/∂y²)\n"
        "     Measures rate of change of gradient\n"
        "     High response → sharp edges → CLOSER\n"
        "     Low response  → smooth region → FARTHER\n"
        "  3) Normalize to [0, 255]\n"
        "  4) Apply MAGMA colormap:\n"
        "     White/yellow = high Laplacian = CLOSE\n"
        "     Dark purple  = low Laplacian  = FAR\n\n"
        "Limitations:\n"
        "  ✗ Fails for objects outside depth of field\n"
        "  ✗ Flat-colored objects have no gradient\n"
        "  ✗ Not metric (no actual distance in meters)\n\n"
        "Block Size: blur kernel — larger = smoother depth"
    ),

    "Saliency (Spectral)": (
        "🔎 Spectral Residual Saliency\n"
        "──────────────────────\n"
        "Finds what your eye is drawn to first.\n"
        "Based on Hou & Zhang 2007 (CVPR).\n\n"
        "Core idea — Log Spectrum:\n"
        "  Average/expected parts of spectrum = background\n"
        "  Unexpected/residual parts = salient regions\n\n"
        "Algorithm (frequency domain):\n"
        "  1) Resize to 64×64 (fast FFT)\n"
        "  2) FFT(image) → complex spectrum\n"
        "  3) L(f) = log|FFT(image)|  (log amplitude)\n"
        "  4) A(f) = L(f) - h_n(f)   (subtract avg spectrum)\n"
        "     h_n = Gaussian blur of L(f)\n"
        "     This is the 'spectral residual'\n"
        "  5) Reconstruct: IFFT(exp(A(f) + iφ(f)))\n"
        "     where φ = original phase (preserves location)\n"
        "  6) saliency = |IFFT result|² → Gaussian blur\n\n"
        "Output:\n"
        "  JET colormap: Red = most salient\n"
        "  White circle  = single peak attention point\n\n"
        "Use case: auto-crop, gaze prediction, thumbnails"
    ),

    "SIFT Matching": (
        "🔑 SIFT Feature Matching\n"
        "──────────────────────\n"
        "SIFT = Scale-Invariant Feature Transform (Lowe 2004)\n\n"
        "Invariant to: scale, rotation, illumination, viewpoint\n\n"
        "Step 1 — Scale-space extrema detection:\n"
        "  Build DoG pyramid (Difference of Gaussians)\n"
        "  DoG ≈ Laplacian of Gaussian (blob detector)\n"
        "  Find local min/max across scale AND space\n\n"
        "Step 2 — Keypoint localization:\n"
        "  Sub-pixel refinement via Taylor expansion\n"
        "  Reject low contrast (contrastThreshold)\n"
        "  Reject edge responses (edgeThreshold)\n\n"
        "Step 3 — Orientation assignment:\n"
        "  Gradient histogram in 36 bins (10° each)\n"
        "  Peak direction = keypoint orientation\n"
        "  Makes descriptor rotation-invariant\n\n"
        "Step 4 — 128-D descriptor:\n"
        "  4×4 grid of 8-bin gradient histograms\n"
        "  = 4×4×8 = 128 floats per keypoint\n\n"
        "Step 5 — Lowe's ratio test:\n"
        "  good match: dist₁ < ratio × dist₂\n"
        "  Ratio=0.7 → strict, Ratio=0.9 → relaxed\n\n"
        "First run: 'Waiting for reference'\n"
        "Capture a frame → then tracks live matches"
    ),

    "Watershed Segmentation": (
        "💧 Watershed Segmentation\n"
        "──────────────────────\n"
        "Analogy: image = landscape, dark=valley, bright=hill\n"
        "Flood from valley bottoms → watersheds = boundaries\n\n"
        "Step 1 — Otsu threshold:\n"
        "  Auto-find best global threshold\n"
        "  Creates binary: sure object vs sure background\n\n"
        "Step 2 — Morphological opening:\n"
        "  Erosion → Dilation removes tiny noise blobs\n"
        "  Preserves main object shapes\n\n"
        "Step 3 — Distance transform:\n"
        "  Each FG pixel = distance to nearest BG pixel\n"
        "  Peaks of distance map = centers of objects\n\n"
        "Step 4 — Sure regions:\n"
        "  Sure FG = distance > Dist_Threshold × max_dist\n"
        "  Sure BG = dilated opening region\n"
        "  Unknown = sure_BG - sure_FG (boundary zone)\n\n"
        "Step 5 — Marker-based watershed:\n"
        "  Label sure FG regions as seeds (markers)\n"
        "  Unknown = 0 (let watershed decide)\n"
        "  cv2.watershed fills from seeds outward\n"
        "  Collision between regions = boundary (-1)\n\n"
        "Boundaries drawn in RED\n"
        "Each segment gets a random overlay color"
    ),

    "KMeans Segmentation": (
        "🎨 KMeans Color Segmentation\n"
        "──────────────────────\n"
        "Unsupervised clustering: groups similar colors.\n\n"
        "Algorithm (Lloyd's algorithm):\n"
        "  1) Flatten: (H×W×3) → (N, 3) point cloud\n"
        "     Each pixel = one point in RGB space\n"
        "  2) Initialize K centroids (KMeans++ method)\n"
        "     Spreads initial centroids intelligently\n"
        "     → faster convergence than random init\n"
        "  3) E-step: assign each pixel to nearest centroid\n"
        "     Uses Euclidean distance in color space\n"
        "  4) M-step: update centroid = mean of cluster\n"
        "  5) Repeat E+M until convergence or max_iter\n"
        "  6) Replace each pixel with centroid color\n\n"
        "Choosing K:\n"
        "  K=2  → binary foreground/background\n"
        "  K=3  → skin/clothing/background\n"
        "  K=4  → typical scene segmentation\n"
        "  K=8+ → fine-grained color regions\n\n"
        "Color overlay = random palette per cluster\n"
        "Corner labels = centroid BGR values\n"
        "Runs on 50% scale for real-time performance"
    ),

    "YOLO Object Detection": (
        "🤖 YOLOv8 — You Only Look Once\n"
        "──────────────────────\n"
        "End-to-end real-time object detector by Ultralytics.\n"
        "Trained on COCO: 80 classes, 118K images.\n\n"
        "Architecture (single pass):\n"
        "  Backbone: CSPDarknet (feature extraction)\n"
        "  Neck:     FPN+PAN (multi-scale features)\n"
        "  Head:     Decoupled detection head\n"
        "  → Divides image into grid of cells\n"
        "  → Each cell predicts: box + confidence + class\n"
        "  → NMS removes overlapping duplicate boxes\n\n"
        "Models (press 🔄 to cycle):\n"
        "  yolov8n → nano  3.2M params  ~6MB   ~80fps CPU\n"
        "  yolov8s → small 11.2M params ~22MB  ~45fps CPU\n"
        "  yolov8m → med   25.9M params ~50MB  ~25fps CPU\n\n"
        "Params:\n"
        "  Conf Thresh: min objectness × class score\n"
        "    0.25 = detect more  |  0.7 = only sure ones\n"
        "  NMS Thresh: IoU threshold to merge boxes\n"
        "    Low = aggressive merging (miss nearby objs)\n"
        "    High = keeps overlapping boxes\n"
        "  Skip Frames: inference every N frames\n"
        "    Higher = faster UI, slightly stale boxes\n\n"
        "First run downloads weights automatically (~6MB)"
    ),
}
# ═══════════════════════════════════════════════════════════════
#  CV PLAYGROUND PANEL
# ═══════════════════════════════════════════════════════════════
class CVPlaygroundPanel(QWidget):
    settings_changed = pyqtSignal(str, dict, bool)

    _CATEGORIES = {
        "── Detection ──":      ["Face Recognition","Eye Detection","Smile Detection"],
        "── Motion/Temporal ──": ["Background Subtraction","Dense Optical Flow",
                                "Sparse Optical Flow (LK)","Motion Detection"],
        "── Structural ──":      ["Contour Detection","Corner Detection (Harris)"],
        "── Analysis ──":        ["Depth Map (Stereo approx)","Saliency (Spectral)"],
        "── Features ──":        ["SIFT Matching"],
        "── Segmentation ──":    ["Watershed Segmentation","KMeans Segmentation"],
        "── 🤖 AI Section ──":   ["YOLO Object Detection"],
    }

    _YOLO_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sliders        = {}
        self._building       = False
        self._selected_model = "yolov8n.pt"

        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(6)

        title = QLabel("🔟  CV Algorithm Playground")
        title.setStyleSheet("color:#ff44dd;font-size:13px;font-weight:bold;letter-spacing:1px;")
        main.addWidget(title)

        sub = QLabel("Real-time CV algorithms overlaid on the processed frame")
        sub.setStyleSheet("color:#555;font-size:10px;font-style:italic;")
        sub.setWordWrap(True)
        main.addWidget(sub)

        self.btn_enable = QPushButton("⭕  DISABLED  —  Click to Enable")
        self.btn_enable.setCheckable(True)
        self.btn_enable.setStyleSheet(
            "background:#111;color:#444;font-size:12px;font-weight:bold;"
            "border:2px solid #333;border-radius:6px;padding:8px;")
        self.btn_enable.toggled.connect(self._on_toggle)
        main.addWidget(self.btn_enable)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#333;"); main.addWidget(sep)

        alg_row = QHBoxLayout()
        alg_lbl = QLabel("Algorithm:")
        alg_lbl.setStyleSheet("color:#aaa;font-size:10px;"); alg_lbl.setFixedWidth(72)
        self.alg_combo = QComboBox()
        self.alg_combo.addItem("None")
        for cat, items in self._CATEGORIES.items():
            self.alg_combo.addItem(cat)
            idx = self.alg_combo.count()-1
            self.alg_combo.model().item(idx).setEnabled(False)
            self.alg_combo.model().item(idx).setData(QColor("#444"), Qt.ForegroundRole)
            for alg in items:
                self.alg_combo.addItem(f"  {alg}")
        self.alg_combo.currentTextChanged.connect(self._on_alg_changed)
        alg_row.addWidget(alg_lbl); alg_row.addWidget(self.alg_combo, 1)
        main.addLayout(alg_row)

        self.params_form = QFormLayout()
        main.addLayout(self.params_form)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color:#333;"); main.addWidget(sep2)

        info_hdr = QLabel("📡  Detection Info:")
        info_hdr.setStyleSheet("color:#ff44dd;font-size:10px;font-weight:bold;")
        main.addWidget(info_hdr)

        self.info_lbl = QLabel("—  Enable to start")
        self.info_lbl.setStyleSheet(
            "color:#888;font-size:11px;border:1px solid #2a2a2a;padding:8px;"
            "border-radius:4px;background:#0d0d0d;min-height:36px;")
        self.info_lbl.setWordWrap(True)
        main.addWidget(self.info_lbl)
        # ── Algorithm explanation box ─────────────────────────
        self.algo_explain_box = QGroupBox("💡  How It Works")
        self.algo_explain_box.setStyleSheet(
            "QGroupBox{color:#bb88ff;font-weight:bold;border:1px solid #442255;"
            "margin-top:8px;border-radius:4px;background:#0d0a14;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}")
        ex_lay = QVBoxLayout()
        self.algo_explain_lbl = QLabel("← Select an algorithm to see how it works")
        self.algo_explain_lbl.setStyleSheet(
            "color:#888;font-size:11px;padding:5px;")
        self.algo_explain_lbl.setWordWrap(True)
        self.algo_explain_lbl.setAlignment(Qt.AlignTop)
        ex_lay.addWidget(self.algo_explain_lbl)
        self.algo_explain_box.setLayout(ex_lay)
        main.addWidget(self.algo_explain_box)
        self.btn_reset = QPushButton("⟳  Reset Temporal State")
        self.btn_reset.setStyleSheet(
            "background:#1a0a1a;color:#aa44aa;font-size:10px;"
            "border:1px solid #442244;border-radius:4px;padding:5px;")
        self.btn_reset.setToolTip("Clears optical flow tracks, BG model, motion buffers")
        main.addWidget(self.btn_reset)

        main.addStretch()
        self._build_params("None")

    def _on_toggle(self, checked):
        if checked:
            self.btn_enable.setText("✅  ENABLED  —  Click to Disable")
            self.btn_enable.setStyleSheet(
                "background:#0a1f0a;color:#44ff88;font-size:12px;font-weight:bold;"
                "border:2px solid #2a7a46;border-radius:6px;padding:8px;")
        else:
            self.btn_enable.setText("⭕  DISABLED  —  Click to Enable")
            self.btn_enable.setStyleSheet(
                "background:#111;color:#444;font-size:12px;font-weight:bold;"
                "border:2px solid #333;border-radius:6px;padding:8px;")
        self._emit()

    def _on_alg_changed(self, text):
        mode = text.strip()
        if mode in PLAYGROUND_CONFIG:
            self._build_params(mode); self._emit()
            self._update_algo_explain(mode)
            self._emit()
        elif mode == "None":
            self._build_params("None"); self._emit()
            self._update_algo_explain("None") # ← أضف
            self._emit()

    def _build_params(self, mode):
        self._building = True
        while self.params_form.rowCount() > 0:
            self.params_form.removeRow(0)
        self._sliders.clear()
        cfg = PLAYGROUND_CONFIG.get(mode, [])

        if not cfg:
            ph = QLabel("No adjustable parameters")
            ph.setStyleSheet("color:#444;font-size:10px;font-style:italic;")
            ph.setAlignment(Qt.AlignCenter)
            self.params_form.addRow(ph)
        else:
            for p in cfg:
                nl = QLabel(p["name"]+":")
                nl.setStyleSheet("color:#bb88ff;font-size:10px;min-width:120px;")
                sl = QSlider(Qt.Horizontal)
                sl.setRange(p["min"], p["max"]); sl.setValue(p["default"])
                sl.setMinimumWidth(100)
                sl.setStyleSheet(
                    "QSlider::groove:horizontal{border:1px solid #442255;height:8px;"
                    "background:#1a0a2a;border-radius:4px;}"
                    "QSlider::handle:horizontal{background:#ff44dd;width:14px;"
                    "margin:-3px 0;border-radius:7px;}"
                    "QSlider::sub-page:horizontal{background:#882288;border-radius:4px;}")
                vl = QLabel(str(p["default"]))
                vl.setFixedWidth(38)
                vl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                vl.setStyleSheet("color:#ff44dd;font-size:10px;font-weight:bold;")
                sl.valueChanged.connect(lambda v, l=vl: l.setText(str(v)))
                sl.valueChanged.connect(self._emit)
                rw = QWidget(); rw.setStyleSheet("background:transparent;")
                rl = QHBoxLayout(rw); rl.setContentsMargins(0,0,0,0); rl.setSpacing(4)
                rl.addWidget(sl, 1); rl.addWidget(vl)
                self.params_form.addRow(nl, rw)
                self._sliders[p["name"]] = sl

        if mode == "YOLO Object Detection":
            sep = QFrame(); sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color:#441100;")
            self.params_form.addRow(sep)

            self.yolo_model_lbl = QLabel(f"Model: {self._selected_model}")
            self.yolo_model_lbl.setStyleSheet("color:#ff8844;font-size:10px;font-weight:bold;")
            self.params_form.addRow(self.yolo_model_lbl)

            btn_change = QPushButton("🔄  Change Model  (nano → small → medium)")
            btn_change.setStyleSheet(
                "background:#1a0800;color:#ff8844;font-size:10px;font-weight:bold;"
                "border:1px solid #884400;border-radius:4px;padding:6px;")
            btn_change.clicked.connect(self._change_yolo_model)
            self.params_form.addRow(btn_change)

            hint = QLabel("nano=fastest  small=balanced  medium=accurate\n"
                        "First run downloads model automatically (~6MB)")
            hint.setStyleSheet("color:#555;font-size:9px;")
            hint.setWordWrap(True)
            self.params_form.addRow(hint)

        self._building = False

    def _change_yolo_model(self):
        models = self._YOLO_MODELS
        idx    = models.index(self._selected_model) if self._selected_model in models else 0
        self._selected_model = models[(idx + 1) % len(models)]
        if hasattr(self, "yolo_model_lbl"):
            self.yolo_model_lbl.setText(f"Model: {self._selected_model}")
        self.settings_changed.emit(
            "YOLO Object Detection",
            {"_yolo_model_name": self._selected_model,
             **{n: s.value() for n, s in self._sliders.items()}},
            self.btn_enable.isChecked())
        self.info_lbl.setText(f"🔄 Model switched to: {self._selected_model}  (reloading…)")

    def _emit(self):
        if self._building: return
        mode = self.alg_combo.currentText().strip()
        if mode not in PLAYGROUND_CONFIG: mode = "None"
        params  = {n: s.value() for n, s in self._sliders.items()}
        enabled = self.btn_enable.isChecked()
        self.settings_changed.emit(mode, params, enabled)

    def update_info(self, text):
        self.info_lbl.setText(text if text else "—")

    def get_mode(self):
        m = self.alg_combo.currentText().strip()
        return m if m in PLAYGROUND_CONFIG else "None"

    @property
    def save_with_overlay(self):
        return self.chk_save_overlay.isChecked()
    def _update_algo_explain(self, mode):
        text = PLAYGROUND_EXPLANATIONS.get(mode, "")
        if not text:
            self.algo_explain_box.setVisible(False)
            return
        self.algo_explain_box.setVisible(True)
        self.algo_explain_lbl.setText(text)
        # لون حسب الـ category
        color_map = {
            "🧑": "#44ff88", "👁": "#ffaa44", "😊": "#ffcc44",
            "🎭": "#44ccff", "🌊": "#4488ff", "✨": "#00ffcc",
            "🏃": "#ff6644", "📐": "#ff44dd", "🔶": "#ffcc00",
            "🌄": "#cc44ff", "🔎": "#ff4488", "🔑": "#ffaa00",
            "💧": "#44aaff", "🎨": "#ff66cc", "🤖": "#44ff44",
        }
        box_color = "#442255"
        for emoji, col in color_map.items():
            if emoji in text:
                box_color = col; break
        self.algo_explain_box.setStyleSheet(
            f"QGroupBox{{color:{box_color};font-weight:bold;"
            f"border:1px solid {box_color}66;"
            f"margin-top:8px;border-radius:4px;background:#0d0a14;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;}}")
        self.algo_explain_lbl.setStyleSheet(f"color:#cccccc;font-size:11px;padding:5px;")