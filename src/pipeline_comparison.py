"""
Interactive GUI Application for People Counting Pipeline Comparison.

Allows testing different pipeline configurations:
- Enable / Disable Semantic ROI filter (YOLO Semantic).
- Enable / Disable Depth Estimation & Depth-Aware NMS (Depth Anything V2).
- Enable / Disable Bounding Box Refinement.
- Adjust Detection Confidence, IoU, IoA, and Depth Difference thresholds via interactive sliders.
- Real-time re-filtering without re-running heavy model inference.
- Select any image file via file browser.
- Switch between visualization modes (Final Result, 2x2 Summary, ROI Mask, Depth Map, Debug Duplicates).
- Detailed detections table showing bounding boxes, confidence, depth, and suppression status.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import tkinter as tk
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageTk
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineComparisonGUI")

# Resolve workspace root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "models").exists() else SCRIPT_DIR


@dataclass
class DetectionRecord:
    box_id: int
    raw_bbox: list[int]
    refined_bbox: list[int]
    confidence: float
    depth: float
    depth_std: float
    kept: bool = True
    suppressed_by: int | None = None
    suppression_reason: str | None = None


class PipelineEngine:
    """Manages model loading, caching, and inference."""

    def __init__(
        self,
        sem_model_path: str = str(PROJECT_ROOT / "models" / "yolo26x-sem.pt"),
        det_model_path: str = str(PROJECT_ROOT / "models" / "yolo26x.pt"),
        depth_checkpoint: str = "depth-anything/Depth-Anything-V2-Large-hf",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sem_model_path = sem_model_path
        self.det_model_path = det_model_path
        self.depth_checkpoint = depth_checkpoint

        # Eagerly initialize the CUDA context here in the main thread.
        # Lazy CUDA init inside a background thread causes a deadlock with
        # Tkinter's event loop on Linux (the CUDA driver calls back into the
        # main thread for initialization, which is blocked by root.mainloop).
        if self.device == "cuda":
            logger.info("Pre-warming CUDA context on main thread...")
            _dummy = torch.zeros(1, device="cuda")
            del _dummy
            torch.cuda.synchronize()
            logger.info("CUDA context ready.")

        # Models (lazily initialized on first use)
        self.sem_model: YOLO | None = None
        self.det_model: YOLO | None = None
        self.depth_processor = None
        self.depth_model = None

        # Pending colormap for depth visualization (applied in main thread)
        self._pending_colormap: str = "viridis"

        # Cached image and intermediate results
        self.current_image_path: str | None = None
        self.orig_bgr: np.ndarray | None = None
        self.sem_mask: np.ndarray | None = None
        self.masked_bgr: np.ndarray | None = None
        self.raw_boxes: np.ndarray | None = None
        self.raw_confs: np.ndarray | None = None
        self.depth_norm: np.ndarray | None = None
        self.depth_colored_bgr: np.ndarray | None = None

    def ensure_sem_model(self):
        if self.sem_model is None:
            logger.info(f"Loading semantic model: {self.sem_model_path}")
            self.sem_model = YOLO(self.sem_model_path)

    def ensure_det_model(self):
        if self.det_model is None:
            logger.info(f"Loading detection model: {self.det_model_path}")
            self.det_model = YOLO(self.det_model_path)

    def ensure_depth_model(self):
        if self.depth_model is None:
            logger.info(f"Loading Depth Anything V2: {self.depth_checkpoint}")
            self.depth_processor = AutoImageProcessor.from_pretrained(self.depth_checkpoint)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.depth_checkpoint).to(self.device)
            self.depth_model.eval()

    def load_image(self, image_path: str) -> bool:
        if not os.path.exists(image_path):
            return False
        img = cv2.imread(image_path)
        if img is None:
            return False
        self.current_image_path = image_path
        self.orig_bgr = img
        # Invalidate previous caches
        self.sem_mask = None
        self.masked_bgr = None
        self.raw_boxes = None
        self.raw_confs = None
        self.depth_norm = None
        self.depth_colored_bgr = None
        return True

    def run_inference(
        self,
        use_sem_roi: bool,
        use_depth: bool,
        non_roi_fill: str = "black",
        colormap: str = "viridis",
        imgsz: int = 1280,
    ):
        """Runs the neural network stages and populates caches."""
        if self.orig_bgr is None:
            raise ValueError("No image loaded.")

        h, w = self.orig_bgr.shape[:2]

        # 1. Semantic ROI
        if use_sem_roi:
            self.ensure_sem_model()
            sem_res = self.sem_model(self.orig_bgr, classes=[11], imgsz=imgsz, verbose=False)[0]
            if sem_res.semantic_mask is not None:
                mask_data = sem_res.semantic_mask.data
                if isinstance(mask_data, torch.Tensor):
                    mask_data = mask_data.cpu().numpy()
                self.sem_mask = mask_data == 11
            else:
                self.sem_mask = np.ones((h, w), dtype=bool)

            masked = self.orig_bgr.copy()
            if non_roi_fill == "black":
                masked[~self.sem_mask] = 0
            elif non_roi_fill == "white":
                masked[~self.sem_mask] = 255
            elif non_roi_fill == "blur":
                blurred = cv2.GaussianBlur(self.orig_bgr, (51, 51), 0)
                masked[~self.sem_mask] = blurred[~self.sem_mask]
            self.masked_bgr = masked
        else:
            self.sem_mask = np.ones((h, w), dtype=bool)
            self.masked_bgr = self.orig_bgr.copy()

        # 2. Object Detection
        self.ensure_det_model()
        input_for_det = self.masked_bgr if use_sem_roi else self.orig_bgr
        det_res = self.det_model(input_for_det, classes=[0], imgsz=imgsz, conf=0.05, verbose=False)[0]
        if det_res.boxes is not None and len(det_res.boxes) > 0:
            self.raw_boxes = det_res.boxes.xyxy.cpu().numpy().astype(int)
            self.raw_confs = det_res.boxes.conf.cpu().numpy()
        else:
            self.raw_boxes = np.empty((0, 4), dtype=int)
            self.raw_confs = np.empty((0,), dtype=float)

        # 3. Depth Estimation
        if use_depth:
            self.ensure_depth_model()
            pil_image = Image.fromarray(cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2RGB))
            inputs = self.depth_processor(images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                pred_depth = self.depth_model(**inputs).predicted_depth
            pred_depth = torch.nn.functional.interpolate(
                pred_depth.unsqueeze(1),
                size=pil_image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

            d_min, d_max = float(pred_depth.min()), float(pred_depth.max())
            if d_max > d_min:
                self.depth_norm = (pred_depth - d_min) / (d_max - d_min)
            else:
                self.depth_norm = np.zeros_like(pred_depth)

            # Matplotlib is not thread-safe: store colormap name and apply it
            # in the main thread via apply_depth_colormap().
            self._pending_colormap = colormap
            self.depth_colored_bgr = None  # Will be populated in main thread
        else:
            self.depth_norm = np.zeros((h, w), dtype=np.float32)
            self.depth_colored_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    def apply_depth_colormap(self):
        """Apply depth colormap using matplotlib (call from main thread only)."""
        if self.depth_norm is None or self.depth_colored_bgr is not None:
            return
        cmap = plt.colormaps.get_cmap(self._pending_colormap)
        depth_colored_rgb = (cmap(self.depth_norm)[:, :, :3] * 255).astype(np.uint8)
        self.depth_colored_bgr = cv2.cvtColor(depth_colored_rgb, cv2.COLOR_RGB2BGR)

    def filter_and_disambiguate(
        self,
        conf_thresh: float,
        iou_thresh: float,
        ioa_thresh: float,
        depth_thresh: float,
        use_depth: bool,
        refine_boxes: bool,
    ) -> list[DetectionRecord]:
        """Fast post-processing step to filter detections and apply Depth-NMS."""
        if self.raw_boxes is None or len(self.raw_boxes) == 0:
            return []

        h, w = self.orig_bgr.shape[:2]

        # Filter by confidence first
        valid_idx = np.where(self.raw_confs >= conf_thresh)[0]
        if len(valid_idx) == 0:
            return []

        boxes = self.raw_boxes[valid_idx]
        confs = self.raw_confs[valid_idx]

        records: list[DetectionRecord] = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop_mask = self.sem_mask[y1:y2, x1:x2] if self.sem_mask is not None else None
            crop_depth = self.depth_norm[y1:y2, x1:x2] if self.depth_norm is not None else None

            # Bounding box refinement
            if refine_boxes and crop_mask is not None and np.any(crop_mask):
                ys, xs = np.where(crop_mask)
                rx1 = int(x1 + xs.min())
                ry1 = int(y1 + ys.min())
                rx2 = int(x1 + xs.max() + 1)
                ry2 = int(y1 + ys.max() + 1)
                person_depth = crop_depth[crop_mask] if crop_depth is not None else np.array([0.0])
            else:
                rx1, ry1, rx2, ry2 = int(x1), int(y1), int(x2), int(y2)
                person_depth = crop_depth if crop_depth is not None else np.array([0.0])

            med_depth = float(np.median(person_depth)) if len(person_depth) > 0 else 0.0
            std_depth = float(np.std(person_depth)) if len(person_depth) > 0 else 0.0

            records.append(
                DetectionRecord(
                    box_id=i,
                    raw_bbox=[int(x1), int(y1), int(x2), int(y2)],
                    refined_bbox=[rx1, ry1, rx2, ry2],
                    confidence=float(confs[i]),
                    depth=med_depth,
                    depth_std=std_depth,
                )
            )

        # Apply NMS / Depth-NMS
        order = np.argsort(confs)[::-1]
        active = list(order)

        while len(active) > 0:
            curr_i = active.pop(0)
            curr = records[curr_i]

            remaining = list(active)
            active.clear()

            for other_i in remaining:
                other = records[other_i]
                b1 = curr.refined_bbox
                b2 = other.refined_bbox

                x1 = max(b1[0], b2[0])
                y1 = max(b1[1], b2[1])
                x2 = min(b1[2], b2[2])
                y2 = min(b1[3], b2[3])

                is_overlap = False
                metric_name = ""

                if x2 > x1 and y2 > y1:
                    inter = (x2 - x1) * (y2 - y1)
                    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    iou = inter / float(a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0
                    ioa = inter / float(min(a1, a2)) if min(a1, a2) > 0 else 0.0

                    if iou >= iou_thresh:
                        is_overlap = True
                        metric_name = f"IoU={iou:.2f}"
                    elif ioa >= ioa_thresh:
                        is_overlap = True
                        metric_name = f"IoA={ioa:.2f}"

                if is_overlap:
                    if use_depth:
                        depth_diff = abs(curr.depth - other.depth)
                        if depth_diff < depth_thresh:
                            other.kept = False
                            other.suppressed_by = curr.box_id
                            other.suppression_reason = (
                                f"Duplicate of #{curr.box_id} ({metric_name}, d_diff={depth_diff:.4f} < {depth_thresh})"
                            )
                        else:
                            active.append(other_i)
                    else:
                        # Standard 2D NMS (no depth awareness)
                        other.kept = False
                        other.suppressed_by = curr.box_id
                        other.suppression_reason = f"Standard NMS duplicate of #{curr.box_id} ({metric_name})"
                else:
                    active.append(other_i)

        return records


class PipelineComparisonGUI:
    """Tkinter based desktop GUI application for pipeline comparison."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Student & People Counting - Pipeline Comparison Lab")
        self.root.geometry("1480x920")
        self.root.minsize(1100, 700)

        # Style configuration
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.engine = PipelineEngine()
        self.detections: list[DetectionRecord] = []
        self.current_display_img: np.ndarray | None = None
        self.photo_img: ImageTk.PhotoImage | None = None

        # Thread-safe message queue: background thread puts a message here,
        # the main thread polls via root.after and picks it up.
        # NOTE: root.after() called FROM a background thread is silently
        # dropped in Tk 9.0 on Linux. Polling a queue from the main thread
        # is the only reliable cross-thread callback pattern.
        import queue
        self._msg_queue: queue.Queue[str] = queue.Queue()
        self._start_queue_poll()

        # Build UI
        self._create_layout()

        # Load default image if available
        default_img = PROJECT_ROOT / "data" / "images" / "classroom.png"
        if default_img.exists():
            self.image_path_var.set(str(default_img))
            self._load_selected_image()

    def _start_queue_poll(self):
        """Poll the message queue every 100 ms from the main thread."""
        try:
            msg = self._msg_queue.get_nowait()
            if msg == "inference_done":
                self._on_inference_complete()
            elif msg.startswith("inference_error:"):
                self._on_inference_error(msg[len("inference_error:"):])
        except Exception:
            pass  # queue.Empty or any other transient error
        self.root.after(100, self._start_queue_poll)

    def _create_layout(self):
        # Main PanedWindow (Left Controls, Right Visualizations)
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left panel: Scrollable controls
        left_frame = ttk.Frame(paned, width=440)
        paned.add(left_frame, weight=0)

        canvas = tk.Canvas(left_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.scrollable_ctrl = ttk.Frame(canvas, padding=12)

        self.scrollable_ctrl.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.scrollable_ctrl, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Right panel: Visualizations and Notebook
        right_frame = ttk.Frame(paned, padding=10)
        paned.add(right_frame, weight=1)

        self._build_control_widgets(self.scrollable_ctrl)
        self._build_display_widgets(right_frame)

    def _build_control_widgets(self, parent: ttk.Frame):
        # 1. Image Selection Section
        img_sec = ttk.LabelFrame(parent, text=" 1. Input Image ", padding=10)
        img_sec.pack(fill=tk.X, pady=(0, 10))

        self.image_path_var = tk.StringVar()
        path_entry = ttk.Entry(img_sec, textvariable=self.image_path_var, font=("Helvetica", 9))
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        browse_btn = ttk.Button(img_sec, text="Browse...", command=self._browse_image)
        browse_btn.pack(side=tk.RIGHT)

        # 2. Pipeline Architecture Toggles
        mod_sec = ttk.LabelFrame(parent, text=" 2. Pipeline Configuration ", padding=10)
        mod_sec.pack(fill=tk.X, pady=(0, 10))

        self.use_sem_var = tk.BooleanVar(value=True)
        cb_sem = ttk.Checkbutton(
            mod_sec,
            text="Enable Semantic ROI (yolo26x-sem)",
            variable=self.use_sem_var,
            command=self._on_config_toggle,
        )
        cb_sem.pack(anchor=tk.W, pady=2)

        # Non-ROI Fill mode
        fill_frame = ttk.Frame(mod_sec)
        fill_frame.pack(fill=tk.X, padx=(20, 0), pady=(0, 6))
        ttk.Label(fill_frame, text="Non-ROI fill:").pack(side=tk.LEFT)
        self.non_roi_fill_var = tk.StringVar(value="black")
        fill_combo = ttk.Combobox(
            fill_frame,
            textvariable=self.non_roi_fill_var,
            values=["black", "white", "blur", "none"],
            state="readonly",
            width=10,
        )
        fill_combo.pack(side=tk.LEFT, padx=6)

        self.use_depth_var = tk.BooleanVar(value=True)
        cb_depth = ttk.Checkbutton(
            mod_sec,
            text="Enable Depth Estimation & Depth-NMS (DA2)",
            variable=self.use_depth_var,
            command=self._on_config_toggle,
        )
        cb_depth.pack(anchor=tk.W, pady=2)

        # Colormap
        cmap_frame = ttk.Frame(mod_sec)
        cmap_frame.pack(fill=tk.X, padx=(20, 0), pady=(0, 6))
        ttk.Label(cmap_frame, text="Depth Colormap:").pack(side=tk.LEFT)
        self.colormap_var = tk.StringVar(value="viridis")
        cmap_combo = ttk.Combobox(
            cmap_frame,
            textvariable=self.colormap_var,
            values=["viridis", "plasma", "inferno", "turbo"],
            state="readonly",
            width=10,
        )
        cmap_combo.pack(side=tk.LEFT, padx=6)

        self.refine_boxes_var = tk.BooleanVar(value=True)
        cb_refine = ttk.Checkbutton(
            mod_sec,
            text="Snap Bounding Boxes to Exact Person Pixels",
            variable=self.refine_boxes_var,
            command=self._on_slider_change,
        )
        cb_refine.pack(anchor=tk.W, pady=2)

        self.show_suppressed_var = tk.BooleanVar(value=False)
        cb_supp = ttk.Checkbutton(
            mod_sec,
            text="Display Suppressed Duplicates (Red Boxes)",
            variable=self.show_suppressed_var,
            command=self._on_slider_change,
        )
        cb_supp.pack(anchor=tk.W, pady=2)

        # Box representation style
        box_style_frame = ttk.Frame(mod_sec)
        box_style_frame.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(
            box_style_frame,
            text="Box Style:",
            font=("Helvetica", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(2, 8))
        self.box_style_var = tk.StringVar(value="box")
        for _text, _val in [("Box", "box"), ("Dot", "dot"), ("Box + Dot", "both")]:
            rb = ttk.Radiobutton(
                box_style_frame,
                text=_text,
                value=_val,
                variable=self.box_style_var,
                command=self._on_slider_change,
            )
            rb.pack(side=tk.LEFT, padx=4)

        # Dot size slider (meaningful in Dot / Box+Dot modes)
        dot_hdr = ttk.Frame(mod_sec)
        dot_hdr.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(dot_hdr, text="Dot Size Scale:", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        self._dot_size_lbl = ttk.Label(dot_hdr, text="1.0×", foreground="#0066cc")
        self._dot_size_lbl.pack(side=tk.RIGHT)

        self.dot_size_var = tk.DoubleVar(value=1.0)

        def _sync_dot(val):
            snapped = round(float(val) / 0.1) * 0.1
            self.dot_size_var.set(snapped)
            self._dot_size_lbl.config(text=f"{snapped:.1f}×")
            self._on_slider_change()

        self.dot_size_scale = ttk.Scale(
            mod_sec, from_=0.3, to=4.0, variable=self.dot_size_var, command=_sync_dot
        )
        self.dot_size_scale.pack(fill=tk.X, pady=(2, 4))

        def _dot_key(event, direction):
            cur = round(self.dot_size_var.get() / 0.1) * 0.1
            new = max(0.3, min(4.0, cur + direction * 0.1))
            _sync_dot(new)
            return "break"

        self.dot_size_scale.bind("<Left>",  lambda e: _dot_key(e, -1))
        self.dot_size_scale.bind("<Down>",  lambda e: _dot_key(e, -1))
        self.dot_size_scale.bind("<Right>", lambda e: _dot_key(e, +1))
        self.dot_size_scale.bind("<Up>",    lambda e: _dot_key(e, +1))

        # Show / hide labels
        self.show_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            mod_sec,
            text="Show Labels (#id, confidence, depth)",
            variable=self.show_labels_var,
            command=self._on_slider_change,
        ).pack(anchor=tk.W, pady=(0, 4))

        # 3. Hyperparameters & Sliders
        hyp_sec = ttk.LabelFrame(parent, text=" 3. Thresholds & Sliders ", padding=10)
        hyp_sec.pack(fill=tk.X, pady=(0, 10))

        # Slider helper
        def make_slider(sec, label, from_, to_, default, resolution, fmt):
            var = tk.DoubleVar(value=default)
            hdr = ttk.Frame(sec)
            hdr.pack(fill=tk.X, pady=(4, 0))
            lbl_title = ttk.Label(hdr, text=label, font=("Helvetica", 9, "bold"))
            lbl_title.pack(side=tk.LEFT)
            lbl_val = ttk.Label(hdr, text=fmt.format(default), foreground="#0066cc")
            lbl_val.pack(side=tk.RIGHT)

            def _sync(val):
                snapped = round(float(val) / resolution) * resolution
                var.set(snapped)
                lbl_val.config(text=fmt.format(snapped))
                self._on_slider_change()

            scale = ttk.Scale(sec, from_=from_, to=to_, variable=var, command=_sync)
            scale.pack(fill=tk.X, pady=(2, 6))

            # Arrow key stepping: focus the scale widget then use Left/Right/Up/Down.
            # ttk.Scale passes arrow events to Tk's built-in handler which moves by 1
            # unit (not by our resolution), so we intercept and handle it ourselves.
            def _on_key(event, direction):
                cur = round(var.get() / resolution) * resolution
                new = max(from_, min(to_, cur + direction * resolution))
                new = round(new / resolution) * resolution
                _sync(new)
                return "break"  # Prevent Tk's default 1-unit movement

            scale.bind("<Left>",  lambda e: _on_key(e, -1))
            scale.bind("<Down>",  lambda e: _on_key(e, -1))
            scale.bind("<Right>", lambda e: _on_key(e, +1))
            scale.bind("<Up>",    lambda e: _on_key(e, +1))

            return var, scale, lbl_title

        self.conf_var, self.conf_scale, _ = make_slider(
            hyp_sec, "Confidence Threshold", 0.05, 0.90, 0.25, 0.05, "{:.2f}"
        )
        self.iou_var, self.iou_scale, _ = make_slider(
            hyp_sec, "IoU Overlap Threshold", 0.05, 0.90, 0.25, 0.05, "{:.2f}"
        )
        self.ioa_var, self.ioa_scale, _ = make_slider(
            hyp_sec, "IoA (Containment) Threshold", 0.10, 0.90, 0.35, 0.05, "{:.2f}"
        )
        self.depth_thresh_var, self.depth_scale, self.depth_lbl = make_slider(
            hyp_sec, "Depth Difference Threshold", 0.001, 0.050, 0.010, 0.001, "{:.3f}"
        )

        # 4. Actions
        act_sec = ttk.Frame(parent)
        act_sec.pack(fill=tk.X, pady=(0, 10))

        self.btn_run = tk.Button(
            act_sec,
            text="▶  RUN FULL INFERENCE",
            bg="#007acc",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief=tk.RAISED,
            pady=8,
            command=self._start_inference_thread,
        )
        self.btn_run.pack(fill=tk.X, pady=(0, 6))

        btn_save = ttk.Button(act_sec, text="Save Output Images & JSON", command=self._save_results)
        btn_save.pack(fill=tk.X)

        # 5. Results & Statistics Card
        res_sec = ttk.LabelFrame(parent, text=" 4. Results Summary ", padding=10)
        res_sec.pack(fill=tk.X, pady=(0, 10))

        self.lbl_count = tk.Label(
            res_sec,
            text="Count: --",
            font=("Helvetica", 20, "bold"),
            fg="#228b22",
        )
        self.lbl_count.pack(pady=4)

        self.lbl_stats = ttk.Label(
            res_sec,
            text="Initial Detections: --\nFiltered Duplicates: --\nOcclusions Kept: --",
            justify=tk.LEFT,
            font=("Helvetica", 9),
        )
        self.lbl_stats.pack(anchor=tk.W, pady=2)

        self.lbl_status = ttk.Label(
            res_sec,
            text="Ready. Select image and click 'Run Full Inference'.",
            wraplength=380,
            foreground="#666666",
        )
        self.lbl_status.pack(anchor=tk.W, pady=(6, 0))

    def _build_display_widgets(self, parent: ttk.Frame):
        # Notebook for Visualizations & Table
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Image Canvas Viewer
        tab_viewer = ttk.Frame(notebook)
        notebook.add(tab_viewer, text="  Visualization Viewer  ")

        # View Mode Toolbar
        bar = ttk.Frame(tab_viewer, padding=(0, 0, 0, 6))
        bar.pack(fill=tk.X)

        ttk.Label(bar, text="View Mode: ", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        self.view_mode_var = tk.StringVar(value="final")
        modes = [
            ("Final Result", "final"),
            ("2x2 Summary Grid", "grid"),
            ("Semantic ROI Mask", "sem"),
            ("Depth Map", "depth"),
            ("Debug (Show Duplicates)", "debug"),
        ]
        for text, val in modes:
            rb = ttk.Radiobutton(
                bar,
                text=text,
                value=val,
                variable=self.view_mode_var,
                command=self._update_display_image,
            )
            rb.pack(side=tk.LEFT, padx=6)

        # Image Canvas
        self.canvas_frame = ttk.Frame(tab_viewer)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, bg="#202020", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._render_canvas_image())

        # Tab 2: Detections Table
        tab_table = ttk.Frame(notebook, padding=10)
        notebook.add(tab_table, text="  Detections Table & Logs  ")

        columns = ("id", "bbox", "conf", "depth", "status", "reason")
        self.tree = ttk.Treeview(tab_table, columns=columns, show="headings", height=15)
        self.tree.heading("id", text="ID")
        self.tree.heading("bbox", text="Bounding Box [x1, y1, x2, y2]")
        self.tree.heading("conf", text="Confidence")
        self.tree.heading("depth", text="Median Depth")
        self.tree.heading("status", text="Status")
        self.tree.heading("reason", text="Notes / Suppression Reason")

        self.tree.column("id", width=40, anchor=tk.CENTER)
        self.tree.column("bbox", width=180, anchor=tk.CENTER)
        self.tree.column("conf", width=90, anchor=tk.CENTER)
        self.tree.column("depth", width=90, anchor=tk.CENTER)
        self.tree.column("status", width=90, anchor=tk.CENTER)
        self.tree.column("reason", width=400, anchor=tk.W)

        tree_scroll = ttk.Scrollbar(tab_table, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _browse_image(self):
        f = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")],
            initialdir=str(PROJECT_ROOT / "data" / "images"),
        )
        if f:
            self.image_path_var.set(f)
            self._load_selected_image()

    def _load_selected_image(self):
        img_path = self.image_path_var.get()
        if not img_path:
            return
        success = self.engine.load_image(img_path)
        if success:
            self.lbl_status.config(
                text=f"Loaded image: {Path(img_path).name} ({self.engine.orig_bgr.shape[1]}x{self.engine.orig_bgr.shape[0]})"
            )
            # Display original image directly
            self.current_display_img = self.engine.orig_bgr.copy()
            self._render_canvas_image()
            self.lbl_count.config(text="Count: --")
            self.lbl_stats.config(text="Click 'Run Full Inference' to process.")
        else:
            messagebox.showerror("Error", f"Failed to load image:\n{img_path}")

    def _on_config_toggle(self):
        use_depth = self.use_depth_var.get()
        if use_depth:
            self.depth_scale.state(["!disabled"])
        else:
            self.depth_scale.state(["disabled"])

    def _start_inference_thread(self):
        img_path = self.image_path_var.get()
        if not img_path or not os.path.exists(img_path):
            messagebox.showwarning("Warning", "Please select a valid image file first.")
            return

        self.btn_run.config(state=tk.DISABLED, text="Processing Inference...")
        self.lbl_status.config(text="Running models in background...")

        thread = threading.Thread(target=self._run_inference_worker, daemon=True)
        thread.start()

    def _run_inference_worker(self):
        try:
            self.engine.run_inference(
                use_sem_roi=self.use_sem_var.get(),
                use_depth=self.use_depth_var.get(),
                non_roi_fill=self.non_roi_fill_var.get(),
                colormap=self.colormap_var.get(),
            )
            # Signal completion via queue (root.after from threads is unreliable in Tk 9.0).
            self._msg_queue.put("inference_done")
        except Exception as e:
            logger.exception("Inference error")
            self._msg_queue.put(f"inference_error:{e}")

    def _on_inference_complete(self):
        # Apply matplotlib colormap here in the main thread (thread-safe).
        self.engine.apply_depth_colormap()
        self.btn_run.config(state=tk.NORMAL, text="▶  RUN FULL INFERENCE")
        self.lbl_status.config(text="Inference complete. Adjust sliders to fine-tune results.")
        self._reapply_filtering()

    def _on_inference_error(self, err: str):
        self.btn_run.config(state=tk.NORMAL, text="▶  RUN FULL INFERENCE")
        self.lbl_status.config(text=f"Error: {err}")
        messagebox.showerror("Inference Error", f"An error occurred during inference:\n{err}")

    def _on_slider_change(self):
        # Quick re-filter without re-running models
        if self.engine.raw_boxes is not None:
            self._reapply_filtering()

    def _reapply_filtering(self):
        self.detections = self.engine.filter_and_disambiguate(
            conf_thresh=self.conf_var.get(),
            iou_thresh=self.iou_var.get(),
            ioa_thresh=self.ioa_var.get(),
            depth_thresh=self.depth_thresh_var.get(),
            use_depth=self.use_depth_var.get(),
            refine_boxes=self.refine_boxes_var.get(),
        )

        kept_people = [d for d in self.detections if d.kept]
        suppressed_dups = [d for d in self.detections if not d.kept]
        people_count = len(kept_people)

        # Update stats
        self.lbl_count.config(text=f"People Count: {people_count}")
        self.lbl_stats.config(
            text=f"Initial Candidate Detections: {len(self.detections)}\n"
            f"Filtered Duplicates: {len(suppressed_dups)}\n"
            f"Active Person Boxes: {people_count}"
        )

        # Update Table
        self._update_table()

        # Update visualizer
        self._update_display_image()

    def _update_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for d in self.detections:
            status_text = "KEPT" if d.kept else "SUPPRESSED"
            reason_text = d.suppression_reason or "Valid Person"
            self.tree.insert(
                "",
                tk.END,
                values=(
                    f"#{d.box_id}",
                    str(d.refined_bbox),
                    f"{d.confidence:.2f}",
                    f"{d.depth:.4f}",
                    status_text,
                    reason_text,
                ),
            )

    def _update_display_image(self):
        if self.engine.orig_bgr is None:
            return

        mode = self.view_mode_var.get()
        show_supp = self.show_suppressed_var.get() or (mode == "debug")

        if mode in ("final", "debug"):
            self.current_display_img = self._render_annotated_image(show_supp)
        elif mode == "sem":
            if self.engine.masked_bgr is not None:
                self.current_display_img = self.engine.masked_bgr.copy()
            else:
                self.current_display_img = self.engine.orig_bgr.copy()
        elif mode == "depth":
            if self.engine.depth_colored_bgr is not None:
                self.current_display_img = self.engine.depth_colored_bgr.copy()
            else:
                self.current_display_img = self.engine.orig_bgr.copy()
        elif mode == "grid":
            self.current_display_img = self._render_grid_image()

        self._render_canvas_image()

    def _render_annotated_image(self, show_suppressed: bool) -> np.ndarray:
        annotated = self.engine.orig_bgr.copy()
        kept = [d for d in self.detections if d.kept]
        style = self.box_style_var.get()          # "box" | "dot" | "both"
        dot_scale = self.dot_size_var.get()       # user-controlled multiplier
        show_labels = self.show_labels_var.get()  # show #id / conf / depth labels

        for det in self.detections:
            x1, y1, x2, y2 = det.refined_bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # Base radius: 15% of the shorter box side; user scale then applied.
            dot_r = max(3, int(min(x2 - x1, y2 - y1) * 0.15 * dot_scale))

            if det.kept:
                color = (0, 220, 0)

                if style in ("box", "both"):
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                if style in ("dot", "both"):
                    cv2.circle(annotated, (cx, cy), dot_r, color, -1)
                    cv2.circle(annotated, (cx, cy), dot_r, (0, 0, 0), 1)

                if show_labels:
                    label = f"#{det.box_id} {det.confidence:.2f} D:{det.depth:.3f}"
                    label_y = y1 if style != "dot" else cy - dot_r
                    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    by = max(0, label_y - th - base - 4)
                    cv2.rectangle(annotated, (x1, by), (x1 + tw + 4, label_y), color, -1)
                    cv2.putText(
                        annotated, label, (x1 + 2, label_y - base - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
                    )

            elif show_suppressed:
                color = (0, 0, 255)

                if style in ("box", "both"):
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

                if style in ("dot", "both"):
                    cv2.circle(annotated, (cx, cy), dot_r, color, -1)
                    cv2.circle(annotated, (cx, cy), dot_r, (0, 0, 0), 1)

                if show_labels:
                    lx = cx - dot_r if style == "dot" else x1
                    ly = cy - dot_r if style == "dot" else y2 - 4
                    cv2.putText(
                        annotated, f"#{det.box_id} DUP", (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
                    )


        count_badge = f"People Count: {len(kept)}"
        (cw, ch), _ = cv2.getTextSize(count_badge, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(annotated, (15, 15), (35 + cw, 30 + ch), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            count_badge,
            (25, 20 + ch),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _render_grid_image(self) -> np.ndarray:
        target_h, target_w = 480, 800

        def panel(img: np.ndarray, title: str) -> np.ndarray:
            r = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            banner = np.zeros((36, target_w, 3), dtype=np.uint8)
            cv2.putText(banner, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            return np.vstack([banner, r])

        orig = self.engine.orig_bgr
        sem = self.engine.masked_bgr if self.engine.masked_bgr is not None else orig
        depth = self.engine.depth_colored_bgr if self.engine.depth_colored_bgr is not None else np.zeros_like(orig)
        ann = self._render_annotated_image(show_suppressed=False)

        count = sum(1 for d in self.detections if d.kept)
        p1 = panel(orig, "1. Input Image")
        p2 = panel(sem, "2. Semantic ROI Mask (Background Extracted)")
        p3 = panel(depth, "3. Depth Map (Depth Anything V2)")
        p4 = panel(ann, f"4. People Counting Result ({count} People)")

        top = np.hstack([p1, p2])
        bot = np.hstack([p3, p4])
        return np.vstack([top, bot])

    def _render_canvas_image(self):
        if self.current_display_img is None:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 10 or canvas_h <= 10:
            return

        img_h, img_w = self.current_display_img.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        disp_w = max(1, int(img_w * scale))
        disp_h = max(1, int(img_h * scale))

        resized = cv2.resize(self.current_display_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.photo_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        x_center = canvas_w // 2
        y_center = canvas_h // 2
        self.canvas.create_image(x_center, y_center, image=self.photo_img, anchor=tk.CENTER)

    def _save_results(self):
        if self.engine.orig_bgr is None or len(self.detections) == 0:
            messagebox.showinfo("Info", "No detection results to save. Run inference first.")
            return

        out_dir = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=str(PROJECT_ROOT / "runs"),
        )
        if not out_dir:
            return

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save images
        ann_img = self._render_annotated_image(show_suppressed=False)
        debug_img = self._render_annotated_image(show_suppressed=True)
        grid_img = self._render_grid_image()

        cv2.imwrite(str(out_path / "final_detections.png"), ann_img)
        cv2.imwrite(str(out_path / "debug_with_duplicates.png"), debug_img)
        cv2.imwrite(str(out_path / "summary_grid.png"), grid_img)
        if self.engine.masked_bgr is not None:
            cv2.imwrite(str(out_path / "roi_masked.png"), self.engine.masked_bgr)
        if self.engine.depth_colored_bgr is not None:
            cv2.imwrite(str(out_path / "depth_map.png"), self.engine.depth_colored_bgr)

        # Save JSON
        kept_count = sum(1 for d in self.detections if d.kept)
        meta = {
            "image_path": self.engine.current_image_path,
            "people_count": kept_count,
            "total_candidates": len(self.detections),
            "suppressed_duplicates": len(self.detections) - kept_count,
            "config": {
                "use_semantic_roi": self.use_sem_var.get(),
                "use_depth": self.use_depth_var.get(),
                "refine_boxes": self.refine_boxes_var.get(),
                "conf_thresh": self.conf_var.get(),
                "iou_thresh": self.iou_var.get(),
                "ioa_thresh": self.ioa_var.get(),
                "depth_thresh": self.depth_thresh_var.get(),
            },
            "detections": [asdict(d) for d in self.detections],
        }
        with open(out_path / "comparison_results.json", "w") as f:
            json.dump(meta, f, indent=2)

        messagebox.showinfo(
            "Saved",
            f"Successfully saved results to:\n{out_path.resolve()}\n\nPeople Count: {kept_count}",
        )


def main():
    root = tk.Tk()
    app = PipelineComparisonGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
