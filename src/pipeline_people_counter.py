"""
Demonstration pipeline for counting people in an image.

Pipeline Stages:
1. Semantic Segmentation (YOLO Semantic): Extract person Region of Interest (ROI) and mask non-ROI pixels.
2. Object Detection (YOLO Normal): Detect people within the ROI.
3. Depth Estimation (Depth Anything V2): Estimate dense relative depth map.
4. Depth-Aware Overlap Disambiguation (Depth-NMS):
   - Refines bounding boxes to the exact person pixel boundaries.
   - Compares depth distributions between overlapping bounding boxes.
   - If depth difference < depth_thresh: identifies duplicate detections of the SAME person and suppresses them.
   - If depth difference >= depth_thresh: identifies TWO DIFFERENT PEOPLE occluding each other and preserves both.
5. Counting & Visualization: Saves individual stage outputs and a multi-panel comparison image.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("PeopleCountingPipeline")


@dataclass
class Detection:
    box_id: int
    bbox: list[int]  # [x1, y1, x2, y2]
    confidence: float
    depth: float
    depth_std: float
    kept: bool = True
    suppressed_by: int | None = None
    suppression_reason: str | None = None


class PeopleCountingPipeline:
    def __init__(
        self,
        sem_model_path: str = "models/yolo26x-sem.pt",
        det_model_path: str = "models/yolo26x.pt",
        depth_checkpoint: str = "depth-anything/Depth-Anything-V2-Small-hf",
        sem_class_id: int = 11,
        det_class_id: int = 0,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.25,
        ioa_thresh: float = 0.35,
        depth_thresh: float = 0.010,
        non_roi_fill: str = "black",
        refine_boxes: bool = True,
        colormap: str = "viridis",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Models configuration
        self.sem_model_path = sem_model_path
        self.det_model_path = det_model_path
        self.depth_checkpoint = depth_checkpoint
        self.sem_class_id = sem_class_id
        self.det_class_id = det_class_id

        # Pipeline parameters
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.ioa_thresh = ioa_thresh
        self.depth_thresh = depth_thresh
        self.non_roi_fill = non_roi_fill
        self.refine_boxes = refine_boxes
        self.colormap = colormap

        # Load models
        logger.info(f"Loading YOLO semantic model from {self.sem_model_path}...")
        self.sem_model = YOLO(self.sem_model_path)

        logger.info(f"Loading YOLO detection model from {self.det_model_path}...")
        self.det_model = YOLO(self.det_model_path)

        logger.info(f"Loading Depth Anything V2 model from {self.depth_checkpoint}...")
        self.depth_processor = AutoImageProcessor.from_pretrained(self.depth_checkpoint)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.depth_checkpoint).to(self.device)
        self.depth_model.eval()

    def step1_extract_roi(self, orig_bgr: np.ndarray, imgsz: int = 1280) -> tuple[np.ndarray, np.ndarray]:
        """
        Step 1: Run YOLO Semantic model to extract person ROI mask.
        Fill non-ROI regions according to non_roi_fill strategy (default: black).
        """
        logger.info("Step 1: Running semantic segmentation for ROI extraction...")
        results = self.sem_model(
            orig_bgr,
            classes=[self.sem_class_id],
            imgsz=imgsz,
            verbose=False,
        )

        r = results[0]
        if r.semantic_mask is None:
            logger.warning("No semantic mask returned. Falling back to full image.")
            sem_mask = np.ones(orig_bgr.shape[:2], dtype=bool)
        else:
            mask_data = r.semantic_mask.data
            if isinstance(mask_data, torch.Tensor):
                mask_data = mask_data.cpu().numpy()
            sem_mask = mask_data == self.sem_class_id

        # Fill non-ROI region
        masked_img = orig_bgr.copy()
        if self.non_roi_fill == "black":
            masked_img[~sem_mask] = 0
        elif self.non_roi_fill == "white":
            masked_img[~sem_mask] = 255
        elif self.non_roi_fill == "blur":
            blurred = cv2.GaussianBlur(orig_bgr, (51, 51), 0)
            masked_img[~sem_mask] = blurred[~sem_mask]
        elif self.non_roi_fill == "none":
            pass
        else:
            logger.warning(f"Unknown non_roi_fill '{self.non_roi_fill}', defaulting to black.")
            masked_img[~sem_mask] = 0

        roi_pixel_count = int(np.sum(sem_mask))
        total_pixels = sem_mask.size
        logger.info(
            f"Step 1 Complete: ROI covers {roi_pixel_count:,} / {total_pixels:,} pixels "
            f"({roi_pixel_count / total_pixels * 100:.2f}% of image)."
        )
        return sem_mask, masked_img

    def step2_detect_people(self, masked_bgr: np.ndarray, imgsz: int = 1280) -> tuple[np.ndarray, np.ndarray]:
        """
        Step 2: Run YOLO object detector on the ROI-masked image to detect people.
        """
        logger.info("Step 2: Running object detection on ROI-masked image...")
        results = self.det_model(
            masked_bgr,
            classes=[self.det_class_id],
            conf=self.conf_thresh,
            imgsz=imgsz,
            verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            logger.info("No people detected in ROI.")
            return np.empty((0, 4), dtype=int), np.empty((0,), dtype=float)

        raw_boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        logger.info(f"Step 2 Complete: Detected {len(raw_boxes)} candidate person bounding boxes.")
        return raw_boxes, confs

    def step3_estimate_depth(self, orig_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Step 3: Run Depth Anything V2 to get high-resolution relative depth map.
        Returns:
            depth_norm: Normalized depth map in range [0, 1] (H, W).
            depth_colored: Colored depth map in BGR for visualization (H, W, 3).
        """
        logger.info("Step 3: Estimating depth map using Depth Anything V2...")
        pil_image = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.depth_processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=pil_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        depth = prediction.squeeze().cpu().numpy()
        # Normalize depth map to [0, 1]
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max > d_min:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth)

        # Generate colorized visualization
        cmap = plt.colormaps.get_cmap(self.colormap)
        depth_colored_rgb = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
        depth_colored_bgr = cv2.cvtColor(depth_colored_rgb, cv2.COLOR_RGB2BGR)

        logger.info(f"Step 3 Complete: Depth map computed with range [{d_min:.2f}, {d_max:.2f}].")
        return depth_norm, depth_colored_bgr

    def step4_refine_and_disambiguate(
        self,
        raw_boxes: np.ndarray,
        confs: np.ndarray,
        sem_mask: np.ndarray,
        depth_norm: np.ndarray,
    ) -> list[Detection]:
        """
        Step 4:
        - Refines bounding boxes to the exact person pixel boundaries using semantic mask.
        - Calculates depth statistics for each person detection.
        - Applies Depth-Aware Non-Maximum Suppression (Depth-NMS):
          Disambiguates overlapping boxes into duplicate detections (same depth plane)
          versus true occlusions (distinct depth planes).
        """
        logger.info("Step 4: Refining bounding boxes and running Depth-NMS...")
        if len(raw_boxes) == 0:
            return []

        h, w = sem_mask.shape
        refined_boxes = []
        depth_stats = []

        # Box refinement and depth extraction
        for b in raw_boxes:
            x1, y1, x2, y2 = b
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop_mask = sem_mask[y1:y2, x1:x2]
            crop_depth = depth_norm[y1:y2, x1:x2]

            if self.refine_boxes and np.any(crop_mask):
                ys, xs = np.where(crop_mask)
                rx1 = int(x1 + xs.min())
                ry1 = int(y1 + ys.min())
                rx2 = int(x1 + xs.max() + 1)
                ry2 = int(y1 + ys.max() + 1)
                person_pixels_depth = crop_depth[crop_mask]
            else:
                rx1, ry1, rx2, ry2 = int(x1), int(y1), int(x2), int(y2)
                person_pixels_depth = crop_depth

            refined_boxes.append([int(rx1), int(ry1), int(rx2), int(ry2)])
            med_depth = float(np.median(person_pixels_depth))
            std_depth = float(np.std(person_pixels_depth))
            depth_stats.append((med_depth, std_depth))

        detections = [
            Detection(
                box_id=int(idx),
                bbox=[int(coord) for coord in refined_boxes[idx]],
                confidence=float(confs[idx]),
                depth=float(depth_stats[idx][0]),
                depth_std=float(depth_stats[idx][1]),
            )
            for idx in range(len(refined_boxes))
        ]

        # Depth-NMS Algorithm:
        # Sort candidate detections by confidence score descending
        order = np.argsort(confs)[::-1]
        active_indices = list(order)

        while len(active_indices) > 0:
            curr_idx = active_indices.pop(0)
            curr_det = detections[curr_idx]

            remaining = list(active_indices)
            active_indices.clear()

            for other_idx in remaining:
                other_det = detections[other_idx]

                b1 = curr_det.bbox
                b2 = other_det.bbox

                # Calculate spatial overlap
                x1 = max(b1[0], b2[0])
                y1 = max(b1[1], b2[1])
                x2 = min(b1[2], b2[2])
                y2 = min(b1[3], b2[3])

                is_overlap = False
                overlap_metric = ""

                if x2 > x1 and y2 > y1:
                    inter_area = (x2 - x1) * (y2 - y1)
                    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    iou = inter_area / float(a1 + a2 - inter_area) if (a1 + a2 - inter_area) > 0 else 0.0
                    ioa_min = inter_area / float(min(a1, a2)) if min(a1, a2) > 0 else 0.0

                    if iou >= self.iou_thresh:
                        is_overlap = True
                        overlap_metric = f"IoU={iou:.3f} >= {self.iou_thresh}"
                    elif ioa_min >= self.ioa_thresh:
                        is_overlap = True
                        overlap_metric = f"IoA_min={ioa_min:.3f} >= {self.ioa_thresh}"

                if is_overlap:
                    depth_diff = abs(curr_det.depth - other_det.depth)
                    if depth_diff < self.depth_thresh:
                        # SAME PERSON: Occupies the same depth plane in 3D
                        other_det.kept = False
                        other_det.suppressed_by = curr_det.box_id
                        other_det.suppression_reason = (
                            f"Duplicate of Box #{curr_det.box_id} ({overlap_metric}, "
                            f"depth_diff={depth_diff:.4f} < {self.depth_thresh})"
                        )
                        logger.info(f"  [SUPPRESSED] Box #{other_det.box_id} -> {other_det.suppression_reason}")
                    else:
                        # TWO DIFFERENT PEOPLE: Distinct depth planes (foreground/background occlusion)
                        logger.info(
                            f"  [KEPT OCCLUSION] Box #{other_det.box_id} overlaps Box #{curr_det.box_id} "
                            f"({overlap_metric}) but depth_diff={depth_diff:.4f} >= {self.depth_thresh} "
                            "(distinct depth planes)."
                        )
                        active_indices.append(other_idx)
                else:
                    active_indices.append(other_idx)

        kept_count = sum(1 for d in detections if d.kept)
        suppressed_count = sum(1 for d in detections if not d.kept)
        logger.info(
            f"Step 4 Complete: Kept {kept_count} people, filtered {suppressed_count} duplicates."
        )
        return detections

    def draw_annotations(
        self,
        image_bgr: np.ndarray,
        detections: list[Detection],
        show_suppressed: bool = False,
    ) -> np.ndarray:
        """
        Draw bounding boxes and information badges on the image.
        """
        annotated = image_bgr.copy()
        kept_people = [d for d in detections if d.kept]

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            if det.kept:
                color = (0, 220, 0)  # Green for kept detections
                label = f"#{det.box_id} P:{det.confidence:.2f} D:{det.depth:.3f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label background badge
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                badge_y1 = max(0, y1 - th - baseline - 4)
                cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            elif show_suppressed:
                color = (0, 0, 255)  # Red for suppressed duplicates
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
                dup_label = f"#{det.box_id} DUP"
                cv2.putText(
                    annotated,
                    dup_label,
                    (x1, y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        # Header banner showing final person count
        count_text = f"People Count: {len(kept_people)}"
        (cw, ch), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(annotated, (15, 15), (35 + cw, 30 + ch), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            count_text,
            (25, 20 + ch),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def create_multipanel_summary(
        self,
        orig_bgr: np.ndarray,
        masked_bgr: np.ndarray,
        depth_bgr: np.ndarray,
        annotated_bgr: np.ndarray,
        people_count: int,
    ) -> np.ndarray:
        """
        Creates a 2x2 multi-panel visualization summarizing the four pipeline stages.
        """
        target_h, target_w = 540, 960

        def prepare_panel(img: np.ndarray, title: str) -> np.ndarray:
            resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            # Add top title banner
            banner = np.zeros((40, target_w, 3), dtype=np.uint8)
            cv2.putText(
                banner,
                title,
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return np.vstack([banner, resized])

        p1 = prepare_panel(orig_bgr, "1. Input Image")
        p2 = prepare_panel(masked_bgr, "2. Semantic ROI Mask (Background Extracted)")
        p3 = prepare_panel(depth_bgr, "3. Depth Map (Depth Anything V2)")
        p4 = prepare_panel(annotated_bgr, f"4. Depth-NMS Counting Result (Total: {people_count} People)")

        row1 = np.hstack([p1, p2])
        row2 = np.hstack([p3, p4])
        grid = np.vstack([row1, row2])
        return grid

    def run(self, image_path: str, output_dir: str = "runs/pipeline_output") -> dict:
        """
        Runs the end-to-end people counting pipeline.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        orig_bgr = cv2.imread(image_path)
        if orig_bgr is None:
            raise FileNotFoundError(f"Failed to read image at '{image_path}'")

        logger.info(f"Loaded image '{image_path}' ({orig_bgr.shape[1]}x{orig_bgr.shape[0]})")

        # Step 1: Semantic ROI extraction
        sem_mask, masked_bgr = self.step1_extract_roi(orig_bgr)
        cv2.imwrite(str(out_path / "1_roi_masked.png"), masked_bgr)

        # Step 2: Object Detection on ROI
        raw_boxes, confs = self.step2_detect_people(masked_bgr)

        # Step 3: Depth Estimation
        depth_norm, depth_bgr = self.step3_estimate_depth(orig_bgr)
        cv2.imwrite(str(out_path / "2_depth_map.png"), depth_bgr)

        # Step 4: Refine & Depth-NMS Overlap Disambiguation
        detections = self.step4_refine_and_disambiguate(
            raw_boxes=raw_boxes,
            confs=confs,
            sem_mask=sem_mask,
            depth_norm=depth_norm,
        )

        # Step 5: Annotations and Summary
        annotated_bgr = self.draw_annotations(orig_bgr, detections, show_suppressed=False)
        annotated_debug_bgr = self.draw_annotations(orig_bgr, detections, show_suppressed=True)
        cv2.imwrite(str(out_path / "3_final_detections.png"), annotated_bgr)
        cv2.imwrite(str(out_path / "3_detections_with_duplicates.png"), annotated_debug_bgr)

        kept_people = [d for d in detections if d.kept]
        people_count = len(kept_people)

        # Multi-panel visualization
        grid = self.create_multipanel_summary(
            orig_bgr=orig_bgr,
            masked_bgr=masked_bgr,
            depth_bgr=depth_bgr,
            annotated_bgr=annotated_bgr,
            people_count=people_count,
        )
        cv2.imwrite(str(out_path / "pipeline_summary_grid.png"), grid)

        # Save results metadata JSON
        results_meta = {
            "image_path": str(image_path),
            "image_dimensions": {"width": orig_bgr.shape[1], "height": orig_bgr.shape[0]},
            "initial_detections": len(detections),
            "suppressed_duplicates": len(detections) - people_count,
            "final_people_count": people_count,
            "parameters": {
                "conf_thresh": self.conf_thresh,
                "iou_thresh": self.iou_thresh,
                "ioa_thresh": self.ioa_thresh,
                "depth_thresh": self.depth_thresh,
                "non_roi_fill": self.non_roi_fill,
                "refine_boxes": self.refine_boxes,
            },
            "detections": [asdict(d) for d in detections],
        }
        with open(out_path / "detection_results.json", "w") as f:
            json.dump(results_meta, f, indent=2)

        print("\n" + "=" * 60)
        print(" PEOPLE COUNTING PIPELINE RESULT")
        print("=" * 60)
        print(f" Input Image:             {image_path}")
        print(f" Initial Detections:      {len(detections)}")
        print(f" Filtered Duplicates:     {len(detections) - people_count}")
        print(f" FINAL PEOPLE COUNT:      {people_count}")
        print(f" Output Saved to:         {out_path.resolve()}/")
        print("=" * 60 + "\n")

        return results_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline demonstration for counting people using YOLO Semantic ROI, YOLO Detection, and Depth Anything V2."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/images/classroom.png",
        help="Path to input image.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/pipeline_output",
        help="Directory to save pipeline outputs and visualizations.",
    )
    parser.add_argument(
        "--sem-model",
        type=str,
        default="models/yolo26x-sem.pt",
        help="Path to YOLO semantic segmentation model.",
    )
    parser.add_argument(
        "--det-model",
        type=str,
        default="models/yolo26x.pt",
        help="Path to YOLO object detection model.",
    )
    parser.add_argument(
        "--depth-checkpoint",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="HuggingFace checkpoint for Depth Anything V2.",
    )
    parser.add_argument(
        "--sem-class",
        type=int,
        default=11,
        help="Class ID for person in the semantic segmentation model (default: 11 for yolo26x-sem).",
    )
    parser.add_argument(
        "--det-class",
        type=int,
        default=0,
        help="Class ID for person in the object detection model (default: 0).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO object detector.",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.25,
        help="IoU threshold for box overlap detection.",
    )
    parser.add_argument(
        "--ioa-thresh",
        type=float,
        default=0.35,
        help="IoA (Intersection over Minimum Area) threshold for nested box overlap.",
    )
    parser.add_argument(
        "--depth-thresh",
        type=float,
        default=0.010,
        help="Normalized depth difference threshold (smaller = duplicate same person, larger = different occluded people).",
    )
    parser.add_argument(
        "--non-roi-fill",
        type=str,
        default="black",
        choices=["black", "white", "blur", "none"],
        help="Fill strategy for non-ROI pixels.",
    )
    parser.add_argument(
        "--no-refine-boxes",
        action="store_true",
        help="Disable tightening bounding boxes with semantic person masks.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        help="Colormap for depth visualization (viridis, inferno, plasma, turbo).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on ('cuda' or 'cpu').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = PeopleCountingPipeline(
        sem_model_path=args.sem_model,
        det_model_path=args.det_model,
        depth_checkpoint=args.depth_checkpoint,
        sem_class_id=args.sem_class,
        det_class_id=args.det_class,
        conf_thresh=args.conf,
        iou_thresh=args.iou_thresh,
        ioa_thresh=args.ioa_thresh,
        depth_thresh=args.depth_thresh,
        non_roi_fill=args.non_roi_fill,
        refine_boxes=not args.no_refine_boxes,
        colormap=args.colormap,
        device=args.device,
    )
    pipeline.run(image_path=args.image, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
