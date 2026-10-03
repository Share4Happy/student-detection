from dataclasses import dataclass

import numpy as np

from student_detection.common import NumpyImage

from .depth_estimation import DepthEstimationModel
from .object_detection import ObjectDetectionModel
from .semantic_segmentation import SemanticSegmentationModel


@dataclass
class PipelineDetection:
    box_id: int
    bbox: list[int]  # [x1, y1, x2, y2]
    confidence: float
    depth: float
    depth_std: float
    kept: bool = True
    suppressed_by: int | None = None
    suppression_reason: str | None = None


@dataclass
class PipelineResult:
    detections: list[PipelineDetection]
    people_count: int
    sem_mask: np.ndarray
    depth_map: np.ndarray


class DetectorPipeline:
    def __init__(
        self,
        sem_model: SemanticSegmentationModel,
        obj_model: ObjectDetectionModel,
        dep_model: DepthEstimationModel
    ):
        self.sem_model = sem_model
        self.obj_model = obj_model
        self.dep_model = dep_model

    def predict(
        self,
        image: NumpyImage,
        conf_threshold: float = 0.25,
        iou_thresh: float = 0.25,
        ioa_thresh: float = 0.35,
        depth_thresh: float = 0.010,
        refine_boxes: bool = True,
        non_roi_fill: int = 0,
        imgsz: int = 1280,
    ) -> PipelineResult:
        """Run the multi-stage person detection and depth-aware counting pipeline.

        This method executes four sequential operations:
        1. Extract the person Region of Interest (ROI) mask with semantic segmentation.
        2. Detect candidate person bounding boxes on the ROI-masked image.
        3. Estimate a dense relative depth map.
        4. Refine box boundaries and apply Depth-NMS to filter duplicate detections.

        Args:
            `image`: Input image as a NumpyImage array of shape (H, W, 3). `conf_threshold`: Minimum confidence score to retain a detection box.
            `iou_thresh`: IoU overlap threshold to identify candidate duplicates.
            `ioa_thresh`: IoA (Intersection-over-Minimum-Area) threshold for nested boxes.
            `depth_thresh`: Normalized depth threshold. Boxes with a depth difference
                smaller than this value are treated as the same person.
            `refine_boxes`: When True, shrink bounding boxes to exact person pixel masks.
            `non_roi_fill`: Pixel value used to fill non-ROI background regions (default: 0).
            `imgsz`: Input resolution passed to segmentation and detection models.

        Returns:
            `PipelineResult`: An object containing final detections, total people count,
                the semantic ROI mask, and the normalized depth map.
        """
        # Step 1: Semantic ROI extraction
        sem_mask, masked_img = self.sem_model.extract_roi(
            image,
            fill_value=non_roi_fill,
            imgsz=imgsz
        )

        # Step 2: Object detection on masked image
        candidate_boxes = self.obj_model.predict(
            NumpyImage(masked_img),
            conf_threshold=conf_threshold,
            imgsz=imgsz
        )

        # Step 3: Depth estimation
        depth_res = self.dep_model.predict(image)
        depth_norm = depth_res.get_normalized()

        if not candidate_boxes:
            return PipelineResult(
                detections=[],
                people_count=0,
                sem_mask=sem_mask,
                depth_map=depth_norm,
            )

        # Step 4a: Box refinement & depth statistics
        h, w = sem_mask.shape[:2]
        refined_boxes: list[list[int]] = []
        depth_stats: list[tuple[float, float]] = []

        for b in candidate_boxes:
            x1 = max(0, b.x1)
            y1 = max(0, b.y1)
            x2 = min(w, b.x2)
            y2 = min(h, b.y2)

            crop_mask = sem_mask[y1:y2, x1:x2]
            crop_depth = depth_norm[y1:y2, x1:x2]

            if refine_boxes and np.any(crop_mask):
                ys, xs = np.where(crop_mask)
                rx1 = int(x1 + xs.min())
                ry1 = int(y1 + ys.min())
                rx2 = int(x1 + xs.max() + 1)
                ry2 = int(y1 + ys.max() + 1)
                person_depths = crop_depth[crop_mask]
            else:
                rx1, ry1, rx2, ry2 = int(x1), int(y1), int(x2), int(y2)
                person_depths = crop_depth

            refined_boxes.append([rx1, ry1, rx2, ry2])
            med = float(np.median(person_depths)) if len(person_depths) > 0 else 0.0
            std = float(np.std(person_depths)) if len(person_depths) > 0 else 0.0
            depth_stats.append((med, std))

        detections = [
            PipelineDetection(
                box_id=idx,
                bbox=refined_boxes[idx],
                confidence=float(candidate_boxes[idx].confidence),
                depth=depth_stats[idx][0],
                depth_std=depth_stats[idx][1],
            )
            for idx in range(len(refined_boxes))
        ]

        # Step 4b: Depth-NMS
        order = np.argsort([b.confidence for b in candidate_boxes])[::-1]
        active_indices = list(order)

        while active_indices:
            curr_idx = active_indices.pop(0)
            curr_det = detections[curr_idx]
            remaining = list(active_indices)
            active_indices.clear()

            for other_idx in remaining:
                other_det = detections[other_idx]
                b1, b2 = curr_det.bbox, other_det.bbox

                # Calculate 2D intersection
                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])

                is_overlap = False
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    iou = inter_area / float(a1 + a2 - inter_area) if (a1 + a2 - inter_area) > 0 else 0.0
                    ioa_min = inter_area / float(min(a1, a2)) if min(a1, a2) > 0 else 0.0

                    if iou >= iou_thresh or ioa_min >= ioa_thresh:
                        is_overlap = True

                if is_overlap:
                    depth_diff = abs(curr_det.depth - other_det.depth)
                    if depth_diff < depth_thresh:
                        # Duplicate detection of same person
                        other_det.kept = False
                        other_det.suppressed_by = curr_det.box_id
                        other_det.suppression_reason = (
                            f"Duplicate of #{curr_det.box_id} (depth_diff={depth_diff:.4f} < {depth_thresh})"
                        )
                    else:
                        # Distinct occluding people
                        active_indices.append(other_idx)
                else:
                    active_indices.append(other_idx)

        people_count = sum(1 for d in detections if d.kept)

        return PipelineResult(
            detections=detections,
            people_count=people_count,
            sem_mask=sem_mask,
            depth_map=depth_norm,
        )
