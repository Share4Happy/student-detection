from __future__ import annotations

import numpy as np

from student_detection.common import NumpyImage

from .depth_estimation import DepthEstimationModel
from .models import PipelineDetection, PipelineResult
from .object_detection import BoundingBox, ObjectDetectionModel
from .policy import (
    ConfidencePolicy,
    DepthThresholdPolicy,
    IoAThresholdPolicy,
    IoUThresholdPolicy,
)
from .semantic_segmentation import SemanticSegmentationModel


class DetectorPipeline:
    """Execute the multi-stage person detection and depth-aware counting pipeline.

    Coordinates semantic ROI masking, object detection, metric depth estimation,
    and depth-aware non-maximum suppression.

    Attributes:
        `sem_model`: Semantic segmentation model instance for ROI extraction.
        `obj_model`: Object detection model instance for candidate localization.
        `dep_model`: Depth estimation model instance for relative depth prediction.

    Methods:
        `predict(...)`: Run the end-to-end multi-stage detection pipeline on an image.

    Examples:
        >>> # pipeline = DetectorPipeline(sem_model, obj_model, dep_model)
        >>> # result = pipeline.predict(image, conf_policy, iou_policy, ioa_policy, depth_policy)
        ... pass
    """

    def __init__(
        self,
        sem_model: SemanticSegmentationModel,
        obj_model: ObjectDetectionModel,
        dep_model: DepthEstimationModel,
    ) -> None:
        """Initialize the pipeline stages with pre-instantiated models.

        Args:
            `sem_model`: Semantic segmentation model used for ROI isolation.
            `obj_model`: Object detection model used for bounding box proposal.
            `dep_model`: Monocular depth estimation model used for depth mapping.

        Examples:
            >>> # pipeline = DetectorPipeline(sem_model, obj_model, dep_model)
            ... pass
        """
        self.sem_model = sem_model
        self.obj_model = obj_model
        self.dep_model = dep_model

    def _refine_and_extract_detections(
        self,
        candidate_boxes: list[BoundingBox],
        sem_mask: np.ndarray,
        depth_norm: np.ndarray,
        refine_boxes: bool = True,
    ) -> list[PipelineDetection]:
        """Snap candidate bounding boxes to person pixels and extract depth metrics.

        Args:
            `candidate_boxes`: Initial bounding box proposals from the detector.
            `sem_mask`: Binary segmentation mask where non-zero denotes foreground.
            `depth_norm`: Normalized relative depth array of shape `(H, W)`.
            `refine_boxes`: Flag to shrink boxes to non-zero pixel bounds. Defaults to `True`.

        Returns:
            List of `PipelineDetection` candidates with populated depth statistics.

        Examples:
            >>> # detections = pipeline._refine_and_extract_detections(boxes, mask, depth)
            ... pass
        """
        h, w = sem_mask.shape[:2]
        detections: list[PipelineDetection] = []

        for idx, b in enumerate(candidate_boxes):
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

            med_depth = float(np.median(person_depths)) if len(person_depths) > 0 else 0.0
            std_depth = float(np.std(person_depths)) if len(person_depths) > 0 else 0.0

            detections.append(
                PipelineDetection(
                    box_id=idx,
                    bbox=[rx1, ry1, rx2, ry2],
                    confidence=float(b.confidence),
                    depth=med_depth,
                    depth_std=std_depth,
                )
            )

        return detections

    def _apply_depth_nms(
        self,
        candidates: list[PipelineDetection],
        iou_policy: IoUThresholdPolicy,
        ioa_policy: IoAThresholdPolicy,
        depth_policy: DepthThresholdPolicy,
    ) -> list[PipelineDetection]:
        """Apply depth-aware non-maximum suppression using dynamic threshold policies.

        Args:
            `candidates`: Sequence of `PipelineDetection` candidates to filter.
            `iou_policy`: Policy calculating pairwise IoU overlap cutoff.
            `ioa_policy`: Policy calculating pairwise IoA containment cutoff.
            `depth_policy`: Policy calculating minimum relative depth difference cutoff.

        Returns:
            List of `PipelineDetection` objects with updated `kept` and suppression flags.

        Examples:
            >>> # filtered = pipeline._apply_depth_nms(candidates, iou_pol, ioa_pol, depth_pol)
            ... pass
        """
        if not candidates:
            return []

        # Re-index box IDs for clarity
        for new_id, det in enumerate(candidates):
            det.box_id = new_id
            det.kept = True
            det.suppressed_by = None
            det.suppression_reason = None

        order = np.argsort([d.confidence for d in candidates])[::-1]
        active_indices = list(order)

        while active_indices:
            curr_idx = active_indices.pop(0)
            curr_det = candidates[curr_idx]
            remaining = list(active_indices)
            active_indices.clear()

            for other_idx in remaining:
                other_det = candidates[other_idx]
                b1, b2 = curr_det.bbox, other_det.bbox

                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])

                is_overlap = False
                metric_info = ""

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    iou = inter_area / float(a1 + a2 - inter_area) if (a1 + a2 - inter_area) > 0 else 0.0
                    ioa_min = inter_area / float(min(a1, a2)) if min(a1, a2) > 0 else 0.0

                    active_iou_thresh = iou_policy(curr_det, other_det)
                    active_ioa_thresh = ioa_policy(curr_det, other_det)

                    if iou >= active_iou_thresh:
                        is_overlap = True
                        metric_info = f"IoU={iou:.2f} >= {active_iou_thresh:.2f}"
                    elif ioa_min >= active_ioa_thresh:
                        is_overlap = True
                        metric_info = f"IoA_min={ioa_min:.2f} >= {active_ioa_thresh:.2f}"

                if is_overlap:
                    depth_diff = abs(curr_det.depth - other_det.depth)
                    active_depth_thresh = depth_policy(curr_det, other_det)

                    if depth_diff < active_depth_thresh:
                        other_det.kept = False
                        other_det.suppressed_by = curr_det.box_id
                        other_det.suppression_reason = (
                            f"Duplicate of #{curr_det.box_id} ({metric_info}, "
                            f"depth_diff={depth_diff:.4f} < {active_depth_thresh:.4f})"
                        )
                    else:
                        active_indices.append(other_idx)
                else:
                    active_indices.append(other_idx)

        return candidates

    def predict(
        self,
        image: NumpyImage,
        conf_policy: ConfidencePolicy,
        iou_policy: IoUThresholdPolicy,
        ioa_policy: IoAThresholdPolicy,
        depth_policy: DepthThresholdPolicy,
        refine_boxes: bool = True,
        non_roi_fill: int = 0,
        imgsz: int = 1280,
        conf_floor: float = 0.01,
    ) -> PipelineResult:
        """Run the multi-stage person detection and depth-aware counting pipeline.

        Args:
            `image`: Input image as a `NumpyImage` array of shape `(H, W, 3)`.
            `conf_policy`: Policy generating required confidence score per detection.
            `iou_policy`: Policy generating IoU overlap threshold between pairs.
            `ioa_policy`: Policy generating IoA containment threshold between pairs.
            `depth_policy`: Policy generating depth difference threshold between pairs.
            `refine_boxes`: Flag to shrink bounding boxes to exact person pixel masks. Defaults to `True`.
            `non_roi_fill`: Pixel value used to fill non-ROI background regions. Defaults to `0`.
            `imgsz`: Input resolution passed to segmentation and detection models. Defaults to `1280`.
            `conf_floor`: Negligible detector confidence floor to capture raw candidates. Defaults to `0.01`.

        Returns:
            `PipelineResult` containing final detections, people count, and stage maps.

        Examples:
            >>> # result = pipeline.predict(img, conf_pol, iou_pol, ioa_pol, depth_pol)
            ... pass
        """
        # Step 1: Semantic ROI extraction
        sem_mask, masked_img = self.sem_model.extract_roi(
            image,
            fill_value=non_roi_fill,
            imgsz=imgsz,
        )

        # Step 2: Object detection with floor threshold to preserve all real candidates
        raw_boxes = self.obj_model.predict(
            NumpyImage(masked_img),
            conf_threshold=conf_floor,
            imgsz=imgsz,
        )

        # Step 3: Depth estimation
        depth_res = self.dep_model.predict(image)
        depth_norm = depth_res.get_normalized()

        if not raw_boxes:
            return PipelineResult(
                detections=[],
                people_count=0,
                sem_mask=sem_mask,
                depth_map=depth_norm,
            )

        # Step 4a: Box refinement & depth statistics extraction
        all_detections = self._refine_and_extract_detections(
            candidate_boxes=raw_boxes,
            sem_mask=sem_mask,
            depth_norm=depth_norm,
            refine_boxes=refine_boxes,
        )

        # Step 4b: Dynamic confidence filtering using person features (depth, scale)
        candidates = [
            d for d in all_detections
            if d.confidence >= conf_policy(d)
        ]

        if not candidates:
            return PipelineResult(
                detections=[],
                people_count=0,
                sem_mask=sem_mask,
                depth_map=depth_norm,
            )

        # Step 4c: Depth-NMS disambiguation using threshold policies
        final_detections = self._apply_depth_nms(
            candidates=candidates,
            iou_policy=iou_policy,
            ioa_policy=ioa_policy,
            depth_policy=depth_policy,
        )

        people_count = sum(1 for d in final_detections if d.kept)

        return PipelineResult(
            detections=final_detections,
            people_count=people_count,
            sem_mask=sem_mask,
            depth_map=depth_norm,
        )


__all__ = [
    "DetectorPipeline",
    "PipelineDetection",
    "PipelineResult",
]
