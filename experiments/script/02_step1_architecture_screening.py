"""
Step 1: Architecture Comparison with a Fixed Object Detection Model (YOLO26x).
Compares:
1. Bare Object Detection (YOLO26x alone, standard NMS)
2. Pipeline with Semantic ROI + Monocular Depth (Fixed Policies)
3. Pipeline with Semantic ROI + Monocular Depth (Dynamic Policies)
4. Pipeline with Monocular Depth Only (EmptyYOLOSemanticPerson, Fixed Policies)
5. Pipeline with Monocular Depth Only (EmptyYOLOSemanticPerson, Dynamic Policies)

Performs a wide hyperparameter search for each architecture variant.
Saves results into experiments/output/step1_architecture_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

# Ensure experiments/script is importable
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from empty_semantic import EmptyYOLOSemanticPerson
from student_detection.core.models import PipelineDetection
from student_detection.core.policy import (
    FixedConfidencePolicy,
    DistanceScaledConfidencePolicy,
    FixedIoUThresholdPolicy,
    DistanceScaledIoUPolicy,
    FixedIoAThresholdPolicy,
    FixedDepthThresholdPolicy,
    DistanceDecayDepthPolicy,
)


def run_pipeline_depth_nms(candidates, iou_policy, ioa_policy, depth_policy):
    if not candidates:
        return []

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
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = inter_area / float(a1 + a2 - inter_area) if (a1 + a2 - inter_area) > 0 else 0.0
                ioa_min = inter_area / float(min(a1, a2)) if min(a1, a2) > 0 else 0.0

                active_iou = iou_policy(curr_det, other_det)
                active_ioa = ioa_policy(curr_det, other_det)

                if iou >= active_iou or ioa_min >= active_ioa:
                    is_overlap = True

            if is_overlap:
                depth_diff = abs(curr_det.depth - other_det.depth)
                active_depth = depth_policy(curr_det, other_det)
                if depth_diff < active_depth:
                    other_det.kept = False
                    other_det.suppressed_by = curr_det.box_id
                else:
                    active_indices.append(other_idx)
            else:
                active_indices.append(other_idx)

    return [d for d in candidates if d.kept]


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    diff = y_pred - y_true
    abs_diff = np.abs(diff)

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mbe = float(np.mean(diff))
    counting_accuracy = float(np.mean(np.maximum(0.0, 1.0 - (abs_diff / np.maximum(y_true, 1.0)))) * 100.0)
    exact_match = float(np.mean(abs_diff == 0) * 100.0)
    within_1 = float(np.mean(abs_diff <= 1) * 100.0)
    within_2 = float(np.mean(abs_diff <= 2) * 100.0)

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mbe": round(mbe, 3),
        "counting_accuracy": round(counting_accuracy, 2),
        "exact_match_pct": round(exact_match, 2),
        "within_1_pct": round(within_1, 2),
        "within_2_pct": round(within_2, 2),
    }


def main():
    features_path = Path("experiments/output/test_features.json")
    if not features_path.exists():
        features_path = Path("experiments/test_features.json")

    with open(features_path) as f:
        data = json.load(f)

    image_ids = list(data.keys())
    gt_list = [data[i]["gt_count"] for i in image_ids]
    print(f"Running Step 1 Architecture Comparison on {len(image_ids)} images with fixed YOLO26x detector...")

    all_experiments = []

    # ----------------------------------------------------
    # Architecture 1: Bare Object Detection (YOLO26x alone) - 5 configurations
    # ----------------------------------------------------
    bare_configs = [
        (0.10, "Bare YOLO26x - Aggressive Recall (conf=0.10)"),
        (0.15, "Bare YOLO26x - Moderate Recall (conf=0.15)"),
        (0.20, "Bare YOLO26x - Balanced (conf=0.20)"),
        (0.22, "Bare YOLO26x - Optimal Peak (conf=0.22)"),
        (0.28, "Bare YOLO26x - Conservative (conf=0.28)"),
    ]
    for conf, label in bare_configs:
        preds = [sum(1 for b in data[i]["bare_yolo26x"] if b["confidence"] >= conf) for i in image_ids]
        m = compute_metrics(gt_list, preds)
        all_experiments.append({
            "architecture": "Arch 1: Bare Object Detection",
            "variant_name": label,
            "sem_model": "None",
            "depth_model": "None",
            "policy_type": "Standard 2D NMS",
            "conf_thresh": conf,
            "iou_thresh": 0.70,
            "ioa_thresh": "-",
            "depth_thresh": "-",
            "params_summary": f"conf={conf:.2f}, IoU=0.70",
            **m,
        })

    # ----------------------------------------------------
    # Architecture 2: Pipeline with Sem-ROI + Depth (Fixed) - 5 configurations
    # ----------------------------------------------------
    sem_fixed_configs = [
        (0.15, 0.25, 0.35, 0.010, "Pipeline Sem+Depth Fixed - Deep Recall (conf=0.15, iou=0.25, dt=0.010)"),
        (0.20, 0.25, 0.35, 0.010, "Pipeline Sem+Depth Fixed - Balanced (conf=0.20, iou=0.25, dt=0.010)"),
        (0.20, 0.45, 0.60, 0.008, "Pipeline Sem+Depth Fixed - Tight Overlap (conf=0.20, iou=0.45, dt=0.008)"),
        (0.22, 0.50, 0.70, 0.006, "Pipeline Sem+Depth Fixed - Standard Fixed (conf=0.22, iou=0.50, dt=0.006)"),
        (0.25, 0.25, 0.35, 0.010, "Pipeline Sem+Depth Fixed - Conservative (conf=0.25, iou=0.25, dt=0.010)"),
    ]
    for conf, iou, ioa, depth, label in sem_fixed_configs:
        conf_pol = FixedConfidencePolicy(conf)
        iou_pol = FixedIoUThresholdPolicy(iou)
        ioa_pol = FixedIoAThresholdPolicy(ioa)
        depth_pol = FixedDepthThresholdPolicy(depth)

        preds = []
        for i in image_ids:
            cands = [
                PipelineDetection(d["box_id"], d["bbox"], d["confidence"], d["depth"], d["depth_std"])
                for d in data[i]["pipeline_with_sem"]
                if d["confidence"] >= conf
            ]
            kept = run_pipeline_depth_nms(cands, iou_pol, ioa_pol, depth_pol)
            preds.append(len(kept))

        m = compute_metrics(gt_list, preds)
        all_experiments.append({
            "architecture": "Arch 2: Pipeline Sem-ROI + Depth (Fixed)",
            "variant_name": label,
            "sem_model": "YOLOSemanticPerson (Cityscapes)",
            "depth_model": "DepthAnythingV2",
            "policy_type": "Fixed",
            "conf_thresh": conf,
            "iou_thresh": iou,
            "ioa_thresh": ioa,
            "depth_thresh": depth,
            "params_summary": f"conf={conf:.2f}, IoU={iou:.2f}, IoA={ioa:.2f}, dt={depth:.3f}",
            **m,
        })

    # ----------------------------------------------------
    # Architecture 3: Pipeline with Sem-ROI + Depth (Dynamic) - 5 configurations
    # ----------------------------------------------------
    dyn_configs = [
        (0.18, 0.12, 0.50, 0.50, 0.10, 0.35, 0.005, 0.002, 0.50, "Dyn-L (Deep Recall 0.18/0.12)"),
        (0.20, 0.16, 0.50, 0.45, 0.15, 0.35, 0.006, 0.002, 0.50, "Dyn-I (Classroom Tuned 0.20/0.16)"),
        (0.22, 0.12, 0.50, 0.18, 0.20, 0.35, 0.018, 0.005, 0.60, "Dyn-C (Balanced 0.22/0.12)"),
        (0.25, 0.15, 0.60, 0.20, 0.15, 0.35, 0.015, 0.004, 0.70, "Dyn-A (Scale 0.25/0.15)"),
        (0.25, 0.18, 0.55, 0.45, 0.15, 0.35, 0.008, 0.002, 0.50, "Dyn-G (Conservative 0.25/0.18)"),
    ]
    for n_c, f_c, d_cut, b_iou, boost_iou, ioa, b_d, min_d, dec_d, label in dyn_configs:
        conf_pol = DistanceScaledConfidencePolicy(near_threshold=n_c, far_threshold=f_c, depth_cutoff=d_cut)
        iou_pol = DistanceScaledIoUPolicy(base_threshold=b_iou, distance_boost=boost_iou)
        ioa_pol = FixedIoAThresholdPolicy(ioa)
        depth_pol = DistanceDecayDepthPolicy(base_threshold=b_d, min_threshold=min_d, decay_factor=dec_d)

        preds = []
        for i in image_ids:
            cands = [
                PipelineDetection(d["box_id"], d["bbox"], d["confidence"], d["depth"], d["depth_std"])
                for d in data[i]["pipeline_with_sem"]
            ]
            cands = [c for c in cands if c.confidence >= conf_pol(c)]
            kept = run_pipeline_depth_nms(cands, iou_pol, ioa_pol, depth_pol)
            preds.append(len(kept))

        m = compute_metrics(gt_list, preds)
        all_experiments.append({
            "architecture": "Arch 3: Pipeline Sem-ROI + Depth (Dynamic)",
            "variant_name": f"Pipeline Sem+Depth Dynamic - {label}",
            "sem_model": "YOLOSemanticPerson (Cityscapes)",
            "depth_model": "DepthAnythingV2",
            "policy_type": "Dynamic",
            "conf_thresh": f"{n_c:.2f}/{f_c:.2f}",
            "iou_thresh": f"{b_iou:.2f}+{boost_iou:.2f}*d",
            "ioa_thresh": ioa,
            "depth_thresh": f"b={b_d:.3f},m={min_d:.3f}",
            "params_summary": f"Near/Far={n_c:.2f}/{f_c:.2f}, BaseIoU={b_iou:.2f}, BaseDt={b_d:.3f}",
            **m,
        })

    # ----------------------------------------------------
    # Architecture 4: Pipeline Depth Only (EmptyYOLOSemantic, Fixed) - 5 configurations
    # ----------------------------------------------------
    fixed_grid_empty_sem = [
        (0.08, 0.55, 0.70, 0.010, "Pipeline Depth-Only Fixed - Aggressive Recall (conf=0.08, iou=0.55, dt=0.010)"),
        (0.12, 0.55, 0.70, 0.006, "Pipeline Depth-Only Fixed - Balanced Peak (conf=0.12, iou=0.55, dt=0.006)"),
        (0.15, 0.55, 0.70, 0.006, "Pipeline Depth-Only Fixed - Standard (conf=0.15, iou=0.55, dt=0.006)"),
        (0.18, 0.55, 0.70, 0.006, "Pipeline Depth-Only Fixed - High Precision (conf=0.18, iou=0.55, dt=0.006)"),
        (0.22, 0.50, 0.70, 0.006, "Pipeline Depth-Only Fixed - Conservative (conf=0.22, iou=0.50, dt=0.006)"),
    ]
    for conf, iou, ioa, depth, label in fixed_grid_empty_sem:
        conf_pol = FixedConfidencePolicy(conf)
        iou_pol = FixedIoUThresholdPolicy(iou)
        ioa_pol = FixedIoAThresholdPolicy(ioa)
        depth_pol = FixedDepthThresholdPolicy(depth)

        preds = []
        for i in image_ids:
            cands = [
                PipelineDetection(d["box_id"], d["bbox"], d["confidence"], d["depth"], d["depth_std"])
                for d in data[i]["pipeline_no_sem"]
                if d["confidence"] >= conf
            ]
            kept = run_pipeline_depth_nms(cands, iou_pol, ioa_pol, depth_pol)
            preds.append(len(kept))

        m = compute_metrics(gt_list, preds)
        all_experiments.append({
            "architecture": "Arch 4: Pipeline Depth Only (EmptySem, Fixed)",
            "variant_name": label,
            "sem_model": "EmptyYOLOSemanticPerson",
            "depth_model": "DepthAnythingV2",
            "policy_type": "Fixed",
            "conf_thresh": conf,
            "iou_thresh": iou,
            "ioa_thresh": ioa,
            "depth_thresh": depth,
            "params_summary": f"conf={conf:.2f}, IoU={iou:.2f}, IoA={ioa:.2f}, dt={depth:.3f}",
            **m,
        })

    # ----------------------------------------------------
    # Architecture 5: Pipeline Depth Only (EmptyYOLOSemantic, Dynamic) - 5 configurations
    # ----------------------------------------------------
    dyn_configs_empty = [
        (0.16, 0.10, 0.50, 0.55, 0.10, 0.70, 0.005, 0.002, 0.50, "Dyn-M (Aggressive Deep Recall 0.16/0.10)"),
        (0.16, 0.12, 0.50, 0.55, 0.10, 0.70, 0.005, 0.002, 0.50, "Dyn-L (Deep Recall 0.16/0.12)"),
        (0.18, 0.14, 0.50, 0.50, 0.15, 0.70, 0.006, 0.002, 0.50, "Dyn-K (High Recall 0.18/0.14)"),
        (0.20, 0.16, 0.50, 0.45, 0.15, 0.70, 0.006, 0.002, 0.50, "Dyn-I (Dense IoU 0.20/0.16)"),
        (0.22, 0.18, 0.55, 0.45, 0.15, 0.70, 0.008, 0.002, 0.50, "Dyn-G (Conservative 0.22/0.18)"),
    ]
    for n_c, f_c, d_cut, b_iou, boost_iou, ioa, b_d, min_d, dec_d, label in dyn_configs_empty:
        conf_pol = DistanceScaledConfidencePolicy(near_threshold=n_c, far_threshold=f_c, depth_cutoff=d_cut)
        iou_pol = DistanceScaledIoUPolicy(base_threshold=b_iou, distance_boost=boost_iou)
        ioa_pol = FixedIoAThresholdPolicy(ioa)
        depth_pol = DistanceDecayDepthPolicy(base_threshold=b_d, min_threshold=min_d, decay_factor=dec_d)

        preds = []
        for i in image_ids:
            cands = [
                PipelineDetection(d["box_id"], d["bbox"], d["confidence"], d["depth"], d["depth_std"])
                for d in data[i]["pipeline_no_sem"]
            ]
            cands = [c for c in cands if c.confidence >= conf_pol(c)]
            kept = run_pipeline_depth_nms(cands, iou_pol, ioa_pol, depth_pol)
            preds.append(len(kept))

        m = compute_metrics(gt_list, preds)
        all_experiments.append({
            "architecture": "Arch 5: Pipeline Depth Only (EmptySem, Dynamic)",
            "variant_name": f"Pipeline Depth-Only Dynamic - {label}",
            "sem_model": "EmptyYOLOSemanticPerson",
            "depth_model": "DepthAnythingV2",
            "policy_type": "Dynamic",
            "conf_thresh": f"{n_c:.2f}/{f_c:.2f}",
            "iou_thresh": f"{b_iou:.2f}+{boost_iou:.2f}*d",
            "ioa_thresh": ioa,
            "depth_thresh": f"b={b_d:.3f},m={min_d:.3f}",
            "params_summary": f"Near/Far={n_c:.2f}/{f_c:.2f}, BaseIoU={b_iou:.2f}, BaseDt={b_d:.3f}",
            **m,
        })

    # Sort by accuracy descending, then MAE ascending
    all_experiments.sort(key=lambda x: (-x["counting_accuracy"], x["mae"]))

    # Add delta_acc relative to rank 1
    top_acc = all_experiments[0]["counting_accuracy"]
    for idx, exp in enumerate(all_experiments):
        exp["rank"] = idx + 1
        if idx == 0:
            exp["delta_acc"] = "-"
            exp["delta_acc_num"] = 0.0
        else:
            diff = exp["counting_accuracy"] - top_acc
            exp["delta_acc"] = f"{diff:.2f}%"
            exp["delta_acc_num"] = round(diff, 2)

    out_file = Path("experiments/output/step1_architecture_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_experiments, f, indent=2)

    print(f"\nStep 1 Completed: Evaluated exactly {len(all_experiments)} configurations (5 per architecture).")
    print(f"Results saved to {out_file}\n")

    print("=" * 115)
    print("RANKED LEADERBOARD (ALL 25 CONFIGURATIONS - SHALLOW COMPARISON)")
    print("=" * 115)
    print(f"{'Rank':<5} | {'Architecture':<26} | {'Accuracy':<9} | {'Delta Acc':<10} | {'MAE':<6} | {'Params Summary':<40}")
    print("-" * 115)
    for e in all_experiments:
        print(f"{e['rank']:<5} | {e['architecture'][:26]:<26} | {e['counting_accuracy']:>6.2f}%   | {e['delta_acc']:<10} | {e['mae']:<6.2f} | {e['params_summary']:<40}")
    print("=" * 115)


if __name__ == "__main__":
    main()
