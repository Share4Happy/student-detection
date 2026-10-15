"""
Deep Search & Evaluation across All 10 YOLO Models on the Winning Pipeline Architecture:
- YOLO26 Family: yolo26n, yolo26s, yolo26m, yolo26l, yolo26x
- YOLO11 Family: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

Evaluates:
1. Bare Model Baseline (Standard 2D NMS across confidences)
2. Winning Architecture: Pipeline Depth-Only (EmptyYOLOSemanticPerson + Depth-NMS across full hyperparameter grid)
3. Dynamic Policies

Outputs:
- Top 30 configurations overall
- experiments/output/all_10_models_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

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
    features_path = Path("experiments/output/test_features_all10.json")
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} not found. Run extract_all_10_models.py first.")

    with open(features_path) as f:
        data = json.load(f)

    image_ids = list(data.keys())
    gt_list = [data[i]["gt_count"] for i in image_ids]

    models_info = [
        ("yolo26n", "YOLO26n (Nano, 5.3MB)", "YOLO26"),
        ("yolo26s", "YOLO26s (Small, 20MB)", "YOLO26"),
        ("yolo26m", "YOLO26m (Medium, 43MB)", "YOLO26"),
        ("yolo26l", "YOLO26l (Large, 51MB)", "YOLO26"),
        ("yolo26x", "YOLO26x (X-large, 114MB)", "YOLO26"),
        ("yolo11n", "YOLO11n (Nano, 5.4MB)", "YOLO11"),
        ("yolo11s", "YOLO11s (Small, 19MB)", "YOLO11"),
        ("yolo11m", "YOLO11m (Medium, 39MB)", "YOLO11"),
        ("yolo11l", "YOLO11l (Large, 50MB)", "YOLO11"),
        ("yolo11x", "YOLO11x (X-large, 110MB)", "YOLO11"),
    ]

    all_experiments = []

    confs = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25]
    ious = [0.45, 0.55, 0.65, 0.75]
    depth_diffs = [0.003, 0.006, 0.010, 0.015]

    for m_key, m_label, m_fam in models_info:
        print(f"Evaluating {m_label}...")

        # 1. Bare baseline
        for c in [0.08, 0.10, 0.15, 0.20, 0.25]:
            preds = [sum(1 for b in data[i]["models"][m_key] if b["confidence"] >= c) for i in image_ids]
            m = compute_metrics(gt_list, preds)
            all_experiments.append({
                "model_key": m_key,
                "model_label": m_label,
                "family": m_fam,
                "architecture": "Bare Object Detection",
                "variant_name": f"{m_label} Bare (conf={c:.2f})",
                "conf_thresh": c,
                "iou_thresh": None,
                "depth_thresh": None,
                **m,
            })

        # 2. Winning Architecture: Pipeline Depth-Only (EmptyYOLOSemanticPerson + Depth-NMS)
        for c in confs:
            conf_pol = FixedConfidencePolicy(c)
            for iou in ious:
                iou_pol = FixedIoUThresholdPolicy(iou)
                ioa_pol = FixedIoAThresholdPolicy(0.70)
                for dt in depth_diffs:
                    depth_pol = FixedDepthThresholdPolicy(dt)

                    preds = []
                    for i in image_ids:
                        cands = [
                            PipelineDetection(d["box_id"], d["bbox"], d["confidence"], d["depth"], d["depth_std"])
                            for d in data[i]["models"][m_key]
                            if d["confidence"] >= c
                        ]
                        kept = run_pipeline_depth_nms(cands, iou_pol, ioa_pol, depth_pol)
                        preds.append(len(kept))

                    m = compute_metrics(gt_list, preds)
                    all_experiments.append({
                        "model_key": m_key,
                        "model_label": m_label,
                        "family": m_fam,
                        "architecture": "Pipeline Depth-Only (EmptySem)",
                        "variant_name": f"{m_label} Depth-Only (conf={c:.2f}, iou={iou:.2f}, dt={dt:.3f})",
                        "conf_thresh": c,
                        "iou_thresh": iou,
                        "ioa_thresh": 0.70,
                        "depth_thresh": dt,
                        **m,
                    })

        # 3. Dynamic Policies
        dyn_policies = [
            (0.18, 0.10, 0.50, 0.50, 0.15, 0.008, 0.002, 0.50, "Dyn-Safe (0.18/0.10, iou=0.50)"),
            (0.16, 0.08, 0.50, 0.55, 0.15, 0.006, 0.002, 0.50, "Dyn-Balanced (0.16/0.08, iou=0.55)"),
            (0.14, 0.06, 0.50, 0.55, 0.15, 0.006, 0.002, 0.50, "Dyn-Recall (0.14/0.06, iou=0.55)"),
        ]
        for n_c, f_c, d_cut, b_iou, boost_iou, b_d, min_d, dec_d, label in dyn_policies:
            conf_pol = DistanceScaledConfidencePolicy(near_threshold=n_c, far_threshold=f_c, depth_cutoff=d_cut)
            iou_pol = DistanceScaledIoUPolicy(base_threshold=b_iou, distance_boost=boost_iou)
            ioa_pol = FixedIoAThresholdPolicy(0.70)
            depth_pol = DistanceDecayDepthPolicy(base_threshold=b_d, min_threshold=min_d, decay_factor=dec_d)

            preds = []
            for i in image_ids:
                cands = [
                    PipelineDetection(d["box_id"], d["bbox"], d["confidence"], d["depth"], d["depth_std"])
                    for d in data[i]["models"][m_key]
                ]
                cands = [c for c in cands if c.confidence >= conf_pol(c)]
                kept = run_pipeline_depth_nms(cands, iou_pol, ioa_pol, depth_pol)
                preds.append(len(kept))

            m = compute_metrics(gt_list, preds)
            all_experiments.append({
                "model_key": m_key,
                "model_label": m_label,
                "family": m_fam,
                "architecture": "Pipeline Depth-Only (EmptySem, Dynamic)",
                "variant_name": f"{m_label} Dynamic - {label}",
                "near_conf": n_c,
                "far_conf": f_c,
                **m,
            })

    # Sort all experiments by Counting Accuracy descending
    all_experiments.sort(key=lambda x: x["counting_accuracy"], reverse=True)

    out_json = Path("experiments/output/all_10_models_results.json")
    with open(out_json, "w") as f:
        json.dump(all_experiments, f, indent=2)

    print(f"\nSuccessfully evaluated {len(all_experiments)} configurations across all 10 models.")
    print(f"Saved results to {out_json}\n")

    # Display Top 30 configurations
    print("=" * 115)
    print("TOP 30 OVERALL CONFIGURATIONS IN DEEP SEARCH (YOLO26 & YOLO11 FAMILIES):")
    print("=" * 115)
    print(f"{'Rank':<4} | {'Model':<22} | {'Variant Name':<52} | {'Acc %':<7} | {'MAE':<6} | {'RMSE':<6} | {'Exact%':<6}")
    print("-" * 115)
    for rank, exp in enumerate(all_experiments[:30], 1):
        print(f"{rank:<4} | {exp['model_label'][:22]:<22} | {exp['variant_name'][:52]:<52} | {exp['counting_accuracy']:<7} | {exp['mae']:<6} | {exp['rmse']:<6} | {exp['exact_match_pct']:<6}")
    print("=" * 115)


if __name__ == "__main__":
    main()
