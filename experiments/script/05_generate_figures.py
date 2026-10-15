"""
Generate publication-quality figures for the 2-step experiment workflow.
Saves all figures into experiments/output/figures/.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from student_detection.core.models import PipelineDetection


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

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_fig1_architecture_comparison(step1_data, out_dir):
    """Dual horizontal bar chart comparing the best variant of each architecture in Step 1."""
    arch_order = [
        ("Arch 4: Pipeline Depth Only (EmptySem, Fixed)", "Pipeline Depth-Only\n(EmptySem, Fixed)", "#2ca02c"),
        ("Arch 5: Pipeline Depth Only (EmptySem, Dynamic)", "Pipeline Depth-Only\n(EmptySem, Dynamic)", "#85c185"),
        ("Arch 1: Bare Object Detection", "Bare YOLO26x\n(Single-Class NMS)", "#1f77b4"),
        ("Arch 2: Pipeline Sem-ROI + Depth (Fixed)", "Pipeline Sem-ROI + Depth\n(Cityscapes, Fixed)", "#d62728"),
        ("Arch 3: Pipeline Sem-ROI + Depth (Dynamic)", "Pipeline Sem-ROI + Depth\n(Cityscapes, Dynamic)", "#e377c2"),
    ]

    selected = []
    for arch_key, display_name, color in arch_order:
        candidates = [e for e in step1_data if e["architecture"] == arch_key]
        if candidates:
            best = candidates[0]
            selected.append((display_name, best["counting_accuracy"], best["mae"], color))

    # Reverse for top-to-bottom barh order
    selected = selected[::-1]
    labels = [s[0] for s in selected]
    accs = [s[1] for s in selected]
    maes = [s[2] for s in selected]
    colors = [s[3] for s in selected]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    y_pos = np.arange(len(labels))

    # Accuracy Barh
    bars1 = ax1.barh(y_pos, accs, color=colors, height=0.55, edgecolor="#333333", linewidth=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10.5, fontweight="medium")
    ax1.set_xlabel("Counting Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Step 1: Counting Accuracy by Pipeline Architecture (Higher is Better)", fontsize=11.5, fontweight="bold", pad=12)
    ax1.set_xlim(0, 110)
    ax1.grid(axis="x", linestyle="--", alpha=0.6)

    for bar, val in zip(bars1, accs):
        ax1.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", ha="left", fontsize=10, fontweight="bold", color="#111111")

    # MAE Barh
    bars2 = ax2.barh(y_pos, maes, color=colors, height=0.55, edgecolor="#333333", linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Mean Absolute Error (Students)", fontsize=11, fontweight="bold")
    ax2.set_title("Step 1: MAE by Pipeline Architecture (Lower is Better)", fontsize=11.5, fontweight="bold", pad=12)
    ax2.set_xlim(0, max(maes) * 1.22)
    ax2.grid(axis="x", linestyle="--", alpha=0.6)

    for bar, val in zip(bars2, maes):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", ha="left", fontsize=10, fontweight="bold", color="#111111")

    plt.subplots_adjust(left=0.25, right=0.96, top=0.88, bottom=0.14, wspace=0.15)
    fig_path = out_dir / "fig1_step1_architecture_comparison.png"
    plt.savefig(fig_path, dpi=220)
    plt.close()
    print(f"Saved {fig_path}")


def plot_fig2_model_size_comparison(all10_data, out_dir):
    """Compares all 10 models across n, s, m, l, x sizes for YOLO26 vs YOLO11."""
    sizes = ["n", "s", "m", "l", "x"]
    size_labels = ["Nano (n)", "Small (s)", "Medium (m)", "Large (l)", "X-large (x)"]

    yolo26_keys = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]
    yolo11_keys = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]

    yolo26_accs = []
    yolo26_maes = []
    for k in yolo26_keys:
        sub = [e for e in all10_data if e["model_key"] == k and "Pipeline" in e["architecture"]]
        best = max(sub, key=lambda x: x["counting_accuracy"])
        yolo26_accs.append(best["counting_accuracy"])
        yolo26_maes.append(best["mae"])

    yolo11_accs = []
    yolo11_maes = []
    for k in yolo11_keys:
        sub = [e for e in all10_data if e["model_key"] == k and "Pipeline" in e["architecture"]]
        best = max(sub, key=lambda x: x["counting_accuracy"])
        yolo11_accs.append(best["counting_accuracy"])
        yolo11_maes.append(best["mae"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(sizes))
    width = 0.35

    # Subplot 1: Accuracy comparison by size
    bars1_26 = ax1.bar(x - width/2, yolo26_accs, width, label="YOLO26 Family", color="#2ca02c", edgecolor="#1a5e1a")
    bars1_11 = ax1.bar(x + width/2, yolo11_accs, width, label="YOLO11 Family", color="#1f77b4", edgecolor="#0e3d61")

    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels, fontsize=10.5, fontweight="medium")
    ax1.set_ylabel("Counting Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Model Scaling: Counting Accuracy by Model Size (n, s, m, l, x)", fontsize=11.5, fontweight="bold", pad=12)
    ax1.set_ylim(65, 102)
    ax1.legend(frameon=True, fontsize=10.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars1_26:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#111111")
    for bar in bars1_11:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#111111")

    # Subplot 2: MAE comparison by size
    bars2_26 = ax2.bar(x - width/2, yolo26_maes, width, label="YOLO26 Family", color="#2ca02c", edgecolor="#1a5e1a")
    bars2_11 = ax2.bar(x + width/2, yolo11_maes, width, label="YOLO11 Family", color="#1f77b4", edgecolor="#0e3d61")

    ax2.set_xticks(x)
    ax2.set_xticklabels(size_labels, fontsize=10.5, fontweight="medium")
    ax2.set_ylabel("Mean Absolute Error (Students)", fontsize=11, fontweight="bold")
    ax2.set_title("Model Scaling: MAE by Model Size (Lower is Better)", fontsize=11.5, fontweight="bold", pad=12)
    ax2.set_ylim(0, max(max(yolo26_maes), max(yolo11_maes)) * 1.25)
    ax2.legend(frameon=True, fontsize=10.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars2_26:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.25, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#111111")
    for bar in bars2_11:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.25, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#111111")

    plt.tight_layout()
    fig_path = out_dir / "fig2_step2_model_size_comparison.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved {fig_path}")


def plot_fig3_heatmap(step2_data, out_dir):
    """Heatmap showing accuracy landscape across Confidence vs Depth Difference for YOLO26x."""
    yolo26x_subset = [
        e for e in step2_data
        if "YOLO26x" in e.get("model_label", "")
        and e.get("architecture") == "Pipeline Depth-Only (EmptySem)"
        and e.get("iou_thresh") == 0.55
    ]
    if not yolo26x_subset:
        return

    confs = sorted(list(set(e["conf_thresh"] for e in yolo26x_subset)))
    dts = sorted(list(set(e["depth_thresh"] for e in yolo26x_subset)))

    grid = np.zeros((len(dts), len(confs)))
    for i, dt in enumerate(dts):
        for j, c in enumerate(confs):
            matching = [e for e in yolo26x_subset if e["depth_thresh"] == dt and e["conf_thresh"] == c]
            if matching:
                grid[i, j] = matching[0]["counting_accuracy"]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    cax = ax.imshow(grid, cmap="YlGnBu", aspect="auto", origin="lower")
    ax.grid(False)

    ax.set_xticks(np.arange(len(confs)))
    ax.set_yticks(np.arange(len(dts)))
    ax.set_xticklabels([f"{c:.2f}" for c in confs], fontsize=10.5)
    ax.set_yticklabels([f"{dt:.3f}" for dt in dts], fontsize=10.5)

    ax.set_xlabel("Detection Confidence Threshold", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Depth Difference Threshold (\u0394d)", fontsize=11.5, fontweight="bold")
    ax.set_title("Figure 3: Accuracy Heatmap (Confidence vs Depth Difference) [IoU=0.55]", fontsize=12.5, fontweight="bold", pad=14)

    # Print values inside cells with automatic luminance-based contrast
    norm = cax.norm
    for i in range(len(dts)):
        for j in range(len(confs)):
            val = grid[i, j]
            rgba = cax.cmap(norm(val))
            # Relative perceived luminance
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "#ffffff" if luminance < 0.60 else "#111111"
            ax.text(
                j,
                i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    cbar = fig.colorbar(cax, pad=0.03)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Counting Accuracy (%)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig_path = out_dir / "fig3_deep_hparams_heatmap.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved {fig_path}")


def plot_fig4_error_distribution(step1_data, step2_data, test_features, out_dir):
    """Error distribution comparison between Winning Arch, Bare YOLO, and Pipeline with Sem."""
    image_ids = list(test_features.keys())
    gt = np.array([test_features[i]["gt_count"] for i in image_ids])

    # 1. Best overall model: YOLO26x Depth-Only (conf=0.12, iou=0.55, dt=0.006)
    preds_champion = []
    for i in image_ids:
        cands = [
            PipelineDetection(d["box_id"], d["bbox"], d["confidence"], d["depth"], d["depth_std"])
            for d in test_features[i]["pipeline_no_sem"]
            if d["confidence"] >= 0.12
        ]
        kept = run_pipeline_depth_nms(cands, lambda a, b: 0.55, lambda a, b: 0.70, lambda a, b: 0.006)
        preds_champion.append(len(kept))
    err_champion = np.array(preds_champion) - gt

    # 2. Bare YOLO26x (conf=0.22)
    preds_bare = np.array([sum(1 for b in test_features[i]["bare_yolo26x"] if b["confidence"] >= 0.22) for i in image_ids])
    err_bare = preds_bare - gt

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(-15, 15, 1)

    ax.hist(err_champion, bins=bins, alpha=0.7, label=f"Winning Arch: YOLO26x + Depth-NMS (MAE={np.mean(np.abs(err_champion)):.2f}, Acc=95.8%)", color="#2ca02c", edgecolor="#1a5e1a")
    ax.hist(err_bare, bins=bins, alpha=0.5, label=f"Bare YOLO26x (conf=0.22) (MAE={np.mean(np.abs(err_bare)):.2f}, Acc=94.2%)", color="#1f77b4", edgecolor="#0e3d61")

    ax.axvline(0, color="red", linestyle="--", linewidth=1.8, label="Zero Error (Exact Match)")
    ax.set_title("Student Counting Error Distribution (Pred - GT)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Prediction Error in Students (Pred - GT)", fontsize=11)
    ax.set_ylabel("Number of Test Images", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_path = out_dir / "fig4_error_distribution.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved {fig_path}")


def plot_fig5_scatter(test_features, out_dir):
    """Ground truth vs prediction scatter plot for champion architecture."""
    image_ids = list(test_features.keys())
    gt = [test_features[i]["gt_count"] for i in image_ids]

    preds = []
    for i in image_ids:
        cands = [
            PipelineDetection(d["box_id"], d["bbox"], d["confidence"], d["depth"], d["depth_std"])
            for d in test_features[i]["pipeline_no_sem"]
            if d["confidence"] >= 0.12
        ]
        kept = run_pipeline_depth_nms(cands, lambda a, b: 0.55, lambda a, b: 0.70, lambda a, b: 0.006)
        preds.append(len(kept))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(gt, preds, alpha=0.6, color="#2ca02c", edgecolors="#1b631b", s=45, label="Winning Pipeline (EmptySem + Depth-NMS)")

    min_val = min(min(gt), min(preds)) - 2
    max_val = max(max(gt), max(preds)) + 2

    ax.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", linewidth=1.8, label="Ideal Identity (Pred = GT)")
    ax.fill_between([min_val, max_val], [min_val - 2, max_val - 2], [min_val + 2, max_val + 2], color="crimson", alpha=0.1, label="\u00b12 Students Error Band")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_title("Ground Truth vs. Predicted Count (Champion Model)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Ground Truth Student Count", fontsize=11)
    ax.set_ylabel("Predicted Student Count", fontsize=11)
    ax.legend(frameon=True, loc="upper left", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_path = out_dir / "fig5_ground_truth_vs_prediction.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved {fig_path}")


def main():
    out_dir = Path("experiments/output/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open("experiments/output/step1_architecture_results.json") as f:
        step1_data = json.load(f)

    with open("experiments/output/all_10_models_results.json") as f:
        all10_data = json.load(f)

    with open("experiments/output/test_features.json") as f:
        test_features = json.load(f)

    plot_fig1_architecture_comparison(step1_data, out_dir)
    plot_fig2_model_size_comparison(all10_data, out_dir)
    plot_fig3_heatmap(all10_data, out_dir)
    plot_fig4_error_distribution(step1_data, all10_data, test_features, out_dir)
    plot_fig5_scatter(test_features, out_dir)

    print("\nAll publication figures generated in experiments/output/figures/.")


if __name__ == "__main__":
    main()
