# Experiments Scripts & Reproduction Guide

This directory contains the complete, modular pipeline for the two-stage student counting evaluation study on the `students_3320imgs_coco` dataset.

---

## 1. Directory Structure

```text
experiments/
├── .gitignore                          # Ignores large model weights (.pt, .pth) and python caches
├── REPORT.md                           # Main publication report with tables and analysis
├── models/                             # Downloaded checkpoints (ignored by git)
│   ├── yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt
│   └── yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
├── output/                             # Generated evaluation caches and figures
│   ├── all_10_models_results.json      # Step 2 full grid evaluation (1,680 runs)
│   ├── dynamic_vs_fixed_results.json   # Confidence & dynamic policy sweep (101 runs)
│   ├── step1_architecture_results.json # Step 1 shallow screening leaderboard (25 runs)
│   ├── test_features_all10.json        # Cached raw detections & depth statistics (332 test images)
│   ├── test_features.json              # Cached Step 1 candidate detections
│   └── figures/                        # Generated publication plots (fig1 - fig6)
└── script/                             # Execution scripts
    ├── 00_download_models.py           # Download & verify all 10 YOLO checkpoints
    ├── 01_extract_features.py          # Run inference & extract depth-calibrated bounding boxes
    ├── 02_step1_architecture_screening.py # Step 1: 5 architectures × 5 configs (25 runs)
    ├── 03_step2_evaluate_10_models.py  # Step 2: 10 models × 168 configs (1,680 runs)
    ├── 04_compare_dynamic_vs_fixed.py  # Step 3: Confidence curve & clutter safety analysis
    ├── 05_generate_figures.py          # Step 4: Render diagnostic & publication figures
    ├── 06_export_report.py             # Step 5: Export REPORT.md to DOCX and PDF
    ├── empty_semantic.py               # Adapter bypassing semantic segmentation in DetectorPipeline
    └── README.md                       # This reproduction guide
```

---

## 2. Core Modules & Pipeline Components

- **`empty_semantic.py`**:
  - Implements `EmptyYOLOSemanticPerson`, a specialized adapter inheriting from `SemanticSegmentationModel`.
  - Bypasses Cityscapes semantic segmentation masking while preserving full compatibility with `DetectorPipeline`.
  - Ensures the exact pipeline class used in production deployment is tested during benchmarking.

---

## 3. Step-by-Step Reproduction Guide

Execute commands from the project root (`student-detection/`) using the project virtual environment (`.venv/bin/python`).

### Step 0: Download and Verify Model Weights
Downloads or copies all 10 YOLO detection checkpoints into `experiments/models/`:
```bash
.venv/bin/python experiments/script/00_download_models.py
```
- **Weights downloaded**: `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt`, and `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`.
- Automatically ignored by `experiments/.gitignore`.

### Step 1: Feature Extraction & Monocular Depth Sampling
Runs raw object detection and Depth Anything V2 depth estimation over all 332 test images:
```bash
.venv/bin/python experiments/script/01_extract_features.py
```
- **Outputs**: `experiments/output/test_features_all10.json` (57 MB).
- Includes automated checkpointing every 25 images so extraction can resume seamlessly if interrupted.

### Step 2: Step 1 Architecture Screening (25 Configurations)
Executes a controlled shallow screening across 5 candidate architectures with fixed base detector `YOLO26x`:
```bash
.venv/bin/python experiments/script/02_step1_architecture_screening.py
```
- **Architectures compared**:
  1. Bare Object Detection (`YOLO26x` alone with standard 2D NMS)
  2. Pipeline with Sem-ROI + Depth (Fixed Policies)
  3. Pipeline with Sem-ROI + Depth (Dynamic Policies)
  4. Pipeline Depth-Only (`EmptyYOLOSemanticPerson`, Fixed Policies)
  5. Pipeline Depth-Only (`EmptyYOLOSemanticPerson`, Dynamic Policies)
- **Outputs**:
  - `experiments/output/step1_architecture_results.json`
  - Complete 25-row ranked leaderboard table with **$\Delta$ Acc** relative to Rank 1.

### Step 3: Step 2 Deep Search Across All 10 Models (1,680 Configurations)
Scales the winning **Depth-Only Pipeline** across all 10 models and conducts a fine-grained grid search:
```bash
.venv/bin/python experiments/script/03_step2_evaluate_10_models.py
```
- **Grid Swept per Model**:
  - 11 Confidence thresholds: `[0.06, 0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.25]`
  - 4 IoU thresholds: `[0.45, 0.55, 0.65, 0.75]`
  - 4 Depth $\Delta d$ thresholds: `[0.003, 0.006, 0.010, 0.015]`
  - 3 Dynamic distance-scaled policies: `[Dyn-Recall, Dyn-Balanced, Dyn-Safe]`
- **Outputs**:
  - `experiments/output/all_10_models_results.json`
  - Generates the **Top 30 Fixed Models** and **Top 30 Dynamic Models** leaderboards with true Global Ranks.

### Step 4: Fixed vs. Dynamic Confidence Trade-Off Study
Evaluates empty-room clutter protection versus full-room back-row recall across 101 configurations:
```bash
.venv/bin/python experiments/script/04_compare_dynamic_vs_fixed.py
```
- **Outputs**:
  - `experiments/output/dynamic_vs_fixed_results.json`
  - `experiments/output/figures/fig6_dynamic_vs_fixed_comparison.png`

### Step 5: Publication Figures Generation
Generates all diagnostic plots and publication-quality figures:
```bash
.venv/bin/python experiments/script/05_generate_figures.py
```
- **Outputs in `experiments/output/figures/`**:
  - `fig1_step1_architecture_comparison.png`: Dual horizontal bar chart (Accuracy & MAE) across architectures.
  - `fig2_step2_model_size_comparison.png`: Model scaling bar chart across all 10 models (YOLO26 vs. YOLO11).
  - `fig3_deep_hparams_heatmap.png`: High-contrast accuracy landscape (Confidence vs. Depth Difference).
  - `fig4_error_distribution.png`: Student counting error distribution histogram (Winning Pipeline vs. Bare YOLO).
  - `fig5_ground_truth_vs_prediction.png`: Parity scatter plot with identity line.

### Step 6: Export Research Report to DOCX and PDF
Converts the consolidated Markdown report to polished DOCX and PDF documents:
```bash
.venv/bin/python experiments/script/06_export_report.py
```
- **Outputs**:
  - `experiments/output/REPORT.docx` (with integrated table of contents and figures)
  - `experiments/output/REPORT.pdf` (rendered via headless LibreOffice)

---

## 4. Hardware & Environment Requirements

- **GPU**: NVIDIA GPU with $\ge 6$ GB VRAM (CUDA 12.0+). Tested on NVIDIA RTX 3050 6GB Laptop GPU.
- **Python**: $\ge 3.10$.
- **Dependencies**: `torch`, `torchvision`, `ultralytics`, `matplotlib`, `numpy`, `tqdm`, `pillow`.
