"""
01_extract_features.py

Extract and cache candidate detections and monocular depth statistics
for all 10 YOLO object detection models + Depth Anything V2:
- YOLO26 Family: yolo26n, yolo26s, yolo26m, yolo26l, yolo26x
- YOLO11 Family: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

Saves consolidated feature cache to:
- experiments/output/test_features_all10.json
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from student_detection.common import NumpyImage
from student_detection.core.object_detection import YOLOPerson
from student_detection.core.depth_estimation import DepthAnythingV2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_root = Path("data/datasets/students_3320imgs_coco/test")
    test_json = dataset_root / "_annotations.coco.json"
    output_path = Path("experiments/output/test_features_all10.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(test_json) as f:
        coco = json.load(f)

    gt_counts = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        gt_counts[img_id] = gt_counts.get(img_id, 0) + 1

    # Check checkpoints in experiments/models first, fallback to data/models
    models_to_load = [
        "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
        "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"
    ]
    model_paths = {}
    for m in models_to_load:
        p_exp = Path(f"experiments/models/{m}.pt")
        p_data = Path(f"data/models/{m}.pt")
        if p_exp.exists():
            model_paths[m] = str(p_exp)
        elif p_data.exists():
            model_paths[m] = str(p_data)
        else:
            raise FileNotFoundError(f"Model checkpoint for {m} not found in experiments/models/ or data/models/.")

    print(f"All {len(model_paths)} model checkpoints located.")

    # Load existing progress if any
    data = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                data = json.load(f)
            print(f"Loaded existing cache with {len(data)} images.")
        except Exception:
            data = {}

    # Initialize models
    print("Initializing models on device...")
    models = {k: YOLOPerson(p, device=device) for k, p in model_paths.items()}
    dep_model = DepthAnythingV2(device=device)
    print("All models initialized successfully.")

    images = coco["images"]
    to_process = [img for img in images if str(img["id"]) not in data or len(data[str(img["id"])].get("models", {})) < 10]
    print(f"Total test images: {len(images)}, Remaining to process: {len(to_process)}")

    if not to_process:
        print(f"Feature cache is already complete at {output_path}.")
        return

    for idx, img_info in enumerate(tqdm(to_process, desc="Extracting features for 10 models")):
        img_id = str(img_info["id"])
        file_name = img_info["file_name"]
        img_path = dataset_root / file_name

        pil_img = Image.open(img_path).convert("RGB")
        numpy_img = NumpyImage(np.array(pil_img))

        # Monocular depth map
        depth_res = dep_model.predict(numpy_img)
        depth_norm = depth_res.get_normalized()
        h, w = depth_norm.shape[:2]

        models_data = {}
        for m_name, model in models.items():
            raw_boxes = model.predict(numpy_img, conf_threshold=0.01)
            box_records = []
            for b_idx, b in enumerate(raw_boxes):
                x1, y1 = max(0, b.x1), max(0, b.y1)
                x2, y2 = min(w, b.x2), min(h, b.y2)
                crop_depth = depth_norm[y1:y2, x1:x2]
                med_depth = float(np.median(crop_depth)) if crop_depth.size > 0 else 0.0
                std_depth = float(np.std(crop_depth)) if crop_depth.size > 0 else 0.0
                box_records.append({
                    "box_id": b_idx,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(b.confidence),
                    "depth": med_depth,
                    "depth_std": std_depth,
                })
            models_data[m_name] = box_records

        data[img_id] = {
            "image_id": img_info["id"],
            "file_name": file_name,
            "gt_count": gt_counts.get(img_info["id"], 0),
            "models": models_data,
        }

        # Periodic checkpoint every 25 images
        if (idx + 1) % 25 == 0 or (idx + 1) == len(to_process):
            with open(output_path, "w") as f:
                json.dump(data, f)

    print(f"\nCompleted feature extraction for all {len(data)} images.")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
