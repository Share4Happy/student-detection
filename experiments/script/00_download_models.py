"""
Download and verify all 10 YOLO object detection checkpoints into experiments/models/:
- YOLO26 Family: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt
- YOLO11 Family: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
"""

from __future__ import annotations

import shutil
from pathlib import Path
from ultralytics import YOLO


MODELS_YOLO26 = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
MODELS_YOLO11 = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
ALL_MODELS = MODELS_YOLO26 + MODELS_YOLO11


def main():
    target_dir = Path("experiments/models")
    target_dir.mkdir(parents=True, exist_ok=True)
    shared_data_dir = Path("data/models")

    print(f"Ensuring all {len(ALL_MODELS)} model checkpoints are present in {target_dir}...")
    print("=" * 70)

    for model_name in ALL_MODELS:
        dest_path = target_dir / model_name
        shared_path = shared_data_dir / model_name

        if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f"[EXISTS] {model_name:<12} -> {dest_path} ({size_mb:.1f} MB)")
            continue

        if shared_path.exists() and shared_path.stat().st_size > 1024 * 1024:
            print(f"[COPY]   {model_name:<12} from {shared_path} to {dest_path}")
            shutil.copy2(shared_path, dest_path)
            continue

        print(f"[FETCH]  Downloading {model_name} via Ultralytics...")
        try:
            # Ultralytics downloads into working dir or ultralytics cache if not found locally
            model = YOLO(model_name)
            # Find downloaded weight
            downloaded = Path(model_name)
            if downloaded.exists():
                shutil.move(str(downloaded), str(dest_path))
            else:
                shutil.copy2(model.ckpt_path, dest_path)
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f"[DONE]   {model_name:<12} saved ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[ERROR]  Failed to download {model_name}: {e}")

    print("=" * 70)
    print("Verification Summary:")
    all_ready = True
    for model_name in ALL_MODELS:
        p = target_dir / model_name
        if p.exists() and p.stat().st_size > 1024 * 1024:
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  ✓ {model_name:<12} ({size_mb:>5.1f} MB)")
        else:
            print(f"  ✗ {model_name:<12} MISSING!")
            all_ready = False

    if all_ready:
        print("\nAll 10 models are verified and ready for evaluation.")
    else:
        print("\nWarning: Some models are missing. Please inspect errors above.")


if __name__ == "__main__":
    main()
