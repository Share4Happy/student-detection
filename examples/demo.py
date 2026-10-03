import numpy as np
from PIL import Image

from student_detection.common import NumpyImage
from student_detection.core import DetectorPipeline
from student_detection.core.depth_estimation import DepthAnythingV2
from student_detection.core.object_detection import YOLOPerson
from student_detection.core.policy import (
    FixedConfidencePolicy,
    FixedDepthThresholdPolicy,
    FixedIoAThresholdPolicy,
    FixedIoUThresholdPolicy,
)
from student_detection.core.semantic_segmentation import YOLOSemanticPerson


def main() -> None:
    image = Image.open("data/images/classroom.png").convert("RGB")
    numpy_image = NumpyImage(np.array(image))

    pipeline = DetectorPipeline(
        sem_model=YOLOSemanticPerson("data/models/yolo26m-seg.pt", "cuda"),
        obj_model=YOLOPerson("data/models/yolo26m.pt", "cpu"),
        dep_model=DepthAnythingV2(device="cuda"),
    )
    result = pipeline.predict(
        image=numpy_image,
        conf_policy=FixedConfidencePolicy(0.05),
        iou_policy=FixedIoUThresholdPolicy(0.25),
        ioa_policy=FixedIoAThresholdPolicy(0.35),
        depth_policy=FixedDepthThresholdPolicy(0.010),
    )
    print(f"Detected {result.people_count} students.")


if __name__ == "__main__":
    main()
