import numpy as np
from PIL import Image

from .common import NumpyImage
from .pipeline import DetectorPipeline
from .pipeline.depth_estimation import DepthAnythingV2
from .pipeline.object_detection import YOLOPerson
from .pipeline.semantic_segmentation import YOLOSemanticPerson

image = Image.open("data/images/classroom.png").convert("RGB")
image = NumpyImage(np.array(image))

pipeline = DetectorPipeline(
    sem_model=YOLOSemanticPerson("data/models/yolo26m-seg.pt", "cuda"),
    obj_model=YOLOPerson("data/models/yolo26m.pt", "cpu"),
    dep_model=DepthAnythingV2(device="cuda")
)
res = pipeline.predict(image)
print(res)
