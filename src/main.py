import numpy as np
from PIL import Image

from .common import NumpyImage
from .pipeline.semantic_segmentation import YOLOSemanticPerson
from .pipeline.depth_estimation import DepthAnythingV2

image = Image.open("data/images/classroom.png").convert("RGB")
image = NumpyImage(np.array(image))

# model = YOLOPerson("models/yolo26m.pt", "cpu")
# model = YOLOSemanticPerson("models/yolo26m-seg.pt", "cuda")
model = DepthAnythingV2(device="cuda")

res = model.predict(image)
print(res)
