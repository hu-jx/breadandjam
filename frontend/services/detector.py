import io
import random
import numpy as np
from PIL import Image

# just some bs here
def predict(image_bytes: bytes):
    random.seed(hash(image_bytes[:200]))
    prob_ai = round(random.random(), 3)
    confidence = round(0.6 + random.random() * 0.4, 3)
    return prob_ai, confidence

def make_heatmap(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(img)
    tinted = np.stack([arr, arr // 2, arr // 3], axis=-1).astype(np.uint8)
    return Image.fromarray(tinted)