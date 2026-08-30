import random

# just some bs here
def predict(image_bytes: bytes):
    random.seed(hash(image_bytes[:200]))
    prob_ai = round(random.random(), 3)
    confidence = round(0.6 + random.random() * 0.4, 3)
    return prob_ai, confidence
