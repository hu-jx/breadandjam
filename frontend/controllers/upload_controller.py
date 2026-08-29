from datetime import datetime
from typing import Iterable
from models.upload import Upload
from services import detector
from state import uploads as uploads_state

def handle_new_files(files: Iterable):
    for f in files:
        if (uploads_state.exists(f.name)):
            continue

        image_bytes = f.getvalue()
        prob_ai, confidence = detector.predict(image_bytes)
        upload = Upload(
            name=f.name,
            bytes=image_bytes,
            prob_ai=prob_ai,
            confidence=confidence,
            verdict="AI" if prob_ai > 0.5 else "Human",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        uploads_state.add(upload)