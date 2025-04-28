import cv2
import numpy as np
from paddleocr import PaddleOCR

# Initialize OCR model once
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

def run_ocr_on_image(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image."}

    results = ocr_model.ocr(img, cls=True)

    extracted_texts = []
    for line in results[0]:
        text, score = line[1]
        extracted_texts.append({
            "text": text,
            "score": float(score)
        })

    return {"results": extracted_texts}
