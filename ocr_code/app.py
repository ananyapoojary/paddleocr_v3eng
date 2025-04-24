import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR, draw_ocr

# --- SETUP ---

# Initialize PaddleOCR with layout detection and English language
ocr = PaddleOCR(use_angle_cls=True, lang='en', layout=True, det_db_score_mode='fast')

# Load the image
image_path = '../images/6.jpeg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
    exit()

# --- PREPROCESSING ---

def preprocess_image(image):
    # 1. Denoising
    img = cv2.fastNlMeansDenoisingColored(image, None, 10, 7, 21)

    # 2. Contrast Boost
    alpha = 2.0
    beta = 0
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 3. Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    return img

# --- VALIDATION ---

def is_valid_result(result):
    return result is not None and isinstance(result, list) and len(result) > 0 and isinstance(result[0], list)

# --- IoU DEDUPLICATION ---

def iou(box1, box2):
    box1 = np.array(box1).reshape(-1, 2)
    box2 = np.array(box2).reshape(-1, 2)
    x1 = max(box1[:, 0].min(), box2[:, 0].min())
    y1 = max(box1[:, 1].min(), box2[:, 1].min())
    x2 = min(box1[:, 0].max(), box2[:, 0].max())
    y2 = min(box1[:, 1].max(), box2[:, 1].max())

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[:, 0].max() - box1[:, 0].min()) * (box1[:, 1].max() - box1[:, 1].min())
    box2_area = (box2[:, 0].max() - box2[:, 0].min()) * (box2[:, 1].max() - box2[:, 1].min())
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def deduplicate_results(results, iou_threshold=0.5):
    deduped = []
    for item in results:
        box, text, score = item[0], item[1][0], item[1][1]
        duplicate = False
        for existing in deduped:
            if iou(box, existing[0]) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            deduped.append(item)
    return deduped

# --- OCR ORIGINAL IMAGE ---

print("\nText detected in original image (no preprocessing):")
results_no_preprocessing = ocr.ocr(image, cls=True)
if is_valid_result(results_no_preprocessing):
    for line in results_no_preprocessing[0]:
        print(f"Detected: '{line[1][0]}' with confidence {line[1][1]:.2f}")
else:
    print("No text detected.")

# --- OCR PREPROCESSED IMAGE ---

preprocessed_image = preprocess_image(image)
print("\nText detected in preprocessed image:")
results_with_preprocessing = ocr.ocr(preprocessed_image, cls=True)
if is_valid_result(results_with_preprocessing):
    for line in results_with_preprocessing[0]:
        print(f"Detected: '{line[1][0]}' with confidence {line[1][1]:.2f}")
else:
    print("No text detected.")

# --- COMBINE + DEDUPLICATE RESULTS ---

combined_results = []
if is_valid_result(results_no_preprocessing):
    combined_results += results_no_preprocessing[0]
if is_valid_result(results_with_preprocessing):
    combined_results += results_with_preprocessing[0]

deduped_results = deduplicate_results(combined_results)

# --- DRAW RESULTS ---

boxes = [res[0] for res in deduped_results]
texts = [res[1][0] for res in deduped_results]
scores = [res[1][1] for res in deduped_results]

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
image_with_boxes = draw_ocr(image, boxes, texts, scores, font_path=font_path)

# Save and display
cv2.imwrite("6ocr_result_combined.png", image_with_boxes)
img_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.title("Final OCR Result (Original + Preprocessed, Deduplicated)")
plt.show()
