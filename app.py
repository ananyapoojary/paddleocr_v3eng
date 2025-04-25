from paddleocr import PaddleOCR, draw_ocr
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- IoU FUNCTION ---
def compute_iou(box1, box2):
    box1 = np.array(box1).reshape(-1, 2)
    box2 = np.array(box2).reshape(-1, 2)
    x_min1, y_min1 = np.min(box1, axis=0)
    x_max1, y_max1 = np.max(box1, axis=0)
    x_min2, y_min2 = np.min(box2, axis=0)
    x_max2, y_max2 = np.max(box2, axis=0)

    inter_xmin = max(x_min1, x_min2)
    inter_ymin = max(y_min1, y_min2)
    inter_xmax = min(x_max1, x_max2)
    inter_ymax = min(y_max1, y_max2)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# --- OCR SETUP ---
ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_score_mode='fast', layout=True)

image_path = 'images/3.jpeg'
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
    exit()

# --- ADVANCED PREPROCESSING ---
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    adaptive = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

    bilateral = cv2.bilateralFilter(adaptive, 9, 75, 75)

    kernel = np.ones((1, 1), np.uint8)
    morph = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, kernel)

    gamma = 1.5
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(morph, table)

    return corrected

preprocessed_image = preprocess_image(image)

# --- OCR RESULTS ---
results_no_preprocessing = ocr.ocr(image, cls=True)
results_with_preprocessing = ocr.ocr(preprocessed_image, cls=True)

# --- DEDUPLICATION ---
final_results = []
for new_box, (new_text, new_score) in results_no_preprocessing[0] + results_with_preprocessing[0]:
    found = False
    for i, (box, (text, score)) in enumerate(final_results):
        if compute_iou(box, new_box) > 0.5 and new_text.strip().lower() == text.strip().lower():
            if new_score > score:
                final_results[i] = (new_box, (new_text, new_score))
            found = True
            break
    if not found:
        final_results.append((new_box, (new_text, new_score)))

# --- DRAW RESULTS ---
boxes = [item[0] for item in final_results]
txts = [item[1][0] for item in final_results]
scores = [item[1][1] for item in final_results]
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
image_with_boxes = draw_ocr(image, boxes, txts, scores, font_path=font_path)

# --- DISPLAY ---
output_path = "ocr_code/c8_3.png"
cv2.imwrite(output_path, image_with_boxes)
img = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title("Enhanced OCR Result w/ Layout + IoU + Preprocessing")
plt.show()
