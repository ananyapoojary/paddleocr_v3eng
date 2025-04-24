from paddleocr import PaddleOCR, draw_ocr
import cv2
import matplotlib.pyplot as plt

# --- SETUP ---

# Initialize the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # English + angle classification

# Path to the image you want to test
image_path = '../images/3.jpeg'  # Make sure this path is correct
image = cv2.imread(image_path)

# --- OCR ---

# Run OCR on the image
results = ocr.ocr(image_path, cls=True)

# Print detected text with confidence
for line in results[0]:
    print(f"Detected: '{line[1][0]}' with confidence {line[1][1]:.2f}")

# --- DRAW RESULTS ---

# Extract boxes, text, and scores
boxes = [element[0] for element in results[0]]
txts = [element[1][0] for element in results[0]]
scores = [element[1][1] for element in results[0]]

# Proper font path (Ubuntu friendly)
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Draw OCR results on image
image_with_boxes = draw_ocr(image, boxes, txts, scores, font_path=font_path)

# Save result
cv2.imwrite("3.png", image_with_boxes)

# Display result using matplotlib
img = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title("OCR Result")
plt.show()
