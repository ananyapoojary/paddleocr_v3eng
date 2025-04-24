from paddleocr import PaddleOCR, draw_ocr
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- SETUP ---
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Angle classification and English model

# Path to the image you want to test
image_path = '../images/3.jpeg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    # --- OPTIONAL PREPROCESSING ---
    # 1. Denoise the image
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 7, 21)

    # 2. Increase contrast (multiply by a factor of 2)
    alpha = 2.0  # Contrast control (1.0-3.0)
    beta = 0     # Brightness control
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 3. Sharpen the image using kernel filter
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)

    # --- OCR ---
    results = ocr.ocr(image_path, cls=True)

    if results is None or len(results) == 0:
        print("No text detected.")
    else:
        # Print detected text with confidence
        for line in results[0]:
            print(f"Detected: '{line[1][0]}' with confidence {line[1][1]:.2f}")

        # --- DRAW RESULTS ---
        boxes = [element[0] for element in results[0]]
        txts = [element[1][0] for element in results[0]]
        scores = [element[1][1] for element in results[0]]

        # Draw OCR results on the original image
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        image_with_boxes = draw_ocr(image, boxes, txts, scores, font_path=font_path)

        # Save result image
        cv2.imwrite("4.png", image_with_boxes)

        # Display result using matplotlib
        img = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        plt.title("OCR Result")
        plt.show()
