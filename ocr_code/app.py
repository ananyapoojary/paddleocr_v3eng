from paddleocr import PaddleOCR, draw_ocr
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- SETUP ---

# Initialize OCR model with angle classification and English language
ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_score_mode='fast')  # Use 'fast' for quicker but less accurate or 'accurate' for better quality

# Path to the image you want to test
image_path = '../images/43.jpeg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    # --- OCR WITHOUT PREPROCESSING ---

    # Run OCR on the original image (without preprocessing)
    results_no_preprocessing = ocr.ocr(image, cls=True)

    if results_no_preprocessing is None or len(results_no_preprocessing) == 0:
        print("No text detected in original image.")
    else:
        print("Text detected in original image (no preprocessing):")
        for line in results_no_preprocessing[0]:
            print(f"Detected: '{line[1][0]}' with confidence {line[1][1]:.2f}")
    
    # --- OPTIONAL PREPROCESSING ---
    
    # 1. Denoise the image using Non-Local Means Denoising
    preprocessed_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 7, 21)

    # 2. Increase contrast (multiply by a factor of 2)
    alpha = 2.0  # Contrast control (1.0-3.0)
    beta = 0     # Brightness control
    preprocessed_image = cv2.convertScaleAbs(preprocessed_image, alpha=alpha, beta=beta)

    # 3. Sharpen the image using kernel filter (using a basic sharpening kernel)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    preprocessed_image = cv2.filter2D(preprocessed_image, -1, kernel)

    # --- OCR WITH PREPROCESSING ---

    # Run OCR on the preprocessed image
    results_with_preprocessing = ocr.ocr(preprocessed_image, cls=True)

    if results_with_preprocessing is None or len(results_with_preprocessing) == 0:
        print("No text detected in preprocessed image.")
    else:
        print("Text detected in preprocessed image:")
        for line in results_with_preprocessing[0]:
            print(f"Detected: '{line[1][0]}' with confidence {line[1][1]:.2f}")

    # --- COMBINE RESULTS ---

    # Combine the results from both images
    combined_results = results_no_preprocessing[0] + results_with_preprocessing[0]

    # --- DRAW RESULTS ---
    # Extract the bounding boxes, texts, and scores from combined results
    boxes = [element[0] for element in combined_results]
    txts = [element[1][0] for element in combined_results]
    scores = [element[1][1] for element in combined_results]

    # Specify the font path (Ubuntu-friendly)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    # Draw OCR results on the original image (not preprocessed)
    image_with_boxes = draw_ocr(image, boxes, txts, scores, font_path=font_path)

    # Save the result image
    cv2.imwrite("43.png", image_with_boxes)

    # Display the result using matplotlib
    img = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Combined OCR Result (Original + Preprocessed)")
    plt.show()
