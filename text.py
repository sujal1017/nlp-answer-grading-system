import cv2
import numpy as np
import torch

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ==========================================
# 1. LOAD IMAGE
# ==========================================

image_path = "images/answer.jpeg"

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Image not found!")
    exit()

print("Image loaded successfully.")


# ==========================================
# 2. CONVERT TO GRAYSCALE
# ==========================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ==========================================
# 3. CREATE BINARY IMAGE
# ==========================================

binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)[1]


# ==========================================
# 4. REMOVE HORIZONTAL NOTEBOOK LINES
# ==========================================

horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (50, 1)
)

horizontal_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel
)

clean = cv2.subtract(
    binary,
    horizontal_lines
)


# ==========================================
# 5. JOIN CHARACTERS INTO TEXT LINES
# ==========================================

line_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (25, 5)
)

line_image = cv2.dilate(
    clean,
    line_kernel,
    iterations=1
)


# ==========================================
# 6. FIND TEXT LINE CONTOURS
# ==========================================

contours, _ = cv2.findContours(
    line_image,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


lines = []

for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)

    # Ignore tiny noise
    if w > 100 and h > 10:

        lines.append(
            (x, y, w, h)
        )


# Sort lines from top to bottom

lines.sort(
    key=lambda item: item[1]
)


print("Detected lines:", len(lines))


# ==========================================
# 7. LOAD TROCR
# ==========================================

print("Loading TrOCR model...")

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)


# ==========================================
# 8. SELECT DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

print("Using device:", device)


# ==========================================
# 9. OCR EACH LINE
# ==========================================

height, width = image.shape[:2]

all_text = []


for i, (x, y, w, h) in enumerate(lines):

    # Add padding
    padding_x = 10
    padding_y = 10

    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)

    x2 = min(width, x + w + padding_x)
    y2 = min(height, y + h + padding_y)

    # Crop original image
    line = image[
        y1:y2,
        x1:x2
    ]
    

    # Convert BGR → RGB
    line = cv2.cvtColor(
        line,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(line)

    # Process image
    pixel_values = processor(
        images=pil_image,
        return_tensors="pt"
    ).pixel_values

    pixel_values = pixel_values.to(device)

    # Generate text
    generated_ids = model.generate(
        pixel_values
    )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    text = text.strip()

    if text:

        print(
            f"Line {i + 1}: {text}"
        )

        all_text.append(text)


# ==========================================
# 10. FINAL TEXT
# ==========================================

final_text = "\n".join(
    all_text
)


print()
print("==========================================")
print("             EXTRACTED TEXT")
print("==========================================")
print()

print(final_text)

print()
print("==========================================")