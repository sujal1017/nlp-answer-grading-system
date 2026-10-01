from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load TrOCR processor
processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

# Load TrOCR model
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

# Load handwritten image
image = Image.open("images/answer.jpeg").convert("RGB")

# Prepare image
pixel_values = processor(
    images=image,
    return_tensors="pt"
).pixel_values

# Generate text
generated_ids = model.generate(pixel_values)

# Convert output to text
text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("\n========== EXTRACTED TEXT ==========\n")
print(text)
print("\n====================================")