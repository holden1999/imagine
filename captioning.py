import os
import requests
from dotenv import load_dotenv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load environment variables
load_dotenv()
data_directory = os.getenv('DATA_DIRECTORY')

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base")

# List all image files in the data directory
image_files = [
    os.path.join(root, file)
    for root, _, files in os.walk(data_directory)
    for file in files
    if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))
]

for img_path in image_files:
    try:
        raw_image = Image.open(img_path).convert('RGB')

        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        print(f"{os.path.basename(img_path)}:",
              processor.decode(out[0], skip_special_tokens=True))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
