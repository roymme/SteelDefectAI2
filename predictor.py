"""
predictor.py
---------------------------------------
ResNet50 Predictor for Steel Defect Classification
"""
from PIL import ImageOps
import os
import requests
import json
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "resnet50_gc10_new.keras"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "model",
    "class_names.json"
)

# =====================================================
# Check files
# =====================================================

print("=" * 60)
print("Steel Defect Predictor")
print("=" * 60)

print("\nChecking files...")

MODEL_URL = "https://huggingface.co/Shashwata17/steel_defect_detection/resolve/main/resnet50_gc10_new.keras"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(MODEL_PATH):

    print("Downloading model from Hugging Face...")

    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    downloaded = 0

    with open(MODEL_PATH, "wb") as f:

        for chunk in response.iter_content(chunk_size=8192):

            if chunk:

                f.write(chunk)

                downloaded += len(chunk)

                if total_size:
                    percent = downloaded / total_size * 100
                    print(f"\rDownloading: {percent:.1f}%", end="")

    print("\n✓ Model Downloaded Successfully")

else:

    print("✓ Model Found")

if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError(
        f"\nClass JSON not found:\n{CLASS_PATH}"
    )

print("✓ Class Names Found")

# =====================================================
# Load model
# =====================================================

print("\nLoading ResNet50 Model...")

MODEL = tf.keras.models.load_model(MODEL_PATH)

print("✓ Model Loaded Successfully")

# =====================================================
# Load class names
# =====================================================

print("\nLoading Class Names...")

with open(CLASS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

print(f"✓ Loaded {len(CLASS_NAMES)} Classes")

print("=" * 60)

# =====================================================
# Prediction
# =====================================================

def predict_image(image: Image.Image):
    print("="*50)
    print("Mode:", image.mode)
    print("Size:", image.size)

    x = np.array(image)

    print("Shape:", x.shape)
    print("Min:", x.min())
    print("Max:", x.max())
    print("="*50)
    image = ImageOps.exif_transpose(image) 
    image = image.convert("RGB")

    image = image.resize((224, 224))

    x = np.array(image, dtype=np.float32)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    prediction = MODEL.predict(
        x,
        verbose=0
    )[0]

    predicted_index = np.argmax(prediction)

    confidence = float(prediction[predicted_index])

    top3_idx = prediction.argsort()[-3:][::-1]

    return {

        "predicted_class":
            CLASS_NAMES[predicted_index],

        "confidence":
            confidence,

        "top3":[

            {

                "class":
                    CLASS_NAMES[i],

                "probability":
                    float(prediction[i])

            }

            for i in top3_idx

        ]

    }



def predict_from_path(image_path):

    image = Image.open(image_path)

    return predict_image(image)



if __name__ == "__main__":

    print("\nPredictor loaded successfully.")

    print("You can now import this file into Streamlit.")
