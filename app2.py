
import os
import io
import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify, render_template

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
MODEL_PATH = os.path.join(BASE_DIR, "breast_cancer_model.h5")

# Initialize Flask
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR
)

# Load the trained CNN model
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Home page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Image prediction
@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({
            "error": "No image provided. Please select an image."
        }), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({
            "error": "Please select an image first."
        }), 400

    try:
        image_bytes = uploaded_file.read()

        if not image_bytes:
            return jsonify({
                "error": "The uploaded image is empty."
            }), 400

        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        image = image.resize((50, 50))

        image_array = np.asarray(
            image, dtype=np.float32
        ) / 255.0

        image_array = np.expand_dims(
            image_array, axis=0
        )

        prediction = float(
            model.predict(image_array, verbose=0)[0][0]
        )

        if prediction > 0.5:
            predicted_class = "IDC"
            confidence = prediction
        else:
            predicted_class = "No IDC"
            confidence = 1.0 - prediction

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 4)
        })

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "error": "Invalid image. Upload a valid JPG or PNG image."
        }), 400

    except Exception:
        app.logger.exception("Prediction error")
        return jsonify({
            "error": "Prediction failed. Check the VS Code terminal."
        }), 500

# Start server
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
