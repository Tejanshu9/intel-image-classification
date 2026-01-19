#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input
from flask import Flask, request, jsonify

# ======================================================
# PARAMETERS
# ======================================================

MODEL_PATH = "models/model.keras"
INPUT_SIZE = 299
CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

# ======================================================
# LOAD MODEL (ONCE, AT STARTUP)
# ======================================================

model = keras.models.load_model(MODEL_PATH)

# ======================================================
# FLASK APP
# ======================================================

app = Flask("intel-image-classification")

# ======================================================
# IMAGE PREPROCESSING FUNCTION
# ======================================================

def prepare_image(image_path):
    """
    Load image from disk and prepare it for prediction
    """
    img = load_img(image_path, target_size=(INPUT_SIZE, INPUT_SIZE))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# ======================================================
# PREDICTION ENDPOINT
# ======================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON with:
    {
        "image_path": "/path/to/image.jpg"
    }
    """

    data = request.get_json()

    if "image_path" not in data:
        return jsonify({"error": "image_path is required"}), 400

    image_path = data["image_path"]

    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 404

    x = prepare_image(image_path)

    preds = model.predict(x)
    probs = tf.nn.softmax(preds[0]).numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    result = {
        "class": pred_class,
        "confidence": confidence
    }

    return jsonify(result)

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
