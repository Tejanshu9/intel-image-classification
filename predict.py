#!/usr/bin/env python
# coding: utf-8

import io
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
# LOAD MODEL (ONCE AT STARTUP)
# ======================================================

model = keras.models.load_model(MODEL_PATH)

# ======================================================
# FLASK APP
# ======================================================

app = Flask("intel-image-classification")

# ======================================================
# IMAGE PREPROCESSING
# ======================================================

def prepare_image(file):
    """
    Convert uploaded file to model-ready tensor
    """
    img = load_img(
        io.BytesIO(file.read()),
        target_size=(INPUT_SIZE, INPUT_SIZE)
    )
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# ======================================================
# PREDICTION ENDPOINT
# ======================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "file is required"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    x = prepare_image(file)

    preds = model.predict(x)
    probs = tf.nn.softmax(preds[0]).numpy()

    pred_idx = int(np.argmax(probs))

    result = {
        "class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx])
    }

    return jsonify(result)

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
