#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ======================================================
# PARAMETERS
# ======================================================

INPUT_SIZE = 299
BATCH_SIZE = 16          # reduced to avoid GPU OOM
EPOCHS = 10
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.5
INNER_SIZE = 100

DATA_DIR = "./data/intel-image-classification"
MODEL_PATH = "models/model.keras"

# ======================================================
# GPU CONFIG (IMPORTANT)
# ======================================================

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ======================================================
# DATA GENERATORS
# ======================================================

def create_data_generators(input_size, batch_size):

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    train_ds = train_gen.flow_from_directory(
        os.path.join(DATA_DIR, "seg_train/seg_train"),
        target_size=(input_size, input_size),
        batch_size=batch_size,
        subset="training"
    )

    val_ds = val_gen.flow_from_directory(
        os.path.join(DATA_DIR, "seg_train/seg_train"),
        target_size=(input_size, input_size),
        batch_size=batch_size,
        subset="validation",
        shuffle=False
    )

    return train_ds, val_ds

# ======================================================
# MODEL DEFINITION
# ======================================================

def make_model(input_size, learning_rate, inner_size, dropout_rate):

    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)

    x = keras.layers.GlobalAveragePooling2D()(base)
    x = keras.layers.Dense(inner_size, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    outputs = keras.layers.Dense(6)(x)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
    )

    return model

# ======================================================
# TRAINING LOGIC
# ======================================================

def train():

    print("Creating data generators...")
    train_ds, val_ds = create_data_generators(INPUT_SIZE, BATCH_SIZE)

    print("Building model...")
    model = make_model(
        input_size=INPUT_SIZE,
        learning_rate=LEARNING_RATE,
        inner_size=INNER_SIZE,
        dropout_rate=DROPOUT_RATE
    )

    print("Training model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    return model, history

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    os.makedirs("models", exist_ok=True)

    model, history = train()

    print("Saving model...")
    model.save(MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
