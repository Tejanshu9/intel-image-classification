# ======================================================
# Base Image (TensorFlow-compatible)
# ======================================================
FROM python:3.11-slim

# ======================================================
# System dependencies (needed for image processing)
# ======================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ======================================================
# Set working directory
# ======================================================
WORKDIR /app

# ======================================================
# Copy requirements and install dependencies
# ======================================================
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ======================================================
# Copy application code and model
# ======================================================
COPY src/predict.py .
COPY models/model.keras models/model.keras

# ======================================================
# Expose API port
# ======================================================
EXPOSE 9696

# ======================================================
# Run with gunicorn
# ======================================================
CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
