# ======================================================
# Base Image
# ======================================================
FROM python:3.11-slim

# ======================================================
# System dependencies
# ======================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ======================================================
# Working directory
# ======================================================
WORKDIR /app

# ======================================================
# Install Python dependencies
# ======================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ======================================================
# Copy application code and model
# ======================================================
RUN mkdir -p models
COPY predict.py .
COPY models/model.keras models/model.keras

# ======================================================
# Non-root user (recommended)
# ======================================================
RUN useradd -m appuser
USER appuser

# ======================================================
# Expose API port
# ======================================================
EXPOSE 9696

# ======================================================
# Run with Gunicorn
# ======================================================
CMD ["python", "-m", "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

