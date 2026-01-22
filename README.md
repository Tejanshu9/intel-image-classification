# Intel Image Classification (CNN + Fly.io Deployment)

This project is an end-to-end **image classification service** built using **transfer learning with Xception**, trained on the **Intel Image Classification dataset**, and deployed as a **containerized inference API on Fly.io**. The service accepts an image file and returns the predicted class along with confidence scores.

---

## 🚀 Project Overview

- **Model**: Xception (ImageNet pretrained, frozen base)
- **Framework**: TensorFlow / Keras
- **Deployment**: Docker + Gunicorn + Fly.io
- **Inference API**: Flask
- **Input**: Image file (`.jpg`, `.png`)
- **Output**: Predicted class + confidence

---

## 🗂 Dataset

**Intel Image Classification Dataset**

Classes:
- buildings
- forest
- glacier
- mountain
- sea
- street

Images are resized to **299 × 299**, matching Xception's expected input.

---

## 🧠 Model Architecture

The model uses transfer learning with the following architecture:

- **Base**: Xception (pretrained on ImageNet, frozen)
- **Global Average Pooling**: Reduces spatial dimensions
- **Dense Layer**: 100 units with ReLU activation
- **Dropout**: 0.2 rate for regularization
- **Output Layer**: 6 units (one per class)

**Loss Function:**
- `CategoricalCrossentropy(from_logits=True)`

**Optimizer:**
- Adam with learning rate `1e-3`

---

## 🏋️ Training Summary

Training configuration from `train.py`:

- **Input size**: 299 × 299
- **Batch size**: 16
- **Epochs**: 20
- **Learning rate**: 0.001
- **Dropout rate**: 0.2
- **Inner layer size**: 100 units
- **Data augmentation**: Shear, zoom, horizontal flip
- **Validation split**: 20%
- **Checkpoint**: Best model saved based on `val_loss`

**Model saved as**: `models/model.keras`

Training was done on GPU. Inference runs on CPU.

---

## 📦 Project Structure
```
intel-image-classification/
├── train.py                     # Training script
├── predict.py                   # Flask inference API
├── Dockerfile                   # Container definition
├── fly.toml                     # Fly.io configuration
├── requirements.txt             # Python dependencies
├── notebook.ipynb              # Exploratory notebook
├── .gitignore                   # Git ignore rules
├── 30.jpg                       # Test image
├── data/
│   └── intel-image-classification/
│       └── seg_train/seg_train/ # Training data
├── models/
│   └── model.keras              # Trained model
└── README.md                    # This file
```

---

## 🔌 API Specification

### Endpoint
```
POST /predict
```

### Request

- **Content-Type**: `multipart/form-data`
- **Field name**: `file`
- **Value**: Image file (`.jpg`, `.png`)

### Response (JSON)
```json
{
  "class": "forest",
  "confidence": 0.94
}
```


---

## 🧪 Local Testing (without Docker)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Flask server
```bash
python predict.py
```

The server runs on port `9696`.

### 3. Test the endpoint
```bash
curl -X POST \
  -F "file=@30.jpg" \
  http://localhost:9696/predict
```

**Expected output:**
```json
{
  "class": "mountain",
  "confidence": 0.97
}
```

---

## 🐳 Docker Usage

### Build the image
```bash
docker build -t intel-image-classifier .
```

### Run the container
```bash
docker run -p 9696:9696 intel-image-classifier
```

### Test the container
```bash
curl -X POST \
  -F "file=@30.jpg" \
  http://localhost:9696/predict
```

---

## ☁️ Fly.io Deployment

The application is deployed on **Fly.io** using Docker and Gunicorn.

### Prerequisites
```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login
```

### Deploy
```bash
flyctl deploy
```

Fly.io automatically:
- Builds the Docker image
- Pushes it to the registry
- Deploys to a global edge network

---



📸 **Screenshot**

> *(Insert terminal screenshot here showing the `curl` request and response)*

---

## ⚙️ Requirements

Key dependencies:
```
tensorflow==2.15.0
flask==3.0.0
gunicorn==21.2.0
pillow==10.1.0
numpy==1.26.2
```

All dependencies are listed in `requirements.txt`.

---


## 🎓 Training Your Own Model

To retrain the model:

### 1. Download the dataset

Download the Intel Image Classification dataset and place it in:
```
data/intel-image-classification/seg_train/seg_train/
```

### 2. Run training
```bash
python train.py
```

**Training features:**
- Uses `ModelCheckpoint` to save best model
- Data augmentation (shear, zoom, flip)
- Reproducible (fixed random seed: 42)
- GPU memory growth enabled
- 20% validation split

**Output:**
- Best model saved to `models/model.keras`
- Training history logged to console

---

## 📌 Key Learnings

* Transfer learning with frozen Xception base dramatically reduces training time
* Correct image preprocessing (`preprocess_input`) is critical for accuracy
* Data augmentation helps prevent overfitting on small datasets
* Containerization with Docker simplifies deployment and dependency management
* GPU is only needed for training, not inference
* Gunicorn provides production-ready WSGI server
* Fly.io deployment issues are usually **DevOps**, not ML

---

## 🐛 Common Issues

### Docker build fails
- Ensure `models/model.keras` exists
- Check Docker daemon is running

### Model loading error
- Verify TensorFlow version matches training version
- Check model file path

### Port already in use
- Change port in `predict.py` or use different port mapping

---

## 🧾 License

This project is for educational purposes as part of **ML Zoomcamp**.

---

## ✨ Acknowledgements

* [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
* TensorFlow & Keras
* Fly.io
* [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)

---
