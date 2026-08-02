# Skin Cancer Screening App

A full-stack machine learning application that takes a photo of a skin lesion and predicts whether it's likely **benign** or **malignant**, using a Convolutional Neural Network (CNN) built with transfer learning.

**Disclaimer:** This is an educational project built while learning CNNs and full-stack ML deployment. It is **not** a certified medical device and should never be used as a substitute for professional diagnosis. Always consult a dermatologist for real skin concerns.

## About this project

This was built as a first hands-on project in deep learning — going from raw image data all the way to a working web application. It covers the full pipeline:

- Training a CNN on real medical imaging data
- Handling class imbalance in medical datasets
- Serving a trained model through a REST API
- Building a simple frontend that captures/uploads photos and displays predictions

## How it works

1. User uploads or takes a photo of a skin lesion through the web frontend
2. The image is sent to a FastAPI backend
3. The backend resizes and normalizes the image to match training data, then runs it through the trained CNN
4. The model returns a prediction (benign/malignant) with a confidence score, shown to the user

## Tech stack

- **Model:** TensorFlow / Keras — MobileNetV2 (pretrained on ImageNet), fine-tuned via transfer learning
- **Dataset:** [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) — 10,015 dermoscopic images across 7 diagnosis categories, simplified here into benign vs. malignant
- **Training:** Google Colab (free T4 GPU)
- **Backend:** FastAPI (Python)
- ## Model performance

Evaluated on a held-out test set of 2,003 images:

| Metric | Benign | Malignant |
|---|---|---|
| Precision | 0.93 | 0.44 |
| Recall | 0.76 | 0.77 |

The model is tuned to prioritize catching malignant cases (higher recall) even at the cost of more false alarms — the safer failure mode for a screening tool.

## Running it locally

**Backend:**
cd backend
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn python-multipart pillow tensorflow
uvicorn main:app --reload

**Frontend:**
Open `frontend/index.html` in your browser (backend must be running at `http://127.0.0.1:8000`).

## Future improvements

- Data augmentation to improve generalization
- Fine-tuning deeper MobileNetV2 layers
- Live webcam capture instead of file upload only
- Public deployment (currently runs locally only)
- **Frontend:** HTML, CSS, JavaScript (no framework — kept simple)

## Project structure
