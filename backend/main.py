from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("skin_cancer_model.keras")
IMG_SIZE = 75

@app.get("/")
def read_root():
    return {"message": "Skin cancer prediction API is running", "model_loaded": True}

def preprocess_image(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    processed = preprocess_image(image)

    prediction = model.predict(processed)
    probability = float(prediction[0][0])

    if probability > 0.5:
        label = "malignant"
        confidence = probability
    else:
        label = "benign"
        confidence = 1 - probability

    return {
        "filename": file.filename,
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "raw_probability": round(probability, 4)
    }
