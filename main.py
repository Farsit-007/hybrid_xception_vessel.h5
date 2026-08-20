from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

MODEL_PATH = "model/hybrid_xception_vessel.h5"

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


@app.get("/")
def root():
    return {
        "message": "Vessel Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # IMPORTANT:
    # এখানে তোমার model-এর actual input size বসাতে হবে
    image = image.resize((224, 224))

    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)

    return {
        "prediction": prediction.tolist()
    }