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
    try:
        print("=== PREDICT REQUEST STARTED ===")
        print("Filename:", file.filename)
        print("Content type:", file.content_type)

        image_bytes = await file.read()

        print("Image bytes:", len(image_bytes))

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        print("Original image size:", image.size)

        image = image.resize((224, 224))

        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        print("Input shape:", image_array.shape)
        print("Input dtype:", image_array.dtype)

        prediction = model.predict(image_array)

        print("Prediction:", prediction)

        return {
            "prediction": prediction.tolist()
        }

    except Exception as e:
        print("=== PREDICTION ERROR ===")
        print(str(e))
        traceback.print_exc()

        return {
            "error": str(e)
        }