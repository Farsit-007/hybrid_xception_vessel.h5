from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import traceback

app = FastAPI(
    title="Vessel Detection API",
    description="AI-powered vessel detection API",
    version="1.0.0"
)

MODEL_PATH = "model/hybrid_xception_vessel.h5"

# =========================
# CLASS LABELS
# =========================

CLASS_NAMES = [
    "class_0",
    "class_1",
    "class_2",
    "class_3",
    "class_4",
    "class_5",
    "class_6",
    "class_7",
    "class_8",
]


# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\n===== MODEL INFORMATION =====")

for tensor in model.inputs:
    print(
        "Input:",
        tensor.name,
        "| Shape:",
        tensor.shape,
        "| Dtype:",
        tensor.dtype
    )

print("Output shape:", model.output_shape)

print("=============================\n")


# =========================
# IMAGE PREPROCESSING
# =========================

def preprocess_image(image_bytes: bytes) -> np.ndarray:

    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    original_size = image.size

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    print(
        f"Image processed: "
        f"{original_size} -> {image_array.shape}"
    )

    return image_array


# =========================
# ROOT
# =========================

@app.get("/")
def root():

    return {
        "success": True,
        "service": "Vessel Detection API",
        "status": "running",
        "version": "1.0.0"
    }


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy",
        "model_loaded": model is not None
    }


# =========================
# PREDICTION
# =========================

@app.post("/predict")
async def predict(
    image_file: UploadFile = File(...),
    vessel_file: UploadFile = File(...)
):

    try:

        print("\n================================")
        print("       PREDICTION REQUEST")
        print("================================")

        # -------------------------
        # Validate file types
        # -------------------------

        allowed_types = {
            "image/jpeg",
            "image/png",
            "image/jpg",
            "image/webp"
        }

        if image_file.content_type not in allowed_types:
            raise ValueError(
                "Invalid image_file format. "
                "Please upload JPG, PNG or WEBP."
            )

        if vessel_file.content_type not in allowed_types:
            raise ValueError(
                "Invalid vessel_file format. "
                "Please upload JPG, PNG or WEBP."
            )

        # -------------------------
        # Read files
        # -------------------------

        image_bytes = await image_file.read()
        vessel_bytes = await vessel_file.read()

        print(
            "Original image:",
            image_file.filename
        )

        print(
            "Vessel image:",
            vessel_file.filename
        )

        # -------------------------
        # Preprocess
        # -------------------------

        image_array = preprocess_image(
            image_bytes
        )

        vessel_array = preprocess_image(
            vessel_bytes
        )

        # -------------------------
        # Model prediction
        # -------------------------

        prediction = model.predict(
            {
                "image_input": image_array,
                "vessel_input": vessel_array
            },
            verbose=0
        )

        # -------------------------
        # Prediction processing
        # -------------------------

        probabilities = prediction[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            np.max(probabilities)
        )

        # Safety check
        if predicted_index >= len(CLASS_NAMES):
            raise ValueError(
                "Predicted class index is outside "
                "the configured class labels."
            )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        # -------------------------
        # Probability details
        # -------------------------

        probability_details = []

        for index, probability in enumerate(
            probabilities
        ):

            probability_details.append({
                "class_index": index,
                "class_name": CLASS_NAMES[index],
                "probability": round(
                    float(probability),
                    6
                ),
                "percentage": round(
                    float(probability) * 100,
                    2
                )
            })

        # Sort highest probability first
        probability_details.sort(
            key=lambda item: item["probability"],
            reverse=True
        )

        # -------------------------
        # Final response
        # -------------------------

        response = {
            "success": True,
            "message": "Prediction completed successfully.",
            "prediction": {
                "class_index": predicted_index,
                "class_name": predicted_class,
                "confidence": round(
                    confidence,
                    6
                ),
                "confidence_percentage": round(
                    confidence * 100,
                    2
                )
            },
            "probabilities": probability_details,
            "model": {
                "name": "Hybrid Xception Vessel",
                "input_size": "224x224",
                "inputs": [
                    "image_input",
                    "vessel_input"
                ]
            }
        }

        print(
            "Predicted class:",
            predicted_class
        )

        print(
            "Confidence:",
            round(confidence * 100, 2),
            "%"
        )

        print("================================")
        print("       PREDICTION SUCCESS")
        print("================================\n")

        return response

    except Exception as e:

        print("\n================================")
        print("       PREDICTION ERROR")
        print("================================")

        print("Error:", str(e))

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Prediction failed.",
                "error": str(e)
            }
        )