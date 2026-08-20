from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import traceback


# =========================================================
# APP CONFIGURATION
# =========================================================

app = FastAPI(
    title="Vessel Detection API",
    description="AI-powered vessel detection API",
    version="1.0.0"
)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_PATH = "model/hybrid_xception_vessel.h5"

IMAGE_SIZE = (224, 224)

CLASS_NAME= [
    "CSCR",
    "Cataract",
    "Diabetic Retinopathy",
    "Disc Edema",
    "Glaucoma",
    "Healthy",
    "Macular Scar",
    "Retinal Detachment",
    "Retinitis Pigmentosa"
  ]

# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("\n========================================")
    print("MODEL LOADED SUCCESSFULLY")
    print("========================================")

    for tensor in model.inputs:
        print(
            f"Input: {tensor.name} "
            f"| Shape: {tensor.shape} "
            f"| Dtype: {tensor.dtype}"
        )

    print("Output shape:", model.output_shape)

    print("========================================\n")

except Exception as e:
    print("\n========================================")
    print("MODEL LOAD ERROR")
    print("========================================")
    print(str(e))
    traceback.print_exc()
    raise


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Validate and preprocess uploaded image.

    Output shape:
        (1, 224, 224, 3)

    Output dtype:
        float32
    """

    if not image_bytes:
        raise ValueError(
            "Uploaded image is empty."
        )

    try:
        # Open image
        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # Verify actual image file
        image.verify()

        # Re-open because verify() invalidates the image object
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:
        raise ValueError(
            "Invalid image file. "
            "Please upload a valid JPG, PNG or WEBP image."
        )

    original_size = image.size

    # Resize
    image = image.resize(
        IMAGE_SIZE
    )

    # Convert to numpy
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    print(
        f"Image processed: "
        f"{original_size} -> {image_array.shape}"
    )

    return image_array


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "success": True,
        "service": "Vessel Detection API",
        "status": "running",
        "version": "1.0.0"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "success": True,
        "status": "healthy",
        "model_loaded": model is not None
    }


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
async def predict(
    image_file: UploadFile = File(...),
    vessel_file: UploadFile = File(...)
):

    try:

        print("\n========================================")
        print("PREDICTION REQUEST STARTED")
        print("========================================")

        # -------------------------------------------------
        # Read uploaded files
        # -------------------------------------------------

        image_bytes = await image_file.read()
        vessel_bytes = await vessel_file.read()

        print(
            "Image filename:",
            image_file.filename
        )

        print(
            "Image content type:",
            image_file.content_type
        )

        print(
            "Vessel filename:",
            vessel_file.filename
        )

        print(
            "Vessel content type:",
            vessel_file.content_type
        )

        print(
            "Image bytes:",
            len(image_bytes)
        )

        print(
            "Vessel image bytes:",
            len(vessel_bytes)
        )

        # -------------------------------------------------
        # Validate uploaded files
        # -------------------------------------------------

        if not image_bytes:
            raise ValueError(
                "Image file is empty."
            )

        if not vessel_bytes:
            raise ValueError(
                "Vessel image file is empty."
            )

        # -------------------------------------------------
        # Preprocess image input
        # -------------------------------------------------

        image_array = preprocess_image(
            image_bytes
        )

        # -------------------------------------------------
        # Preprocess vessel input
        # -------------------------------------------------

        vessel_array = preprocess_image(
            vessel_bytes
        )

        print(
            "Final image_input shape:",
            image_array.shape
        )

        print(
            "Final vessel_input shape:",
            vessel_array.shape
        )

        # -------------------------------------------------
        # Validate model inputs
        # -------------------------------------------------

        model_input_names = [
            tensor.name.split(":")[0]
            for tensor in model.inputs
        ]

        print(
            "Model expects:",
            model_input_names
        )

        if "image_input" not in model_input_names:
            raise ValueError(
                "Model input 'image_input' was not found."
            )

        if "vessel_input" not in model_input_names:
            raise ValueError(
                "Model input 'vessel_input' was not found."
            )

        # -------------------------------------------------
        # Run prediction
        # -------------------------------------------------

        print(
            "Running model prediction..."
        )

        prediction = model.predict(
            {
                "image_input": image_array,
                "vessel_input": vessel_array
            },
            verbose=0
        )

        print(
            "Raw prediction:",
            prediction
        )

        # -------------------------------------------------
        # Validate prediction output
        # -------------------------------------------------

        if prediction is None:
            raise ValueError(
                "Model returned an empty prediction."
            )

        if len(prediction) == 0:
            raise ValueError(
                "Model returned no prediction results."
            )

        probabilities = prediction[0]

        if len(probabilities) != len(CLASS_NAMES):
            raise ValueError(
                f"Model returned {len(probabilities)} classes, "
                f"but {len(CLASS_NAMES)} class names are configured."
            )

        # -------------------------------------------------
        # Get predicted class
        # -------------------------------------------------

        predicted_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_index]
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        # -------------------------------------------------
        # Build probability list
        # -------------------------------------------------

        probability_details = []

        for index, probability in enumerate(
            probabilities
        ):

            probability = float(
                probability
            )

            probability_details.append({
                "class_index": index,
                "class_name": CLASS_NAMES[index],
                "probability": round(
                    probability,
                    6
                ),
                "percentage": round(
                    probability * 100,
                    2
                )
            })

        # Highest probability first
        probability_details.sort(
            key=lambda item: item["probability"],
            reverse=True
        )

        # -------------------------------------------------
        # Final response
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Logs
        # -------------------------------------------------

        print(
            "Predicted class:",
            predicted_class
        )

        print(
            "Confidence:",
            round(
                confidence * 100,
                2
            ),
            "%"
        )

        print("\n========================================")
        print("PREDICTION SUCCESSFUL")
        print("========================================\n")

        return response

    # =====================================================
    # VALIDATION / PREDICTION ERROR
    # =====================================================

    except ValueError as e:

        print("\n========================================")
        print("PREDICTION VALIDATION ERROR")
        print("========================================")

        print(
            "Error:",
            str(e)
        )

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "message": "Invalid prediction request.",
                "error": str(e)
            }
        )

    # =====================================================
    # UNEXPECTED SERVER ERROR
    # =====================================================

    except Exception as e:

        print("\n========================================")
        print("PREDICTION SERVER ERROR")
        print("========================================")

        print(
            "Error:",
            str(e)
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Prediction failed due to a server error.",
                "error": str(e)
            }
        )