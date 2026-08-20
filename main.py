from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import traceback

app = FastAPI()

MODEL_PATH = "model/hybrid_xception_vessel.h5"


# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# =========================
# MODEL INFORMATION
# =========================

print("\n===== MODEL INPUTS =====")

for x in model.inputs:
    print(
        "Name:",
        x.name,
        "| Shape:",
        x.shape,
        "| Dtype:",
        x.dtype
    )

print("Output shape:", model.output_shape)

print("========================\n")


# =========================
# HEALTH CHECK
# =========================

@app.get("/")
def root():
    return {
        "message": "Vessel Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }


# =========================
# IMAGE PREPROCESSING
# =========================

def preprocess_image(image_bytes: bytes):
    """
    Convert uploaded image bytes into
    model-compatible tensor.

    Output shape:
    (1, 224, 224, 3)
    """

    if not image_bytes:
        raise ValueError("Uploaded file is empty.")

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    print("Original image size:", image.size)

    # Resize
    image = image.resize((224, 224))

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
        "Processed image shape:",
        image_array.shape
    )

    print(
        "Processed image dtype:",
        image_array.dtype
    )

    return image_array


# =========================
# PREDICTION
# =========================

@app.post("/predict")
async def predict(
    image_file: UploadFile = File(...),
    vessel_file: UploadFile = File(...)
):

    try:

        print("\n==============================")
        print("=== PREDICT REQUEST STARTED ===")
        print("==============================")

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


        # =========================
        # READ IMAGE FILE
        # =========================

        image_bytes = await image_file.read()

        print(
            "Image bytes:",
            len(image_bytes)
        )


        # =========================
        # READ VESSEL FILE
        # =========================

        vessel_bytes = await vessel_file.read()

        print(
            "Vessel image bytes:",
            len(vessel_bytes)
        )


        # =========================
        # PREPROCESS IMAGE
        # =========================

        image_array = preprocess_image(
            image_bytes
        )


        # =========================
        # PREPROCESS VESSEL IMAGE
        # =========================

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


        # =========================
        # MODEL INPUT CHECK
        # =========================

        input_names = [
            tensor.name.split(":")[0]
            for tensor in model.inputs
        ]

        print(
            "Model expects:",
            input_names
        )


        if "image_input" not in input_names:
            raise ValueError(
                "'image_input' was not found in model."
            )

        if "vessel_input" not in input_names:
            raise ValueError(
                "'vessel_input' was not found in model."
            )


        # =========================
        # PREDICTION
        # =========================

        print("\nRunning model prediction...")

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


        # =========================
        # CONVERT PREDICTION
        # =========================

        prediction_list = prediction.tolist()


        print(
            "Prediction:",
            prediction_list
        )

        print("==============================")
        print("=== PREDICTION SUCCESSFUL ===")
        print("==============================\n")


        return {
            "success": True,
            "prediction": prediction_list
        }


    except Exception as e:

        print("\n==============================")
        print("=== PREDICTION ERROR ===")
        print("==============================")

        print(
            "Error:",
            str(e)
        )

        traceback.print_exc()

        print("==============================\n")


        raise HTTPException(
            status_code=500,
            detail=str(e)
        )