from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import traceback

app = FastAPI()

MODEL_PATH = "model/hybrid_xception_vessel.h5"

# Load model
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
# PREDICTION
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        print("\n=== PREDICT REQUEST STARTED ===")

        print("Filename:", file.filename)
        print("Content type:", file.content_type)

        # -------------------------
        # Read image
        # -------------------------

        image_bytes = await file.read()

        print("Image bytes:", len(image_bytes))

        if not image_bytes:
            raise ValueError("Uploaded file is empty.")

        # -------------------------
        # Open image
        # -------------------------

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        print("Original image size:", image.size)

        # -------------------------
        # Resize
        # -------------------------

        image = image.resize((224, 224))

        # -------------------------
        # Convert to numpy
        # -------------------------

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
            "Image input shape:",
            image_array.shape
        )

        print(
            "Image input dtype:",
            image_array.dtype
        )

        # -------------------------
        # Check model inputs
        # -------------------------

        input_names = [
            tensor.name.split(":")[0]
            for tensor in model.inputs
        ]

        print(
            "Model expects inputs:",
            input_names
        )

        # -------------------------
        # IMPORTANT
        # -------------------------

        if len(model.inputs) != 2:

            raise ValueError(
                f"Expected 2 model inputs, "
                f"but model has {len(model.inputs)} inputs."
            )

        if "image_input" not in input_names:

            raise ValueError(
                f"'image_input' not found. "
                f"Model inputs: {input_names}"
            )

        if "vessel_input" not in input_names:

            raise ValueError(
                f"'vessel_input' not found. "
                f"Model inputs: {input_names}"
            )

        # -------------------------
        # We currently DON'T know
        # vessel_input preprocessing
        # -------------------------

        vessel_tensor = None

        raise ValueError(
            "Model requires a second input named "
            "'vessel_input'. Its preprocessing/shape "
            "is not yet configured. Check Render logs "
            "for the MODEL INPUTS section."
        )

    except Exception as e:

        print("\n=== PREDICTION ERROR ===")

        print("Error:", str(e))

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )