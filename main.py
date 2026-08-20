from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import tensorflow as tf
import io
import traceback
import gc


# =========================================================
# APP CONFIGURATION
# =========================================================

app = FastAPI(
    title="Retinal Disease Detection API",
    description=(
        "AI-powered retinal disease classification API "
        "using a Hybrid Xception model with automatic vessel extraction."
    ),
    version="1.0.0"
)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_PATH = "model/hybrid_xception_vessel.h5"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
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
# MODEL INPUT VALIDATION
# =========================================================

MODEL_INPUT_NAMES = [
    tensor.name.split(":")[0]
    for tensor in model.inputs
]

EXPECTED_INPUTS = {
    "image_input",
    "vessel_input"
}

if not EXPECTED_INPUTS.issubset(set(MODEL_INPUT_NAMES)):

    raise RuntimeError(
        "Model input configuration is not supported. "
        f"Expected inputs: {EXPECTED_INPUTS}. "
        f"Found: {MODEL_INPUT_NAMES}"
    )

if len(CLASS_NAMES) != model.output_shape[-1]:

    raise RuntimeError(
        f"Class configuration mismatch. "
        f"Model outputs {model.output_shape[-1]} classes, "
        f"but {len(CLASS_NAMES)} class names are configured."
    )


# =========================================================
# IMAGE LOADING
# =========================================================

def load_image(image_bytes: bytes) -> Image.Image:

    if not image_bytes:
        raise ValueError(
            "Uploaded image is empty."
        )

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

        image = image.convert("RGB")

        return image

    except Exception:

        raise ValueError(
            "Invalid image file. "
            "Please upload a valid JPG, PNG or WEBP image."
        )


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(
    image: Image.Image
) -> np.ndarray:

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array *= (1.0 / 255.0)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# LIGHTWEIGHT VESSEL EXTRACTION
# =========================================================

def extract_vessels(
    image: Image.Image
) -> Image.Image:

    """
    Lightweight retinal vessel extraction.

    Pipeline:

        RGB
          ↓
        Green channel
          ↓
        Contrast enhancement
          ↓
        Background smoothing
          ↓
        Vessel enhancement
          ↓
        Threshold
          ↓
        Binary vessel map
          ↓
        RGB

    This implementation intentionally avoids OpenCV
    to reduce memory usage on low-memory hosting.
    """

    # -----------------------------------------------------
    # Resize before processing
    # -----------------------------------------------------

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    # -----------------------------------------------------
    # Convert to grayscale
    # -----------------------------------------------------

    gray = ImageOps.grayscale(image)

    # -----------------------------------------------------
    # Enhance local contrast
    # -----------------------------------------------------

    enhanced = ImageOps.autocontrast(gray)

    # -----------------------------------------------------
    # Create blurred background
    # -----------------------------------------------------

    blurred = enhanced.filter(
        ImageFilter.GaussianBlur(radius=5)
    )

    # -----------------------------------------------------
    # Convert to NumPy
    # -----------------------------------------------------

    enhanced_array = np.asarray(
        enhanced,
        dtype=np.float32
    )

    blurred_array = np.asarray(
        blurred,
        dtype=np.float32
    )

    # -----------------------------------------------------
    # Vessel enhancement
    #
    # Blood vessels are darker than surrounding retina.
    # Therefore:
    #
    # background - image
    #
    # produces a vessel-like response.
    # -----------------------------------------------------

    vessel_response = (
        blurred_array - enhanced_array
    )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    min_value = vessel_response.min()
    max_value = vessel_response.max()

    if max_value > min_value:

        vessel_response = (
            vessel_response - min_value
        ) / (
            max_value - min_value
        )

    else:

        vessel_response = np.zeros_like(
            vessel_response
        )

    # -----------------------------------------------------
    # Threshold
    # -----------------------------------------------------

    threshold = np.percentile(
        vessel_response,
        85
    )

    vessel_mask = (
        vessel_response > threshold
    ).astype(np.uint8) * 255

    # -----------------------------------------------------
    # Convert mask to RGB
    # -----------------------------------------------------

    vessel_image = Image.fromarray(
        vessel_mask,
        mode="L"
    ).convert("RGB")

    return vessel_image


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "success": True,
        "service": "Retinal Disease Detection API",
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
        "model_loaded": model is not None,
        "model_inputs": MODEL_INPUT_NAMES,
        "input_size": "224x224",
        "vessel_extraction": "automatic"
    }


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
async def predict(
    image_file: UploadFile = File(...)
):

    try:

        print("\n========================================")
        print("PREDICTION REQUEST STARTED")
        print("========================================")

        # -------------------------------------------------
        # File information
        # -------------------------------------------------

        print(
            "Filename:",
            image_file.filename
        )

        print(
            "Content type:",
            image_file.content_type
        )

        # -------------------------------------------------
        # Read uploaded image
        # -------------------------------------------------

        image_bytes = await image_file.read()

        print(
            "Image bytes:",
            len(image_bytes)
        )

        if not image_bytes:

            raise ValueError(
                "Uploaded image is empty."
            )

        # -------------------------------------------------
        # Load image
        # -------------------------------------------------

        original_image = load_image(
            image_bytes
        )

        print(
            "Original image size:",
            original_image.size
        )

        # -------------------------------------------------
        # Create image_input
        # -------------------------------------------------

        image_array = preprocess_image(
            original_image
        )

        print(
            "image_input:",
            image_array.shape,
            image_array.dtype
        )

        # -------------------------------------------------
        # Automatically extract vessels
        # -------------------------------------------------

        print(
            "Extracting vessels automatically..."
        )

        vessel_image = extract_vessels(
            original_image
        )

        print(
            "Vessel extraction completed."
        )

        # -------------------------------------------------
        # Create vessel_input
        # -------------------------------------------------

        vessel_array = preprocess_image(
            vessel_image
        )

        print(
            "vessel_input:",
            vessel_array.shape,
            vessel_array.dtype
        )

        # -------------------------------------------------
        # Release unnecessary image objects
        # -------------------------------------------------

        del vessel_image
        del original_image
        del image_bytes

        gc.collect()

        # -------------------------------------------------
        # Run model prediction
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

        # -------------------------------------------------
        # Release input arrays after prediction
        # -------------------------------------------------

        del image_array
        del vessel_array

        gc.collect()

        # -------------------------------------------------
        # Validate prediction
        # -------------------------------------------------

        if prediction is None:

            raise ValueError(
                "Model returned an empty prediction."
            )

        probabilities = np.asarray(
            prediction[0]
        )

        if len(probabilities) != len(CLASS_NAMES):

            raise ValueError(
                f"Model returned {len(probabilities)} "
                f"classes, but {len(CLASS_NAMES)} "
                f"class names are configured."
            )

        # -------------------------------------------------
        # Predicted class
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
        # Probability details
        # -------------------------------------------------

        probability_details = []

        for index, probability in enumerate(
            probabilities
        ):

            probability = float(
                probability
            )

            probability_details.append(
                {
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
                }
            )

        # -------------------------------------------------
        # Sort highest probability first
        # -------------------------------------------------

        probability_details.sort(
            key=lambda item: item["probability"],
            reverse=True
        )

        # -------------------------------------------------
        # Top 3
        # -------------------------------------------------

        top_predictions = (
            probability_details[:3]
        )

        # -------------------------------------------------
        # Final response
        # -------------------------------------------------

        response = {

            "success": True,

            "message": (
                "Retinal image analyzed successfully."
            ),

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

            "top_predictions": top_predictions,

            "probabilities": probability_details,

            "model": {

                "name": (
                    "Hybrid Xception Vessel"
                ),

                "version": "1.0.0",

                "input_size": "224x224",

                "inputs": [
                    "image_input",
                    "vessel_input"
                ],

                "vessel_extraction": {
                    "automatic": True,
                    "source": "uploaded_image"
                }
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
    # VALIDATION ERROR
    # =====================================================

    except ValueError as e:

        print(
            "\nPREDICTION VALIDATION ERROR:",
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
    # SERVER ERROR
    # =====================================================

    except Exception as e:

        print(
            "\nPREDICTION SERVER ERROR:",
            str(e)
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": (
                    "Prediction failed due to a server error."
                ),
                "error": str(e)
            }
        )