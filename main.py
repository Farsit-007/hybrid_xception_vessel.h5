from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import io
import traceback


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
# IMAGE DECODING
# =========================================================

def load_image(image_bytes: bytes) -> Image.Image:

    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # Verify image integrity
        image.verify()

        # Re-open after verify()
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        return image

    except Exception:

        raise ValueError(
            "Invalid image file. "
            "Please upload a valid JPG, PNG or WEBP image."
        )


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image: Image.Image) -> np.ndarray:

    original_size = image.size

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
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
        f"Original image: {original_size} "
        f"-> Processed: {image_array.shape}"
    )

    return image_array


# =========================================================
# VESSEL EXTRACTION
# =========================================================

def extract_vessels(image: Image.Image) -> Image.Image:
    """
    Extract an approximate retinal vessel map from the
    uploaded retinal image.

    Pipeline:
        RGB
        -> Green channel
        -> CLAHE enhancement
        -> Gaussian blur
        -> BlackHat morphology
        -> Threshold
        -> Morphological cleanup
        -> 3-channel vessel image

    Returns:
        PIL Image in RGB format.
    """

    print("\n----------------------------------------")
    print("STARTING VESSEL EXTRACTION")
    print("----------------------------------------")

    # Convert PIL -> NumPy
    rgb = np.array(image)

    # OpenCV expects RGB/BGR array here
    green_channel = rgb[:, :, 1]

    print(
        "Green channel shape:",
        green_channel.shape
    )

    # -----------------------------------------------------
    # CLAHE
    # -----------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        green_channel
    )

    # -----------------------------------------------------
    # Gaussian blur
    # -----------------------------------------------------

    blurred = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0
    )

    # -----------------------------------------------------
    # BlackHat morphology
    # -----------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    blackhat = cv2.morphologyEx(
        blurred,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    # -----------------------------------------------------
    # Normalize blackhat result
    # -----------------------------------------------------

    blackhat = cv2.normalize(
        blackhat,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    blackhat = blackhat.astype(
        np.uint8
    )

    # -----------------------------------------------------
    # Threshold
    # -----------------------------------------------------

    _, vessel_mask = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # -----------------------------------------------------
    # Morphological cleanup
    # -----------------------------------------------------

    cleanup_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    vessel_mask = cv2.morphologyEx(
        vessel_mask,
        cv2.MORPH_OPEN,
        cleanup_kernel
    )

    vessel_mask = cv2.morphologyEx(
        vessel_mask,
        cv2.MORPH_CLOSE,
        cleanup_kernel
    )

    # -----------------------------------------------------
    # Resize to model size
    # -----------------------------------------------------

    vessel_mask = cv2.resize(
        vessel_mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # -----------------------------------------------------
    # Convert grayscale vessel map to RGB
    # -----------------------------------------------------

    vessel_rgb = cv2.cvtColor(
        vessel_mask,
        cv2.COLOR_GRAY2RGB
    )

    vessel_image = Image.fromarray(
        vessel_rgb
    )

    print(
        "Vessel extraction completed."
    )

    print(
        "Vessel image size:",
        vessel_image.size
    )

    print("----------------------------------------")
    print("VESSEL EXTRACTION FINISHED")
    print("----------------------------------------\n")

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
        "model_inputs": [
            tensor.name.split(":")[0]
            for tensor in model.inputs
        ],
        "model_input_size": "224x224"
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
        # Load original image
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
            "image_input shape:",
            image_array.shape
        )

        # -------------------------------------------------
        # Automatically extract vessels
        # -------------------------------------------------

        vessel_image = extract_vessels(
            original_image
        )

        # -------------------------------------------------
        # Create vessel_input
        # -------------------------------------------------

        vessel_array = preprocess_image(
            vessel_image
        )

        print(
            "vessel_input shape:",
            vessel_array.shape
        )

        # -------------------------------------------------
        # Check model inputs
        # -------------------------------------------------

        model_input_names = [
            tensor.name.split(":")[0]
            for tensor in model.inputs
        ]

        print(
            "Model inputs:",
            model_input_names
        )

        required_inputs = {
            "image_input",
            "vessel_input"
        }

        if not required_inputs.issubset(
            set(model_input_names)
        ):

            raise ValueError(
                "Model input configuration does not "
                "match expected inputs. "
                f"Expected: {list(required_inputs)}, "
                f"Found: {model_input_names}"
            )

        # -------------------------------------------------
        # Run model
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
        # Validate prediction
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
                f"Model returned "
                f"{len(probabilities)} classes, "
                f"but {len(CLASS_NAMES)} class names "
                f"are configured."
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
        # Probability details
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
        # Top 3 predictions
        # -------------------------------------------------

        top_predictions = probability_details[:3]

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

            "top_predictions": top_predictions,

            "probabilities": probability_details,

            "model": {

                "name": "Hybrid Xception Vessel",

                "version": "1.0.0",

                "input_size": "224x224",

                "inputs": [
                    "image_input",
                    "vessel_input"
                ],

                "vessel_extraction": "automatic"

            }

        }

        # -------------------------------------------------
        # Logs
        # -------------------------------------------------

        print(
            "\nPredicted class:",
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

        print(
            "\n========================================"
        )

        print(
            "PREDICTION SUCCESSFUL"
        )

        print(
            "========================================\n"
        )

        return response

    # =====================================================
    # VALIDATION ERROR
    # =====================================================

    except ValueError as e:

        print(
            "\n========================================"
        )

        print(
            "PREDICTION VALIDATION ERROR"
        )

        print(
            "========================================"
        )

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
    # SERVER ERROR
    # =====================================================

    except Exception as e:

        print(
            "\n========================================"
        )

        print(
            "PREDICTION SERVER ERROR"
        )

        print(
            "========================================"
        )

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