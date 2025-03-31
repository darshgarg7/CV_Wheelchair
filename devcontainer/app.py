import os
import logging
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
from tensorflow.keras.models import load_model
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

HOST = os.getenv("APP_HOST", "0.0.0.0")
PORT = int(os.getenv("APP_PORT", 8000))
MODEL_PATH = os.getenv("MODEL_PATH", "models/gesture_recognition_model.h5")

try:
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the service is running.
    Returns additional details about the service's status.
    """
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "host": HOST,
        "port": PORT
    }
    if model is None:
        status["error"] = "Model failed to load."
        return jsonify(status), 503  # Service Unavailable if the model isn't loaded
    return jsonify(status), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint to serve predictions using the trained model.
    Expects a JSON payload with an image array.
    """
    if model is None:
        logger.error("Prediction failed: Model is not loaded.")
        return jsonify({"error": "Model is not loaded."}), 503

    try:
        data = request.json
        if "image" not in data:
            raise ValueError("Missing 'image' field in request payload.")

        # Preprocess the input image
        image = np.array(data["image"], dtype=np.float32) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][predicted_class])

        response = {
            "class": int(predicted_class),
            "confidence": confidence
        }
        return jsonify(response), 200

    except ValueError as ve:
        logger.error(f"Invalid input: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """
    Error handler for HTTP exceptions.
    Returns a JSON response with the error message and status code.
    """
    response = {
        "error": e.description,
        "status_code": e.code
    }
    return jsonify(response), e.code


@app.errorhandler(Exception)
def handle_generic_exception(e):
    """
    Error handler for generic exceptions.
    Logs the error and returns a generic error response.
    """
    logger.error(f"Unhandled exception: {e}")
    return jsonify({"error": "An unexpected error occurred."}), 500


if __name__ == '__main__':
    # Run the app
    logger.info(f"Starting Flask app on {HOST}:{PORT}...")
    app.run(host=HOST, port=PORT, debug=False)
