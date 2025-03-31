import cv2
import numpy as np
from tensorflow.keras.models import load_model
from typing import Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GestureRecognition:
    """
    Class for gesture recognition using a pre-trained machine learning model.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.6, temperature: float = 1.0):
        """
        Initializes the GestureRecognition class with the specified model.
        
        :param model_path: Path to the pre-trained model.
        :param confidence_threshold: Threshold for confidence in predictions (default: 0.6).
        :param temperature: Softmax temperature scaling factor (default: 1.0).
        """
        self.model = self._load_model(model_path)
        self.class_names = self._load_class_names()
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature

    @staticmethod
    def _load_model(model_path: str):
        """
        Loads the gesture recognition model.
        
        :param model_path: Path to the model file.
        :return: Loaded model.
        :raises RuntimeError: If model loading fails.
        """
        try:
            logger.info(f"Loading model from {model_path}...")
            return load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    @staticmethod
    def _load_class_names() -> Dict[int, str]:
        """
        Loads class names for gesture recognition.
        
        :return: Dictionary of class names.
        """
        return {
            0: "Stop",
            1: "Go",
            2: "Turn Left",
            3: "Turn Right",
        }

    def predict_gesture(self, frame: np.ndarray) -> Optional[str]:
        """
        Predicts the gesture from the provided video frame.
        
        :param frame: Input frame from video or image.
        :return: Predicted gesture as a string if confidence is above threshold, else None.
        """
        try:
            processed_frame = self._preprocess_frame(frame)
            predictions = self.model.predict(processed_frame)
            class_index = np.argmax(predictions)
            confidence = predictions[0][class_index]

            # Confidence threshold adjustment using softmax temperature scaling
            scaled_confidence = confidence / self.temperature

            # Compare scaled confidence against the threshold
            if scaled_confidence > self.confidence_threshold:
                return self.class_names.get(class_index)
            return None
        except Exception as e:
            logger.error(f"Error in gesture prediction: {e}", exc_info=True)
            return None

    @staticmethod
    def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
        """
        Preprocesses the input frame for the model.
        
        :param frame: Input frame from video or image.
        :return: Preprocessed frame ready for model input.
        """
        try:
            resized_frame = cv2.resize(frame, (224, 224))  # Resize to match training dimensions
            normalized_frame = resized_frame / 255.0  # Normalize pixel values
            return np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
        except Exception as e:
            logger.error(f"Error during frame preprocessing: {e}", exc_info=True)
            raise ValueError("Frame preprocessing failed. Ensure the frame is correctly sized.")

# Initialize GestureRecognition
gesture_recognizer = GestureRecognition(
    model_path="models/gesture_recognition_model.h5",
    confidence_threshold=0.7,
    temperature=1.0
)

# Capture a frame from the camera
frame = capture_frame_from_camera()

# Predict the gesture
recognized_gesture = gesture_recognizer.predict_gesture(frame)

if recognized_gesture:
    print(f"Recognized Gesture: {recognized_gesture}")
else:
    print("Gesture not recognized due to low confidence.")
