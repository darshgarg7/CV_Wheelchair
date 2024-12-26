import cv2
import numpy as np
from tensorflow.keras.models import load_model
from typing import Dict, Optional
import logging as logger

class GestureRecognition:
    """
    Class for gesture recognition using a pre-trained machine learning model.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.6):
        """
        Initializes the GestureRecognition class with the specified model.
        
        :param model_path: Path to the pre-trained model.
        :param confidence_threshold: Threshold for confidence in predictions (default: 0.6).
        """
        self.model = self._load_model(model_path)
        self.class_names = self._load_class_names()
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _load_model(model_path: str):
        """
        Loads the gesture recognition model.
        
        :param model_path: Path to the model file.
        :return: Loaded model.
        :raises RuntimeError: If model loading fails.
        """
        try:
            return load_model(model_path)
        except Exception as e:
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
            processed_frame = GestureRecognition._preprocess_frame(frame)
            predictions = self.model.predict(processed_frame)
            class_index = np.argmax(predictions)
            confidence = predictions[0][class_index]

            # Confidence threshold adjustment using softmax temperature scaling
            temperature = 1.0  # Default, can experiment with values
            scaled_confidence = confidence / temperature

            # Dynamic threshold based on confidence
            if scaled_confidence > 0.7:  # Dynamically adjusted threshold
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
            resized_frame = cv2.resize(frame, (64, 64))  # Resize to match training dimensions
            normalized_frame = resized_frame / 255.0  # Normalize pixel values
            return np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
        except Exception as e:
            logger.error(f"Error during frame preprocessing: {e}", exc_info=True)
            raise ValueError("Frame preprocessing failed. Ensure the frame is correctly sized.")
