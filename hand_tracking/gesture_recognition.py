import cv2
import numpy as np
from tensorflow.keras.models import load_model
from typing import Dict, Optional

class GestureRecognition:
    """
    Class for gesture recognition using a pre-trained machine learning model.
    """

    def __init__(self, model_path: str):
        """
        Initializes the GestureRecognition class with the specified model.
        """
        self.model = self._load_model(model_path)
        self.class_names = self._load_class_names()

    @staticmethod
    def _load_model(model_path: str):
        """
        Loads the gesture recognition model.
        """
        try:
            return load_model(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    @staticmethod
    def _load_class_names() -> Dict[int, str]:
        """
        Loads class names for gesture recognition.
        """
        return {
            0: "Stop", 
            1: "Go", 
            2: "Turn Left", 
            3: "Turn Right", 
        }
    
    @staticmethod
    def predict_gesture(self, frame: np.ndarray) -> Optional[str]:
        """
        Predicts the gesture from the provided video frame.
        """
        processed_frame = self._preprocess_frame(frame)
        predictions = self.model.predict(processed_frame)
        class_index = np.argmax(predictions)
        confidence = predictions[0][class_index]

        if confidence > 0.6:  # Confidence threshold for accurate predictions
            return self.class_names.get(class_index)
        return None

    @staticmethod
    def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
        """
        Preprocesses the input frame for the model.
        """
        resized_frame = cv2.resize(frame, (64, 64))  
        normalized_frame = resized_frame / 255.0  
        return np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
