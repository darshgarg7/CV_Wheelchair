import cv2
import threading
import logging
import numpy as np
import tensorflow as tf
from queue import Queue
import RPi.GPIO as GPIO
import time

class GestureRecognition:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.gesture_labels = ["Stop", "Go", "Turn Left", "Turn Right"]

    def predict_gesture(self, frame):
        """
        Predicts the gesture from the input frame.
        """
        processed_frame = cv2.resize(frame, (224, 224))  # Resize to match model input size
        processed_frame = processed_frame / 255.0  # Normalize pixel values
        processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension
        prediction = self.model.predict(processed_frame)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return self.gesture_labels[predicted_class]

class WheelchairController:
    def __init__(self, camera_index: int = 0, model_path: str = "models/gesture_recognition_model.h5"):
        self.gesture_recognizer = GestureRecognition(model_path)
        self.cap = cv2.VideoCapture(camera_index)
        self.frame_queue = Queue(maxsize=10)
        self._initialize_gpio()
        self.gesture_recognition_thread = GestureRecognitionThread(self.frame_queue, self)
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.recognition_thread = threading.Thread(target=self.gesture_recognition_thread.run, daemon=True)

    def _initialize_gpio(self):
        try:
            GPIO.setmode(GPIO.BCM)
            self.forward_pin = 17
            self.left_pin = 27
            self.right_pin = 22
            self.stop_pin = 5
            GPIO.setup(self.forward_pin, GPIO.OUT)
            GPIO.setup(self.left_pin, GPIO.OUT)
            GPIO.setup(self.right_pin, GPIO.OUT)
            GPIO.setup(self.stop_pin, GPIO.OUT)
        except Exception as e:
            logging.error(f"Error initializing GPIO: {e}")
            raise

    def _capture_frames(self):
        while True:
            success, frame = self.cap.read()
            if success and not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                logging.warning("Failed to capture or buffer full.")

    def _handle_gesture(self, gesture: str):
        try:
            logging.info(f"Handling gesture: {gesture}")
            if gesture == "Stop":
                self._stop_wheelchair()
            elif gesture == "Go":
                self._move_wheelchair_forward()
            elif gesture == "Turn Left":
                self._turn_wheelchair_left()
            elif gesture == "Turn Right":
                self._turn_wheelchair_right()
            else:
                logging.warning(f"Unknown gesture: {gesture}")
        except Exception as e:
            logging.error(f"Error handling gesture: {e}")

    def _move_wheelchair_forward(self):
        GPIO.output(self.forward_pin, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(self.forward_pin, GPIO.LOW)

    def _turn_wheelchair_left(self):
        GPIO.output(self.left_pin, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(self.left_pin, GPIO.LOW)

    def _turn_wheelchair_right(self):
        GPIO.output(self.right_pin, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(self.right_pin, GPIO.LOW)

    def _stop_wheelchair(self):
        GPIO.output(self.stop_pin, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(self.stop_pin, GPIO.LOW)

    def start(self):
        try:
            self.capture_thread.start()
            self.recognition_thread.start()
            while True:
                frame = self.frame_queue.get() if not self.frame_queue.empty() else None
                if frame:
                    cv2.imshow("Wheelchair Control", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            logging.error(f"Error during main loop: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()

class GestureRecognitionThread(threading.Thread):
    def __init__(self, frame_queue, wheelchair_controller):
        super().__init__()
        self.frame_queue = frame_queue
        self.wheelchair_controller = wheelchair_controller

    def run(self):
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                gesture = self.wheelchair_controller.gesture_recognizer.predict_gesture(frame)
                if gesture:
                    logging.info(f"Recognized gesture: {gesture}")
                    self.wheelchair_controller._handle_gesture(gesture)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    controller = WheelchairController(camera_index=0, model_path="models/gesture_recognition_model.h5")
    controller.start()
