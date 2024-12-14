import cv2
import threading
import logging
from gesture_recognition import GestureRecognition
from database import DatabaseManager
import RPi.GPIO as GPIO
from queue import Queue
import time


class WheelchairController:
    """
    Controls wheelchair movement based on hand gesture recognition.
    """

    def __init__(self, camera_index: int = 0, model_path: str = "models/gesture_recognition_model.h5"):
        """
        Initializes the WheelchairController with required components.
        """
        self.gesture_recognizer = GestureRecognition(model_path)
        self.db_manager = DatabaseManager()
        self.cap = cv2.VideoCapture(camera_index)
        self.frame_queue = Queue(maxsize=10)
        self._initialize_gpio()

        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.recognition_thread = threading.Thread(target=self._recognize_gestures, daemon=True)

    def _initialize_gpio(self):
        """
        Initializes the GPIO pins for motor control and handles GPIO errors.
        """
        try:
            GPIO.setmode(GPIO.BCM)
            self.forward_pin = 17  # Example pin for moving forward
            self.left_pin = 27     # Example pin for turning left
            self.right_pin = 22    # Example pin for turning right
            self.stop_pin = 5      # Example pin for stopping

            GPIO.setup(self.forward_pin, GPIO.OUT)
            GPIO.setup(self.left_pin, GPIO.OUT)
            GPIO.setup(self.right_pin, GPIO.OUT)
            GPIO.setup(self.stop_pin, GPIO.OUT)

        except Exception as e:
            logging.error(f"Error initializing GPIO: {e}")
            raise

    def _capture_frames(self) -> None:
        """
        Captures frames from the camera and places them into a queue.
        """
        while True:
            success, frame = self.cap.read()
            if success and not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                logging.warning("Failed to capture or buffer full.")

    def _recognize_gestures(self) -> None:
        """
        Recognizes gestures by retrieving frames from the queue.
        """
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                gesture = self.gesture_recognizer.predict_gesture(frame)
                if gesture:
                    logging.info(f"Recognized gesture: {gesture}")
                    self._handle_gesture(gesture)

    def _handle_gesture(self, gesture: str) -> None:
        """
        Handles the recognized gesture and triggers the appropriate wheelchair movement or action.
        """
        try:
            logging.info(f"Handling gesture: {gesture}")
            self.db_manager.save_gesture(gesture)

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
        """ Move wheelchair forward. """
        GPIO.output(self.forward_pin, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(self.forward_pin, GPIO.LOW)

    def _turn_wheelchair_left(self):
        """ Turn wheelchair left. """
        GPIO.output(self.left_pin, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(self.left_pin, GPIO.LOW)

    def _turn_wheelchair_right(self):
        """ Turn wheelchair right. """
        GPIO.output(self.right_pin, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(self.right_pin, GPIO.LOW)

    def _stop_wheelchair(self):
        """ Stop wheelchair. """
        GPIO.output(self.stop_pin, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(self.stop_pin, GPIO.LOW)

    def start(self) -> None:
        """ Starts the controller's main processes. """
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

    def _cleanup(self) -> None:
        """ Releases resources and stops all processes. """
        self.cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    controller = WheelchairController(camera_index=0, model_path="models/gesture_recognition_model.h5")
    controller.start()
