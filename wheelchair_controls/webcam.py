import cv2
from hand_tracking.gesture_recognition import GestureRecognizer

def capture_webcam_frame():
    """
    Captures a frame from the webcam for hand gesture detection.
    
    :return: Captured frame from the webcam
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera feed.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Failed to capture image from webcam.")
        return None

    return frame

def run_webcam():
    recognizer = GestureRecognizer()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        prediction = recognizer.predict_gesture(frame)
        cv2.putText(frame, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()

