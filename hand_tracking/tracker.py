import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        """
        Initializes the hand tracker using MediaPipe.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def get_hand_gesture(self, frame):
        """
        Processes the frame to detect hand gestures.

        :param frame: Input frame from the camera
        :return: Gesture string ("Stop", "Go", "Turn Left", "Turn Right") or None if no gesture detected
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Using hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Basic gesture recognition based on finger positions
            extended_fingers = sum([1 for i in range(5) if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i + 5].y])

            # Implement the gesture recognition based on your gesture definitions
            if extended_fingers == 1:  # One finger extended
                return "Go"
            elif extended_fingers == 0:  # Fist (no fingers extended)
                return "Stop"
            elif hand_landmarks.landmark[4].x < hand_landmarks.landmark[8].x:  # Thumbs Left
                return "Turn Left"
            elif hand_landmarks.landmark[4].x > hand_landmarks.landmark[8].x:  # Thumbs Right
                return "Turn Right"
            
        return None
