import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(self, max_num_hands=1, detection_conf=0.7, tracking_conf=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_draw = mp.solutions.drawing_utils

    def get_palm_center(self, landmarks, frame_width, frame_height):
        palm_points = [0, 5, 17]
        xs = [landmarks[p].x * frame_width for p in palm_points]
        ys = [landmarks[p].y * frame_height for p in palm_points]
        return int(np.mean(xs)), int(np.mean(ys))

    def count_fingers_up(self, landmarks):
        """
        Count number of extended fingers.
        Tips above PIPs = finger up.
        """
        tips = [8, 12, 16, 20]   # index, middle, ring, pinky tips
        pips = [6, 10, 14, 18]   # their PIP joints

        fingers_up = 0

        # Count 4 fingers
        for tip, pip in zip(tips, pips):
            if landmarks[tip].y < landmarks[pip].y:
                fingers_up += 1

        # Check thumb separately (optional, ignore for stability)
        return fingers_up

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def draw_hand(self, frame, hand_landmarks):
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )


def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        h, w, _ = frame.shape
        results = tracker.process_frame(frame)

        command = "OBJECT DETECTION MODE"

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                tracker.draw_hand(frame, hand)

                cx, cy = tracker.get_palm_center(hand.landmark, w, h)
                fingers = tracker.count_fingers_up(hand.landmark)

                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

                # --- MOVEMENT LOGIC --- #
                if fingers == 2:
                    command = "FORWARD"

                elif fingers == 3:
                    command = "BACKWARD"

                elif fingers >= 4:
                    # Only steering allowed when palm is fully open
                    x_norm = cx / w
                    if x_norm < 0.4:
                        command = "LEFT"
                    elif x_norm > 0.6:
                        command = "RIGHT"
                    else:
                        command = "OBJECT DETECTION MODE"

                else:
                    command = "OBJECT DETECTION MODE"

                print(f"Fingers={fingers} Palm=({cx},{cy}) → {command}")

        else:
            command = "OBJECT DETECTION MODE"

        # Overlay command on screen
        cv2.putText(frame, f"Command: {command}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        cv2.imshow("AungBot Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# run code
# python vision/hand_tracking.py
