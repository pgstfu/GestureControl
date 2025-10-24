import cv2
import mediapipe as mp
import math
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)  # blank black canvas

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            hand = results.multi_hand_landmarks[0]
            main_hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            wrist = hand.landmark[0]
            index_tip = hand.landmark[8]

            dist = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y)

            # Map normalized coordinates to canvas size
            x = int(index_tip.x * canvas.shape[1])
            y = int(index_tip.y * canvas.shape[0])

            if dist > 0.25:
                # Open hand → draw a circle
                cv2.circle(canvas, (x, y), 20, (0, 0, 255), -1)
            else:
                # Fist → erase (draw black circle)
                cv2.circle(canvas, (x, y), 20, (0, 0, 0), -1)

            # Resize frame to match canvas
            frame_resized = cv2.resize(frame, (canvas.shape[1], canvas.shape[0]))
            # Detect number of hands
            hands_count = len(results.multi_hand_landmarks)

            # If both hands are visible, clear the canvas
            if hands_count == 2:
                canvas = np.zeros_like(canvas)
            # Show both webcam and canvas side by side
            combined = cv2.hconcat([frame_resized, canvas])
            cv2.imshow("Gesture Canvas", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()