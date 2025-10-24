import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

def draw_stick_figure(canvas, center, action="idle", color=(0, 255, 0), size=50, t=0):
    x, y = center
    angle = 0
    if action == "jump":
        y -= int(30 * abs(math.sin(t / 5)))
    elif action == "wave" or action == "walk":
        angle = int(20 * math.sin(t / 4))

    # Head
    cv2.circle(canvas, (x, y - size // 2), size // 3, color, 2)
    # Body
    cv2.line(canvas, (x, y - size // 6), (x, y + size // 2), color, 2)
    # Arms
    cv2.line(canvas, (x - size // 2, y), (x, y - angle), color, 2)
    cv2.line(canvas, (x, y - angle), (x + size // 2, y - angle), color, 2)
    # Legs
    cv2.line(canvas, (x, y + size // 2), (x - size // 3, y + size), color, 2)
    cv2.line(canvas, (x, y + size // 2), (x + size // 3, y + size), color, 2)

def detect_hand_gesture(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_base = [5, 9, 13, 17]
    extended = sum(
        hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y
        for tip, base in zip(finger_tips, finger_base)
    )
    if extended >= 3:
        return "open"
    elif extended <= 1:
        return "fist"
    else:
        return "unknown"

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    t = 0
    action = "idle"
    frame_w, frame_h = 640, 480
    pos_x = int(frame_w * 0.8)
    pos_y = int(frame_h * 0.8)
    prev_x = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        canvas[:] = 0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = detect_hand_gesture(hand_landmarks)
            if gesture == "open":
                action = "wave"
            elif gesture == "fist":
                action = "jump"
            else:
                action = "idle"

            # --- track horizontal motion ---
            index_tip = hand_landmarks.landmark[8]
            curr_x = int(index_tip.x * frame.shape[1])

            if prev_x is not None:
                dx = curr_x - prev_x
                if abs(dx) > 10:                # small threshold to filter noise
                    if gesture == "open":
                        pos_x += int(np.sign(dx) * 15)  # slower move when waving
                        action = "walk"
                    if gesture == "fist":
                        pos_x += int(np.sign(dx) * 20)  # faster move when jumping
                        action = "jump"
                    pos_x = np.clip(pos_x, 50, frame.shape[1] - 50)
                    action = "walk"
            prev_x = curr_x

        draw_stick_figure(canvas, (pos_x, pos_y), action=action, t=t)

        canvas_resized = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))
        blended = cv2.addWeighted(frame, 0.6, canvas_resized, 1.0, 0)
        cv2.imshow("Gesture Canvas", blended)

        t += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()