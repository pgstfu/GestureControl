import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

def draw_stick_figure(canvas, center, action="idle", color=(0, 255, 0), size=50, t=0):
    """Draw stick figure with simple animated actions."""
    x, y = center
    angle = 0

    # Action-based motion
    if action == "jump":
        y -= int(30 * abs(math.sin(t / 5)))
    elif action == "wave":
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
    """Return 'open', 'fist', or 'unknown'."""
    finger_tips = [8, 12, 16, 20]
    finger_base = [5, 9, 13, 17]

    extended = 0
    for tip, base in zip(finger_tips, finger_base):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            extended += 1

    if extended >= 3:
        return "open"
    elif extended <= 1:
        return "fist"
    else:
        return "unknown"

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    t = 0
    action = "idle"

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

            # Detect gesture
            gesture = detect_hand_gesture(hand_landmarks)
            if gesture == "open":
                action = "wave"
            elif gesture == "fist":
                action = "jump"
            else:
                action = "idle"

        # 🧩 Fixed position for the figure (bottom-right corner)
        frame_h, frame_w = canvas.shape[:2]
        stick_x = int(frame_w * 0.8)
        stick_y = int(frame_h * 0.8)

        draw_stick_figure(canvas, (stick_x, stick_y), action=action, t=t)

        canvas_resized = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))
        blended = cv2.addWeighted(frame, 0.6, canvas_resized, 1.0, 0)
        cv2.imshow("Gesture Canvas", blended)

        t += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()