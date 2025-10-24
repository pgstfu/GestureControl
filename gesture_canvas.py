import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Canvas fixed size (same as camera output)
FRAME_W, FRAME_H = 640, 480
canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

def draw_stick_figure(canvas, center, action="idle", color=(0, 255, 0), size=50, t=0):
    """Draws a simple animated stick figure."""
    x, y = center
    angle = 0
    if action in ["wave", "walk"]:
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
    """Detects open, fist, or unknown gesture."""
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

# ---------------- MAIN LOOP ----------------
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    t = 0
    action = "idle"
    pos_x = int(FRAME_W * 0.5)
    pos_y = int(FRAME_H * 0.7)
    prev_x = None

    # Physics
    vel_y = 0
    gravity = 2
    on_ground = True
    figure_height = 100
    half_height = figure_height // 2
    floor_y = FRAME_H - 80  # visual floor position (constant)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        canvas[:] = 0

        # ---------------- HAND GESTURES ----------------
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

            # Horizontal motion
            index_tip = hand_landmarks.landmark[8]
            curr_x = int(index_tip.x * FRAME_W)
            if prev_x is not None:
                dx = curr_x - prev_x
                if abs(dx) > 10:
                    if gesture == "open":
                        pos_x += int(np.sign(dx) * 15)
                        action = "walk"
                    elif gesture == "fist" and on_ground:
                        pos_x += int(np.sign(dx) * 20)
                        action = "jump"
                pos_x = np.clip(pos_x, 50, FRAME_W - 50)
            prev_x = curr_x

        # ---------------- PHYSICS ----------------
        if action == "jump" and on_ground:
            vel_y = -25
            on_ground = False

        if not on_ground:
            vel_y += gravity
            pos_y += vel_y

        # Stop when feet reach floor
        if pos_y + half_height >= floor_y:
            pos_y = floor_y - half_height
            vel_y = 0
            on_ground = True

        # ---------------- DRAW EVERYTHING ----------------
        # Floor
        cv2.line(canvas, (0, floor_y), (FRAME_W, floor_y), (0, 255, 0), 3)

        # Shadow
        cv2.circle(canvas, (pos_x, floor_y + 10), 15, (50, 50, 50), -1)

        # Stick figure
        draw_stick_figure(canvas, (pos_x, pos_y), action=action, t=t)

        # Combine camera + canvas
        blended = cv2.addWeighted(frame, 0.6, canvas, 1.0, 0)
        cv2.imshow("Gesture Canvas", blended)

        # Exit
        t += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()