import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FRAME_W, FRAME_H = 640, 480
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

def draw_stick_figure(canvas, center, action="idle", color=(0, 255, 0), size=50, t=0):
    x, y = center
    angle = 0
    if action in ["wave", "walk"]:
        angle = int(20 * math.sin(t / 4))

    cv2.circle(canvas, (x, y - size // 2), size // 3, color, 2)
    cv2.line(canvas, (x, y - size // 6), (x, y + size // 2), color, 2)
    cv2.line(canvas, (x - size // 2, y), (x, y - angle), color, 2)
    cv2.line(canvas, (x, y - angle), (x + size // 2, y - angle), color, 2)
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

# ------------- MAIN LOOP -------------
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    t = 0
    action = "idle"

    # Player world vars
    world_offset = 0
    world_speed = 15

    pos_x = FRAME_W // 2
    pos_y = FRAME_H * 3 // 4
    figure_height = 100
    half_height = figure_height // 2

    vel_y = 0
    gravity = 2
    on_ground = True
    prev_x = None
    floor_y = FRAME_H - 80

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        canvas[:] = 0

        # Draw sky gradient (background)
        for i in range(FRAME_H):
            color = (255 - i // 2, 180 - i // 3, 100 + i // 6)
            canvas[i, :] = color

        # ----------- HAND CONTROL -----------
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

            index_tip = hand_landmarks.landmark[8]
            curr_x = int(index_tip.x * FRAME_W)
            if prev_x is not None:
                dx = curr_x - prev_x
                if abs(dx) > 10:
                    if gesture == "open":
                        world_offset -= int(np.sign(dx) * world_speed)
                        action = "walk"
                    elif gesture == "fist" and on_ground:
                        world_offset -= int(np.sign(dx) * world_speed * 1.2)
                        action = "jump"
            prev_x = curr_x

        # ----------- PHYSICS -----------
        if action == "jump" and on_ground:
            vel_y = -25
            on_ground = False

        if not on_ground:
            vel_y += gravity
            pos_y += vel_y

        if pos_y + half_height >= floor_y:
            pos_y = floor_y - half_height
            vel_y = 0
            on_ground = True

        # ----------- DRAW WORLD -----------
        tile_width = 60
        for i in range(-1, FRAME_W // tile_width + 3):
            tile_x = (i * tile_width + world_offset) % (FRAME_W + tile_width) - tile_width
            cv2.rectangle(canvas, (tile_x, floor_y), (tile_x + tile_width - 5, FRAME_H), (30, 120, 30), -1)

        # Shadow
        cv2.circle(canvas, (pos_x, floor_y + 10), 15, (50, 50, 50), -1)

        # Stick figure
        draw_stick_figure(canvas, (pos_x, int(pos_y)), action=action, t=t)

        # Blend webcam + world
        blended = cv2.addWeighted(frame, 0.5, canvas, 1.0, 0)
        cv2.imshow("Gesture Canvas", blended)

        t += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()