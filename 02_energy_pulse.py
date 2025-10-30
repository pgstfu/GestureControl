import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ---- MediaPipe setup ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# ---- Pulse parameters ----
pulses = []           # list of [x, y, radius, intensity]
FADE_SPEED = 3        # how fast intensity fades
EXPAND_SPEED = 8      # how fast circle expands
TRIGGER_DIST = 0.035  # thumb–index distance to trigger pulse (smaller = more sensitive)
COOLDOWN = 0.6        # seconds between pulses
last_pulse_time = 0

# ---- Color & visual tuning ----
def draw_pulse(canvas, x, y, r, intensity):
    color = (int(255 * intensity), int(200 * intensity), 255)
    cv2.circle(canvas, (int(x), int(y)), int(r), color, 2, cv2.LINE_AA)

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        hologram = np.zeros_like(frame, dtype=np.float32)

        # --- Detect pinch gesture (thumb–index) to trigger pulse ---
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            pts = hand.landmark
            thumb_tip = pts[4]
            index_tip = pts[8]

            # draw hand skeleton
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
            now = time.time()
            if dist < TRIGGER_DIST and (now - last_pulse_time) > COOLDOWN:
                last_pulse_time = now
                x = int(index_tip.x * w)
                y = int(index_tip.y * h)
                pulses.append([x, y, 0, 1.0])  # start new pulse

        # --- Animate pulses ---
        for p in pulses[:]:
            x, y, r, intensity = p
            draw_pulse(hologram, x, y, r, intensity)
            p[2] += EXPAND_SPEED          # radius grows
            p[3] -= 0.03 * FADE_SPEED     # intensity fades
            if p[3] <= 0:
                pulses.remove(p)

        # --- Blend hologram with live video ---
        holo_color = hologram.astype(np.float32)
        holo_color[:, :, 0] *= 1.3
        holo_color[:, :, 1] *= 1.8
        holo_color[:, :, 2] *= 2.5
        holo_color = np.clip(holo_color, 0, 255).astype(np.uint8)

        blended = cv2.addWeighted(frame, 0.5, holo_color, 1.0, 0)
        cv2.putText(blended, "Pinch thumb & index to create pulse", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("GestureControl – Energy Pulse", blended)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()