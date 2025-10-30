import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Particle class
class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.life = random.uniform(0.8, 1.2)
        self.age = 0.0
        self.size = random.uniform(2, 4)

    def update(self, dt):
        self.x += self.vx * 60 * dt
        self.y += self.vy * 60 * dt
        self.age += dt
        self.vx *= 0.98
        self.vy *= 0.98

    def is_alive(self):
        return self.age < self.life

    def draw(self, frame):
        fade = max(0, 1 - (self.age / self.life))
        color = (int(255 * fade), int(200 * fade), 255)
        cv2.circle(frame, (int(self.x), int(self.y)), int(self.size), color, -1, cv2.LINE_AA)

particles = []
prev_points = None
last_emit_time = 0

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        now = time.time()
        dt = now - prev_time
        prev_time = now

        hologram = np.zeros_like(frame, dtype=np.float32)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            pts = hand.landmark
            index_tip = pts[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            # emit particles
            for _ in range(8):
                particles.append(Particle(x, y))

            prev_points = (x, y)
        else:
            prev_points = None

        # update & draw particles
        for p in particles[:]:
            p.update(dt)
            p.draw(hologram)
            if not p.is_alive():
                particles.remove(p)

        # color tint for glow
        glow = cv2.GaussianBlur(hologram, (0, 0), 6)
        glow = np.clip(glow * 1.6, 0, 255).astype(np.uint8)

        # blend with original frame
        blended = cv2.addWeighted(frame, 0.5, glow, 1.0, 0)
        cv2.putText(blended, "Move your hand - energy particles follow fingertips", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("GestureControl – Particle Field", blended)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()