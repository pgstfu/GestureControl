import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# grid spacing
GRID_SPACING = 30
WARP_STRENGTH = 80
FADE = 0.9

# build base grid coordinates once
def make_grid(w, h, spacing):
    y_coords, x_coords = np.mgrid[0:h:spacing, 0:w:spacing]
    return np.dstack((x_coords, y_coords)).reshape(-1, 2)

def draw_grid(frame, grid_points):
    for (x, y) in grid_points.astype(int):
        cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
    return frame

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    ret, frame = cap.read()
    h, w, _ = frame.shape
    base_grid = make_grid(w, h, GRID_SPACING).astype(np.float32)
    warped_grid = base_grid.copy()
    smooth_offset = np.zeros_like(base_grid)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hologram = np.zeros_like(frame, dtype=np.float32)

        # detect hand + index fingertip
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            pts = hand.landmark
            index_tip = pts[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(hologram, (cx, cy), 10, (0, 255, 255), -1)

            # compute warp based on distance to fingertip
            diffs = base_grid - np.array([[cx, cy]])
            dist = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-6
            offset = (diffs / dist) * (WARP_STRENGTH / (dist ** 0.6))
            smooth_offset = FADE * smooth_offset + (1 - FADE) * offset
            warped_grid = base_grid + smooth_offset
        else:
            # slowly relax grid back to original position
            smooth_offset *= 0.95
            warped_grid = base_grid + smooth_offset

        # draw grid
        draw_grid(hologram, warped_grid)
        hologram = cv2.GaussianBlur(hologram, (0, 0), 2)
        blended = cv2.addWeighted(frame, 0.6, hologram.astype(np.uint8), 1.2, 0)

        cv2.putText(blended, "Move your hand - holographic grid reacts", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("GestureControl – Depth Grid", blended)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()