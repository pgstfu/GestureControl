import cv2
import mediapipe as mp
import numpy as np
import math
import time

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

FRAME_W, FRAME_H = 640, 480
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

def distance(pt1, pt2):
    return math.hypot(pt1.x - pt2.x, pt1.y - pt2.y)

def rotate_point(x, y, angle_deg, cx, cy):
    rad = math.radians(angle_deg)
    x -= cx; y -= cy
    xr = x * math.cos(rad) - y * math.sin(rad)
    yr = x * math.sin(rad) + y * math.cos(rad)
    return xr + cx, yr + cy

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False) as face_mesh, \
     mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:

    rotation_angle = 0.0
    zoom = 1.0
    morph_factor = 0.0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_res = face_mesh.process(rgb)
        hand_res = hands.process(rgb)

        hologram = np.zeros_like(frame)

        # --- Hand fingertip control ---
        if hand_res.multi_hand_landmarks:
            hand = hand_res.multi_hand_landmarks[0]
            pts = hand.landmark

            thumb_tip = pts[4]
            index_tip = pts[8]
            middle_tip = pts[12]
            wrist = pts[0]

            # 1. Pinch distance → zoom
            pinch = distance(thumb_tip, index_tip)
            zoom = np.clip(1.2 + (0.15 - pinch) * 5.0, 0.6, 1.6)

            # 2. Index finger horizontal → rotation
            rotation_angle = (index_tip.x - 0.5) * 120  # rotate ±60°

            # 3. Middle finger bend → morph intensity
            morph_factor = np.clip((middle_tip.y - wrist.y) * 2.5, 0.0, 1.0)

            # Visualize fingertips
            for tip in [thumb_tip, index_tip, middle_tip]:
                x, y = int(tip.x * FRAME_W), int(tip.y * FRAME_H)
                cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)

        # --- Face mesh & morph ---
        if face_res.multi_face_landmarks:
            face = face_res.multi_face_landmarks[0]
            pts = np.array([[p.x * FRAME_W, p.y * FRAME_H] for p in face.landmark], dtype=np.float32)

            cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
            ripple = 6 * math.sin((time.time() - t0) * 3.0)

            warped = []
            for x, y in pts:
                dx, dy = x - cx, y - cy
                r = math.hypot(dx, dy)
                if r != 0:
                    nx = x + dx / r * morph_factor * 40
                    ny = y + dy / r * morph_factor * 40
                else:
                    nx, ny = x, y
                nxr, nyr = rotate_point(nx, ny, rotation_angle, cx, cy)
                nxr = cx + (nxr - cx) * zoom
                nyr = cy + (nyr - cy) * zoom
                warped.append((int(nxr), int(nyr)))

            for (x, y) in warped:
                cv2.circle(hologram, (x, y), 1, (0, 255, 255), -1)

        # --- Color + blend ---
        rgb_glow = hologram.astype(np.float32)
        rgb_glow[:, :, 0] *= 1.3
        rgb_glow[:, :, 1] *= 1.8
        rgb_glow[:, :, 2] *= 2.5
        rgb_glow = np.clip(rgb_glow, 0, 255).astype(np.uint8)

        blended = cv2.addWeighted(frame, 0.35, rgb_glow, 1.0, 0)
        cv2.putText(blended, f"Zoom:{zoom:.2f} Rot:{rotation_angle:.1f} Morph:{morph_factor:.2f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Holographic Finger Interaction", blended)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()