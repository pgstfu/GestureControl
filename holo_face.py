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

# Store previous fingertip positions for trail effect
trail_canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.float32)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False) as face_mesh, \
     mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:

    rotation_angle = 0.0
    zoom = 1.0
    morph_factor = 0.0
    t0 = time.time()

    prev_positions = {}
    angle_min, angle_max = 180.0, 0.0
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
        glow_canvas = np.zeros_like(frame, dtype=np.float32)

        # --- Hand fingertip control + trails ---
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
            rotation_angle = (index_tip.x - 0.5) * 120

            # 3. Middle finger bend → morph intensity (self-calibrating angle-based)
            middle_knuckle = pts[9]
            wrist_vec = np.array([wrist.x - middle_knuckle.x, wrist.y - middle_knuckle.y])
            tip_vec   = np.array([middle_tip.x - middle_knuckle.x, middle_tip.y - middle_knuckle.y])

            dot = np.dot(wrist_vec, tip_vec)
            norms = np.linalg.norm(wrist_vec) * np.linalg.norm(tip_vec) + 1e-6
            cos_angle = np.clip(dot / norms, -1.0, 1.0)
            angle = math.degrees(math.acos(cos_angle))

            # --- dynamic calibration ---
            angle_min = min(angle_min, angle)
            angle_max = max(angle_max, angle)
            angle_range = angle_max - angle_min + 1e-6

            morph_factor = np.clip((angle - angle_min) / angle_range, 0.0, 1.0)

            # Draw fingertip trails
            tips = {'thumb': thumb_tip, 'index': index_tip, 'middle': middle_tip}
            for name, tip in tips.items():
                x, y = int(tip.x * FRAME_W), int(tip.y * FRAME_H)
                if name in prev_positions:
                    px, py = prev_positions[name]
                    speed = math.hypot(x - px, y - py)
                    color = (100 + int(155 * min(speed, 1)), 255, 255 - int(150 * min(speed, 1)))
                    cv2.line(trail_canvas, (px, py), (x, y), color, 3)
                prev_positions[name] = (x, y)
                cv2.circle(glow_canvas, (x, y), 8, (0, 255, 255), -1)

        # Slowly fade old trails
        trail_canvas *= 0.92  # fade factor

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

        # --- Combine hologram, glow, trails ---
        total = hologram.astype(np.float32) + trail_canvas + glow_canvas
        total = np.clip(total, 0, 255).astype(np.uint8)

        blended = cv2.addWeighted(frame, 0.35, total, 1.0, 0)
        cv2.putText(blended, f"Zoom:{zoom:.2f} Rot:{rotation_angle:.1f} Morph:{morph_factor:.2f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Holographic Finger Trails", blended)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()