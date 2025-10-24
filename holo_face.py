import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Drawing style configs for holographic effect
draw_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255))
conn_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0, color=(255, 255, 255))

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Create black canvas for hologram
        hologram = np.zeros_like(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh using official MediaPipe connections
                mp_drawing.draw_landmarks(
                    image=hologram,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=draw_spec,
                    connection_drawing_spec=conn_spec,
                )

        # Add holographic shimmer (light waves)
        # Create RGB glow manually
        rgb_glow = hologram.copy().astype(np.float32)

        # Slightly amplify colors for neon feel
        rgb_glow[:, :, 0] *= 1.5   # Blue
        rgb_glow[:, :, 1] *= 2.0   # Green
        rgb_glow[:, :, 2] *= 3.0   # Red

        rgb_glow = np.clip(rgb_glow, 0, 255).astype(np.uint8)

        # Blend with live video
        blended = cv2.addWeighted(frame, 0.4, rgb_glow, 0.9, 0)

        cv2.imshow("Holographic Face Mesh", blended)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()