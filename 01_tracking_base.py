import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def draw_safe(landmarks, connections, frame, l_spec=None, c_spec=None):
    if landmarks:
        try:
            mp_drawing.draw_landmarks(
                frame, landmarks, connections,
                landmark_drawing_spec=l_spec,
                connection_drawing_spec=c_spec)
        except Exception:
            pass

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for a mirror view
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # --- Draw components ---
        draw_safe(results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, frame,
                  mp_styles.get_default_face_mesh_tesselation_style(),
                  mp_styles.get_default_face_mesh_contours_style())

        draw_safe(results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, frame,
                  mp_styles.get_default_pose_landmarks_style(),
                  mp_styles.get_default_pose_landmarks_style())

        draw_safe(results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, frame,
                  mp_styles.get_default_hand_landmarks_style(),
                  mp_styles.get_default_hand_landmarks_style())

        draw_safe(results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, frame,
                  mp_styles.get_default_hand_landmarks_style(),
                  mp_styles.get_default_hand_landmarks_style())

        # --- Label detections clearly ---
        labels = []
        if results.face_landmarks:
            labels.append("FACE")
        if results.left_hand_landmarks:
            labels.append("YOUR RIGHT HAND")   # mirrored view
        if results.right_hand_landmarks:
            labels.append("YOUR LEFT HAND")
        if results.pose_landmarks:
            labels.append("POSE")
        if not labels:
            labels.append("NO LANDMARKS")

        text = " | ".join(labels)
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("GestureControl Base Tracker (Corrected)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()