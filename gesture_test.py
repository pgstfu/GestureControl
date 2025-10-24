import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        gesture_text = "No hand detected"

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of wrist and index finger tip
            wrist = hand.landmark[0]
            index_tip = hand.landmark[8]

            # Compute Euclidean distance
            dist = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y)

            # Simple threshold to decide open or closed hand
            if dist > 0.2:
                gesture_text = "Open hand"
            else:
                gesture_text = "Fist"

        cv2.putText(frame, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Gesture Detection", frame)

        canvas_resized = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))
        blended = cv2.addWeighted(frame, 0.6, canvas_resized, 1.0, 0)
        cv2.imshow("Gesture Canvas", blended)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()