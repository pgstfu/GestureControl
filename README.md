# GestureControl
Trying to make interactive images with gesture control

Create and activate a minimal Python virtual environment (keeps project packages isolated).
This makes a private Python environment for GestureCanvas so packages you install won’t affect your system Python.
""python3 -m venv GestureCanvas-env && source GestureCanvas-env/bin/activate""

Install core libraries for gesture detection and image handling.
We’ll start with three:
	•	opencv-python → lets Python access your webcam and images.
	•	mediapipe-silicon → detects hands and poses efficiently on Apple Silicon Macs.
	•	numpy → handles math and image data.
""pip install opencv-python mediapipe numpy""

Test your webcam can be accessed with OpenCV.
Before adding gestures, let’s make sure your Python can see your camera.

explaining the code in test_hands.py
```cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)```
	•	0 → Refers to the built-in webcam.
	•	cv2.CAP_AVFOUNDATION → macOS-specific backend for camera capture.

```with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:```
	•	max_num_hands=2 → Detect up to 2 hands at the same time.
	•	min_detection_confidence=0.7 → Minimum confidence to consider a hand detected.
	•	min_tracking_confidence=0.5 → Minimum confidence to continue tracking a hand from frame to frame.
	•	with ... as hands: → Context manager ensures proper initialization and cleanup of the model.


	1.	Grab frames from webcam.
	2.	Flip and convert to RGB.
	3.	Feed to Mediapipe Hands.
	4.	Draw landmarks on the frame.
	5.	Display in a window.
	6.	Exit when you press q.


test_hands.py
Map hand landmarks to simple gestures
Now that Mediapipe can detect hands, we’ll turn the hand positions into a basic “gesture” that the program can recognize. For starters, let’s detect if the user’s hand is open or closed (fist vs. open palm) using the distance between the tip of the index finger and the wrist.

gesture_canvas.py
Step 8 — Draw generative art based on gestures
Now we’ll connect gestures to drawing on a canvas. For this step, we’ll create a blank canvas and make it draw a circle when your hand is open and erase when it’s a fist.

changed the fist distance for .25 as even a little bit difference in fingers make it erase


