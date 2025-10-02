import cv2
import numpy as np

# --- 1. Initialize the Kalman Filter ---
# State is [x, y, dx, dy] - position and velocity
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

# Open video and initialize tracker
video_path = 'basketball.mp4'
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

# Find the ball in the first frame to initialize the state
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_orange = np.array([5, 100, 100])
upper_orange = np.array([15, 255, 255])
mask = cv2.inRange(hsv, lower_orange, upper_orange)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

initial_measurement = None
if len(contours) > 0:
    c = max(contours, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    center = np.array([x + w/2, y + h/2], np.float32)
    kalman.statePost = np.array([center[0], center[1], 0, 0], np.float32)
    kalman.statePre = np.array([center[0], center[1], 0, 0], np.float32)

# --- 2. Main Tracking Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Predict Step ---
    prediction = kalman.predict()
    pred_pt = (int(prediction[0]), int(prediction[1]))

    # --- Detect (Measurement) Step ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detection_successful = False
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        # Filter out small contours
        if cv2.contourArea(c) > 500:
            (x, y, w, h) = cv2.boundingRect(c)
            center = np.array([x + w/2, y + h/2], np.float32)
            
            # --- Correct Step ---
            kalman.correct(center)
            detection_successful = True
            
    # Get the corrected state (or the predicted state if no detection)
    corrected_state = kalman.statePost
    corr_pt = (int(corrected_state[0]), int(corrected_state[1]))

    # --- Draw the result ---
    # Draw prediction in red
    cv2.circle(frame, pred_pt, 20, (0, 0, 255), 2)
    # Draw corrected position in green
    cv2.circle(frame, corr_pt, 20, (0, 255, 0), 2)
    
    # If detection failed, show the last known good position from the filter
    if not detection_successful:
        cv2.putText(frame, "Coasting", (corr_pt[0] + 25, corr_pt[1] - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


    cv2.imshow("Kalman Filter Tracking", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()