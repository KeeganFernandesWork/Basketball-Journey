# main.py

import cv2
# Import both functions from our module
from pose_detector import find_pose, label_keypoints

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 1. Find the pose in the current frame
    results = find_pose(frame)

    # The results object is a generator, so we loop through it
    for r in results:
        # 2. Get the frame with the basic skeleton drawn on it
        annotated_frame = r.plot()

        # 3. Pass this frame and the result object to our new function to add labels
        final_frame = label_keypoints(annotated_frame, r)
        
        # 4. Display the final, fully labeled frame
        cv2.imshow("YOLOv8 Pose Estimation", final_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()