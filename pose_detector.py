# pose_detector.py

import cv2
from ultralytics import YOLO

# This list maps the keypoint indices to their names.
# YOLOv8-pose was trained on the COCO dataset, which has 17 keypoints.
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Load the YOLOv8n-pose model once for efficiency.
model = YOLO('yolov8n-pose.pt')

def find_pose(frame):
    """
    Takes an image frame and returns the processed results from the YOLOv8 model.
    """
    results = model(frame, stream=True)
    return results

def label_keypoints(frame, result_object):
    """
    Takes a frame and a single result object, and draws text labels for each keypoint.

    Args:
        frame: The image frame (preferably with the skeleton already drawn).
        result_object: A single result instance from the YOLO model's output.

    Returns:
        The frame with keypoint labels drawn on it.
    """
    # Check if any keypoints were detected
    if result_object.keypoints and len(result_object.keypoints.data) > 0:
        # Get the coordinates and confidence scores of the keypoints
        keypoints_data = result_object.keypoints.data[0]

        for i, keypoint in enumerate(keypoints_data):
            # The keypoint tensor contains (x, y, confidence)
            x, y, confidence = keypoint

            # Only draw the label if the confidence is above a threshold
            if confidence > 0.75:
                # Get the name of the keypoint
                label = KEYPOINT_NAMES[i]
                
                # Convert coordinates to integers for drawing
                ix, iy = int(x), int(y)
                
                # Draw a small circle at the keypoint location
                cv2.circle(frame, (ix, iy), 3, (0, 255, 0), -1) # Green circle
                
                # Put the text label slightly above and to the right of the keypoint
                cv2.putText(frame, label, (ix + 5, iy - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA) # Blue text
    
    return frame