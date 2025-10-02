# pose_detector.py

import cv2
import time
import math
from ultralytics import YOLO

# This list maps the keypoint indices to their names.
# YOLOv8-pose was trained on the COCO dataset, which has 17 keypoints.
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]
points = [8, 6,12]
# A dictionary to store the last known position, timestamp, and smoothed velocity for each tracked object
# Key: track_id, Value: { "pos": (x, y), "time": timestamp, "smoothed_v": float }
velocity_tracker = {}

# Smoothing factor for the Exponential Moving Average (EMA). A smaller value means more smoothing.
SMOOTHING_ALPHA = 0.1

# Load the YOLOv8n-pose model once for efficiency.
model = YOLO('yolov8n-pose.pt')

def find_pose(frame, track=False):
    """
    Takes an image frame and returns the processed results from the YOLOv8 model.
    
    Args:
        frame: The image frame.
        track: Boolean to indicate if object tracking should be enabled.
    """
    if track:
        # Use model.track() for persistent tracking across frames
        results = model.track(frame, persist=True, verbose=False)
    else:
        results = model(frame, stream=True, verbose=False)
    return results

def calculate_velocity(track_id, current_pos):
    """Calculates a smoothed velocity for a tracked object using EMA."""
    current_time = time.time()
    raw_velocity = 0.0
    smoothed_velocity = 0.0

    if track_id in velocity_tracker:
        last_info = velocity_tracker[track_id]
        last_pos = last_info["pos"]
        last_time = last_info["time"]
        last_smoothed_v = last_info.get("smoothed_v", 0.0) # Get last smoothed velocity, default to 0

        # Calculate distance and time difference
        distance = ((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)**0.5
        time_diff = current_time - last_time
        
        # Calculate raw velocity for this frame
        if time_diff > 0:
            raw_velocity = distance / time_diff

        # Apply Exponential Moving Average (EMA) for smoothing
        smoothed_velocity = (SMOOTHING_ALPHA * raw_velocity) + (1 - SMOOTHING_ALPHA) * last_smoothed_v
    
    # Update the tracker with the new position, time, and smoothed velocity
    velocity_tracker[track_id] = {"pos": current_pos, "time": current_time, "smoothed_v": smoothed_velocity}
    
    return smoothed_velocity
def calculate_angle(points):
    """
    Calculates the angle between three points, with the second point as the vertex.

    Args:
        points (list): A list of three tuples, where each tuple represents a point (x, y).

    Returns:
        float: The angle in degrees.
    """
    p1, p2, p3 = points[0], points[1], points[2]

    # Vectors from the vertex (p2)
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    # Dot product
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

    # Magnitudes
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

    # Cosine of the angle
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0 # Or handle as an error, as points are coincident
    
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)

    # Ensure the value is within the valid domain for acos
    cosine_angle = max(min(cosine_angle, 1.0), -1.0)

    # Angle in radians
    angle_rad = math.acos(cosine_angle)

    # Convert to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg
def label_keypoints(frame, result_object):
    """
    Takes a frame and a single result object, and draws text labels for each keypoint
    and calculates velocity if tracking is enabled.
    """
    # Check if any keypoints were detected
    if result_object.keypoints and len(result_object.keypoints.data) > 0:
        keypoints_data = result_object.keypoints.data[0]

        # --- Velocity Calculation (if tracking is on) ---
        # We need a track ID, which comes from the bounding box results
        track_id = None
        if result_object.boxes and result_object.boxes.id is not None:
            track_id = int(result_object.boxes.id[0])
            
            # Use the "Nose" keypoint (index 0) for tracking velocity
            nose_keypoint = keypoints_data[0]
            nx, ny, n_conf = nose_keypoint
            if n_conf > 0.5:
                velocity = calculate_velocity(track_id, (nx, ny))
                # Display the velocity on the frame
                cv2.putText(frame, f"ID {track_id} V: {velocity:.2f} px/s", (int(nx), int(ny) - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- Angle Calculation ---
        # Define the keypoint indices for the angle: 8 (R_Elbow), 6 (R_Shoulder), 12 (R_Hip)
        p1_idx, p2_idx, p3_idx = 8, 6, 12

        # Check if all three keypoints are detected with sufficient confidence
        if len(keypoints_data) > max(p1_idx, p2_idx, p3_idx):
            p1_data = keypoints_data[p1_idx]  # Right Elbow
            p2_data = keypoints_data[p2_idx]  # Right Shoulder (Vertex)
            p3_data = keypoints_data[p3_idx]  # Right Hip

            # Check confidence of all 3 points
            if p1_data[2] > 0.5 and p2_data[2] > 0.5 and p3_data[2] > 0.5:
                # Get coordinates
                p1 = (p1_data[0], p1_data[1])
                p2 = (p2_data[0], p2_data[1])
                p3 = (p3_data[0], p3_data[1])

                # Calculate angle
                angle = calculate_angle([p1, p2, p3])

                # Position for the text (near the vertex)
                text_pos = (int(p2[0]) + 10, int(p2[1]))

                # Display the angle on the frame
                cv2.putText(frame, f"Angle: {angle:.1f}", text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # --- Keypoint Labeling ---
        for i, keypoint in enumerate(keypoints_data):
            x, y, confidence = keypoint
            if confidence > 0.5:
                label = KEYPOINT_NAMES[i]
                ix, iy = int(x), int(y)
                cv2.circle(frame, (ix, iy), 3, (0, 255, 0), -1)
                cv2.putText(frame, label, (ix + 5, iy - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    return frame