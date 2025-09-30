import cv2
from ultralytics import YOLO
import sys
import os
from pathlib import Path

# --- Configuration ---
# You can change these values to fit your needs.
VIDEO_PATH = 'basketball.mp4'  # Path to your basketball video file.
MODEL_NAME = 'yolov8n.pt'      # The YOLOv8 model to use (e.g., yolov8n.pt, yolov8m.pt). 'n' is fast but less accurate, 'm' is a good balance.
CONFIDENCE_THRESHOLD = 0.4     # Only detect objects with confidence greater than this value (0.0 to 1.0).
BASKETBALL_CLASS_ID = 32       # This is the class ID for 'sports ball' in the COCO dataset, which YOLO is trained on.
OUTPUT_DIR = 'output'          # Directory to save the output videos.

# --- Main Tracking Logic ---

def main():
    """
    Main function to run the basketball tracking process.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the pre-trained YOLOv8 model
    try:
        model = YOLO(MODEL_NAME)
        print(f"Successfully loaded YOLO model: {MODEL_NAME}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}", file=sys.stderr)
        return

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{VIDEO_PATH}'", file=sys.stderr)
        print("Please make sure the file exists and is a valid video format.", file=sys.stderr)
        return

    # Get video properties for the output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- Construct the output video path ---
    # Get the original filename without the extension
    base_name = Path(VIDEO_PATH).stem
    # Create the new filename with a suffix
    output_filename = f"{base_name}_tracked.mp4"
    # Create the full path into the output directory
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Starting video processing... Output will be saved to '{output_path}'")
    while cap.isOpened():
        # Read a single frame from the video
        success, frame = cap.read()

        if not success:
            print("Reached the end of the video or failed to read a frame.")
            break

        # Run YOLOv8 tracking on the current frame.
        results = model.track(
            frame,
            persist=True,
            classes=[BASKETBALL_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        # The 'results[0].plot()' method draws all detections and tracks on the frame automatically.
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video file
        out.write(annotated_frame)

    # Clean up resources
    cap.release()
    out.release()
    print(f"Video processing finished. Tracked video saved to '{output_path}'.")

if __name__ == "__main__":
    main()