# image_analyzer.py

import cv2
import tkinter as tk
from tkinter import filedialog
from pose_detector import find_pose, label_keypoints


KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]


def select_image_file():
    """Opens a file dialog for the user to select an image file."""
    root = tk.Tk()
    root.withdraw()  # Hide the small tkinter window
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
    )
    return file_path

def process_image(image_path):
    """
    Processes a single image file to perform pose estimation, display the result,
    and save the output to a new file.
    """
    # Check if a file was selected
    if not image_path:
        print("No image file selected. Exiting.")
        return

    # Read the image file
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image file at {image_path}")
        return

    # Create a variable to hold the final processed frame
    final_frame = frame.copy()

    # 1. Find the pose in the image
    results = find_pose(frame)

    # The results object is a generator, so we loop through it
    # For a static image, this will likely run once per detected person
    for r in results:
        # 2. Get the frame with the basic skeleton drawn on it
        annotated_frame = r.plot()

        # 3. Pass this frame and the result object to our new function to add labels
        # We use final_frame here to accumulate drawings if multiple people are detected
        final_frame = label_keypoints(annotated_frame, r)

    # Create the output file path
    output_path = image_path.rsplit('.', 1)[0] + '_output.' + image_path.rsplit('.', 1)[1]
    
    # Save the processed image
    cv2.imwrite(output_path, final_frame)
    print(f"Processing finished. Output image saved to: {output_path}")

    # Display the final, fully labeled frame
    cv2.imshow("YOLOv8 Pose Estimation - Image", final_frame)
    
    # Wait for a key press and then close the image window
    print("Press any key to close the image window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Main execution block ---
if __name__ == "__main__":
    image_to_process = select_image_file()
    process_image(image_to_process)
