# video_analyzer.py

import cv2
import tkinter as tk
from tkinter import filedialog
from pose_detector import find_pose, label_keypoints

def select_video_file():
    """Opens a file dialog for the user to select a video file."""
    root = tk.Tk()
    root.withdraw()  # Hide the small tkinter window
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    return file_path

def process_video(video_path):
    """
    Processes a video file to perform pose estimation, display the result,
    and save the output to a new file.
    """
    # Check if a file was selected
    if not video_path:
        print("No video file selected. Exiting.")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Get video properties to create a VideoWriter object
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create the output file path
    output_path = video_path.rsplit('.', 1)[0] + '_output.mp4'
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video... Press 'q' in the video window to stop early.")

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
            cv2.imshow("YOLOv8 Pose Estimation - Video", final_frame)

            # 5. Write the processed frame to the output file
            out.write(final_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release everything when the job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing finished. Output video saved to: {output_path}")

# --- Main execution block ---
if __name__ == "__main__":
    video_to_process = select_video_file()
    process_video(video_to_process)
