# Project Files TODO

This file provides an overview of all files and directories in the project. You can use it as a checklist to review or work on different components.

### Application & Logic Files
- [ ] `app.py`: The Flask web server that runs your web application. It handles receiving video from the browser, processing it, and sending it back.
- [ ] `main.py`: The original script that runs real-time pose estimation from your computer's webcam and displays it in an OpenCV window.
- [ ] `video_analyzer.py`: A script that processes a local video file for pose estimation and saves the output as a new video.
- [ ] `image_analyzer.py`: A script that processes a local image file for pose estimation and saves the output as a new image.
- [ ] `ball_tracker.py`: A script specifically for tracking a basketball in a video file and saving the output.
- [ ] `pose_detector.py`: A core module containing the main functions for finding poses and labeling keypoints. This is used by all the other analysis scripts.

### Web Files
- [.] `templates/index.html`: The HTML file that creates the structure and layout of your web application's front end.

### Model & Data Files
- [ ] `yolov8n-pose.pt`: The pre-trained YOLOv8 model file used for pose estimation.
- [ ] `yolov8n.pt`: The pre-trained YOLOv8 model file used for general object detection (like in `ball_tracker.py`).
- [ ] `basketball.mp4`: A sample video file used for testing.

### Project & Documentation
- [ ] `README.md`: The documentation file for your project, which includes setup and usage instructions.
- [ ] `requirements.txt`: A file listing the Python libraries needed to run your project.
- [ ] `.gitignore`: A file that tells Git which files and folders to ignore.

### Directories
- [ ] `output/`: The folder where processed videos from `ball_tracker.py` are saved.
- [ ] `static/`: A folder for static assets like your `example_output.gif`.
- [ ] `templates/`: The folder where Flask looks for HTML files.
- [ ] `.venv/`: The directory for your Python virtual environment.
- [ ] `__pycache__/`: A folder where Python stores compiled bytecode to speed up execution.
- [ ] `.git/`: Directory containing all the necessary files for the Git repository.
- [ ] `Resources/`: Directory for project resources.
