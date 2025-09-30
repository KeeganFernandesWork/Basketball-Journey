import cv2
import numpy as np
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from pose_detector import find_pose, label_keypoints
import eventlet

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

def process_image_data(data_url):
    """
    Decodes a base64 data URL into an image frame, processes it for pose estimation,
    and re-encodes it to be sent back to the client.
    """
    # Extract the base64 part of the data URL
    header, encoded = data_url.split(",", 1)
    
    # Decode the base64 string
    data = base64.b64decode(encoded)
    
    # Convert the binary data to a NumPy array
    nparr = np.frombuffer(data, np.uint8)
    
    # Decode the NumPy array into an image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # --- Perform Pose Estimation ---
    final_frame = frame.copy() # Start with a copy
    # Note: Tracking is not effective in this request-response setup, so we set it to False.
    results = find_pose(frame, track=False) 

    for r in results:
        annotated_frame = r.plot()
        final_frame = label_keypoints(annotated_frame, r)
    
    # Encode the processed frame back to JPEG
    _, buffer = cv2.imencode('.jpg', final_frame)
    
    # Encode the buffer to base64
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    return "data:image/jpeg;base64," + encoded_image

@app.route('/')
def index():
    """Render the main web page."""
    return render_template('index.html')

@socketio.on('image')
def handle_image(data_url):
    """
    Receives an image from the client, processes it, and sends it back.
    """
    # Process the image
    processed_image_url = process_image_data(data_url)
    
    # Send the processed image back to the client
    emit('processed_image', {'image': processed_image_url})

if __name__ == '__main__':
    # Run the app with SocketIO
    print("Starting server... Go to http://127.0.0.1:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
