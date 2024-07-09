from flask import Flask, render_template, Response, jsonify
import threading
import cv2
import json
from people_counter import people_counter  # Import your people counter function

# Create Flask app
app = Flask(__name__)

# Global variables for storing people count data
total_entered = 0
total_exited = 0
total_inside = 0

# Lock for thread-safe counting
lock = threading.Lock()

# Load configuration
with open("utils/config.json", "r") as file:
    config = json.load(file)

# Function to update counts
def update_counts(entered, exited):
    global total_entered, total_exited, total_inside
    with lock:
        total_entered += entered
        total_exited += exited
        total_inside = total_entered - total_exited

# Wrapper function to run the people counter
def run_people_counter():
    global total_entered, total_exited, total_inside

    for entered, exited, frame in people_counter():  # Modify your people_counter to yield entered, exited, frame
        update_counts(entered, exited)

        # Stream the frame
        _, encoded_image = cv2.imencode('.jpg', frame)
        frame = encoded_image.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('try.html')

@app.route('/video_feed')
def video_feed():
    return Response(run_people_counter(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts')
def counts():
    with lock:
        data = {
            "total_entered": total_entered,
            "total_exited": total_exited,
            "total_inside": total_inside
        }
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
