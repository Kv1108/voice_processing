from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import os
import time
from final import record_and_transcribe, stop_transcription

app = Flask(__name__)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Enable CORS for all routes and WebSocket connections
CORS(app, resources={r"/*": {"origins": "*"}})

os.environ["SB_LOCAL_FETCH_STRATEGY"] = "copy"

# Global variables for controlling the transcription process
transcription_thread = None
transcription_running = False
transcription_output = []  # To hold the live transcription text
transcription_file_path = "transcriptions"  # Directory where transcription files are stored


def read_latest_transcription_file():
    """
    Reads the latest transcription file and returns its content.
    """
    if not os.path.exists(transcription_file_path):
        return []

    # Get the latest file in the transcription folder
    files = sorted(os.listdir(transcription_file_path), reverse=True)
    if not files:
        return []

    latest_file = os.path.join(transcription_file_path, files[0])

    # Read the contents of the latest file
    try:
        with open(latest_file, "r") as f:
            return f.readlines()  # Return all lines as a list
    except Exception as e:
        print(f"Error reading file {latest_file}: {e}")
        return []


def stream_transcription():
    """
    Streams the transcription output to clients via WebSocket.
    """
    global transcription_output

    while transcription_running:
        # Read the latest file content
        updated_transcription = read_latest_transcription_file()

        # Only send new lines to the client
        if len(updated_transcription) > len(transcription_output):
            new_lines = updated_transcription[len(transcription_output):]
            transcription_output = updated_transcription

            # Emit each new line to the WebSocket client
            for line in new_lines:
                socketio.emit('transcription_update', {'transcription': line.strip()})
        
        socketio.sleep(1)  # Sleep for a second before checking again


@app.route('/start', methods=['POST'])
def start_transcription():
    print("Received request to start transcription")  # Log request

    """
    API endpoint to start the recording and transcription process.
    """
    global transcription_thread, transcription_running

    if transcription_running:
        return jsonify({"status": "error", "message": "Transcription already running."}), 400

    # Reset transcription output
    transcription_output.clear()

    # Start the transcription process in a separate thread
    transcription_running = True
    transcription_thread = threading.Thread(target=record_and_transcribe)
    transcription_thread.start()

    # Start streaming transcription updates via WebSocket
    socketio.start_background_task(target=stream_transcription)

    return jsonify({"status": "success", "message": "Transcription started."})


@app.route('/stop', methods=['POST'])
def stop_transcription_api():
    """
    API endpoint to stop the recording and transcription process.
    """
    global transcription_running

    if not transcription_running:
        return jsonify({"status": "error", "message": "No transcription is running."}), 400

    # Stop the transcription process
    stop_transcription()
    transcription_running = False

    return jsonify({"status": "success", "message": "Transcription stopped."})


@socketio.on('connect')
def handle_connect():
    """
    Handles a new WebSocket connection.
    """
    print("Client connected")


@socketio.on('disconnect')
def handle_disconnect():
    """
    Handles a WebSocket disconnection.
    """
    print("Client disconnected")


if __name__ == "__main__":
    # Run the Flask app with WebSocket support
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
