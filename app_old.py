from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import os
from final import record_and_transcribe, stop_transcription

app = Flask(__name__)
socketio = SocketIO(app)  # Initialize Flask-SocketIO
os.environ["SB_LOCAL_FETCH_STRATEGY"] = "copy"

# Global variables for controlling the transcription process
transcription_thread = None
transcription_running = False
transcription_output = []  # To hold the live transcription text


def stream_transcription():
    """
    Streams the transcription output to clients via WebSocket.
    """
    global transcription_output

    while transcription_running:
        if transcription_output:
            # Send the latest transcription chunk to clients
            socketio.emit('transcription_update', {'transcription': transcription_output[-1]})
        socketio.sleep(1)  # Sleep for a second before checking again


@app.route('/start', methods=['POST'])
def start_transcription():
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
