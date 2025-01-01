from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import os
import time
from final import record_and_transcribe, stop_transcription

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources={r"/*": {"origins": "*"}})

transcription_running = False
transcription_thread = None
transcription_output = []
transcription_folder = "transcriptions"


def get_latest_transcription():
    """
    Finds and returns the content of the latest transcription file.
    """
    if not os.path.exists(transcription_folder):
        return {"file": None, "content": []}

    files = [f for f in os.listdir(transcription_folder) if f.endswith("_transcription.txt")]
    if not files:
        return {"file": None, "content": []}

    latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(transcription_folder, f)))
    file_path = os.path.join(transcription_folder, latest_file)

    with open(file_path, "r") as file:
        content = file.readlines()

    return {"file": latest_file, "content": content}


@app.route('/latest-transcription', methods=['GET'])
def latest_transcription():
    """
    API to fetch the latest transcription file's content.
    """
    try:
        latest = get_latest_transcription()
        if not latest["file"]:
            return jsonify({"status": "error", "message": "No transcription files found"}), 404
        return jsonify({"status": "success", "file": latest["file"], "content": latest["content"]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/start', methods=['POST'])
def start_transcription():
    global transcription_running, transcription_thread
    if transcription_running:
        return jsonify({"status": "error", "message": "Transcription is already running."}), 400

    transcription_running = True
    transcription_thread = threading.Thread(target=record_and_transcribe)
    transcription_thread.start()

    socketio.start_background_task(target=stream_transcription)
    return jsonify({"status": "success", "message": "Transcription started."})


@app.route('/stop', methods=['POST'])
def stop_transcription_api():
    global transcription_running
    if not transcription_running:
        return jsonify({"status": "error", "message": "No transcription is running."}), 400

    stop_transcription()
    transcription_running = False
    return jsonify({"status": "success", "message": "Transcription stopped."})


def stream_transcription():
    global transcription_output
    while transcription_running:
        latest = get_latest_transcription()
        if latest["content"] and len(latest["content"]) > len(transcription_output):
            new_lines = latest["content"][len(transcription_output):]
            transcription_output = latest["content"]

            for line in new_lines:
                socketio.emit('transcription_update', {'transcription': line.strip()})
        socketio.sleep(1)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
