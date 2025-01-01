from flask import Flask, jsonify, request
import threading
import logging
import os
from final import record_and_transcribe, stop_transcription, transcription_folder

app = Flask(__name__)
os.environ["SB_LOCAL_FETCH_STRATEGY"] = "copy"

# Global variables for controlling the transcription process
transcription_thread = None
transcription_running = False
transcription_output = []  # To hold the live transcription text


@app.route('/start', methods=['POST'])
def start_transcription():
    global transcription_thread, transcription_running

    if transcription_running:
        return jsonify({"status": "error", "message": "Transcription already running."}), 400

    # Start the transcription process in a separate thread
    transcription_running = True
    transcription_thread = threading.Thread(target=record_and_transcribe)
    transcription_thread.start()

    return jsonify({"status": "success", "message": "Transcription started."})


@app.route('/stop', methods=['POST'])
def stop_transcription_api():
    global transcription_running

    if not transcription_running:
        return jsonify({"status": "error", "message": "No transcription is running."}), 400

    # Stop the transcription process
    stop_transcription()
    transcription_thread.join()
    transcription_running = False

    return jsonify({"status": "success", "message": "Transcription stopped.", "transcription": transcription_output})


@app.route('/get_transcription', methods=['GET'])
def get_transcription():
    global transcription_running

    if not transcription_running:
        return jsonify({"status": "error", "message": "No transcription is running."}), 400

    transcription_folder = "transcriptions"  # Folder containing transcription files
    
    try:
        # List all files in the transcription folder and sort by file name to get the latest file
        files = sorted(
            [f for f in os.listdir(transcription_folder) if f.endswith('_transcription.txt')],
            reverse=True
        )

        # Check if there are any files
        if not files:
            return jsonify({"status": "error", "message": "No transcription file found."}), 404

        # Get the latest file
        latest_file = os.path.join(transcription_folder, files[0])

        # Read the transcription data from the latest file
        with open(latest_file, "r") as file:
            transcription_data = file.read()

        return jsonify({"status": "success", "transcription": transcription_data})

    except Exception as e:
        logging.error(f"Error in /get_transcription: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/fetch_transcription', methods=['GET'])
def fetch_transcription():
    transcription_folder = 'transcriptions'
    if not os.path.exists(transcription_folder):
        return jsonify({"status": "error", "message": "Transcription folder not found."}), 404

    # Get the most recent transcription file
    transcription_files = [
        os.path.join(transcription_folder, f) for f in os.listdir(transcription_folder)
        if f.endswith('_transcription.txt')
    ]
    if not transcription_files:
        return jsonify({"status": "error", "message": "No transcription files found."}), 404

    latest_file = max(transcription_files, key=os.path.getmtime)

    def generate():
        with open(latest_file, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break
                yield line

    return app.response_class(generate(), mimetype='text/plain')

if __name__ == "__main__":
    # Run the Flask app on localhost and port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)