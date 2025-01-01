from flask import Flask, jsonify, request
import threading
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
    """
    API endpoint to start the recording and transcription process.
    """
    global transcription_thread, transcription_running, transcription_output

    if transcription_running:
        return jsonify({"status": "error", "message": "Transcription already running."}), 400

    # Reset transcription output
    transcription_output = []

    # Start the transcription process in a separate thread
    transcription_running = True
    transcription_thread = threading.Thread(target=record_and_transcribe, args=(transcription_output,))
    transcription_thread.start()

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
    transcription_thread.join()
    transcription_running = False

    return jsonify({"status": "success", "message": "Transcription stopped.", "transcription": transcription_output})


@app.route('/get_transcription', methods=['GET'])
def get_transcription():
    """
    API endpoint to fetch the live transcription text.
    """
    global transcription_output

    if not transcription_running:
        return jsonify({"status": "error", "message": "No transcription is running."}), 400

    return jsonify({"status": "success", "transcription": transcription_output})

# @app.route('/fetch_transcription', methods=['GET'])
#   tried to read by line instead of whole document 
#  def fetch_transcription():
#     transcription_folder = 'transcriptions'  # Folder where transcription files are stored

#     if not os.path.exists(transcription_folder):
#         return jsonify({"status": "error", "message": "Transcription folder not found."}), 404

#     # Get all transcription files in the folder
#     transcription_files = [
#         os.path.join(transcription_folder, f) for f in os.listdir(transcription_folder)
#         if f.endswith('_transcription.txt')
#     ]

#     if not transcription_files:
#         return jsonify({"status": "error", "message": "No transcription files found."}), 404

#     # Find the most recently created transcription file
#     latest_file = max(transcription_files, key=os.path.getmtime)

#     # Read and return the content of the latest file
#     with open(latest_file, 'r') as file:
#         transcription_content = file.read()

#     return jsonify({
#         "status": "success",
#         "file_name": os.path.basename(latest_file),
#         "transcription": transcription_content
#     })

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