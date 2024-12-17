import os
import subprocess
import threading
from utils import ensure_folder_exists

def run_script(script_name, args=None):
    command = ["python", script_name]
    if args:
        command.extend(args)
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def wait_for_completion(process):
    try:
        output, error = process.communicate()  # Waits for the process to complete
        if output:
            print(f"Output from {process.args[1]}:\n{output.decode()}")
        if error:
            print(f"Error from {process.args[1]}:\n{error.decode()}")
    except Exception as e:
        print(f"Error while waiting for {process.args[1]}: {e}")

def preprocess_audio(latest_audio_file, preprocessed_audio):
    try:
        print("Preprocessing the audio...")
        preprocess_command = ["python", "preprocess_audio.py", latest_audio_file, preprocessed_audio]
        result = subprocess.run(preprocess_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error during audio preprocessing: {e.stderr.decode()}")
    except Exception as e:
        print(f"Unexpected error during audio preprocessing: {e}")

def clustering_and_transcription(preprocessed_audio):
    try:
        print("Starting clustering.py...")
        clustering_process = run_script("clustering.py", [preprocessed_audio])
        wait_for_completion(clustering_process)

        print("Starting transcription.py...")
        transcription_process = run_script("transcription.py")
        wait_for_completion(transcription_process)

    except Exception as e:
        print(f"Error during clustering or transcription: {e}")

def monitor_recorded_audio():
    audio_folder = "recorded_audio"
    processed_files = set()

    while True:
        try:
            audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
            for file in audio_files:
                if file not in processed_files:
                    latest_audio_file = os.path.join(audio_folder, file)
                    preprocessed_audio_file = latest_audio_file.replace(".wav", "_preprocessed.wav")

                    preprocess_audio(latest_audio_file, preprocessed_audio_file)

                    # Run clustering and transcription
                    clustering_and_transcription(preprocessed_audio_file)

                    processed_files.add(file)
        except KeyboardInterrupt:
            print("Process interrupted by user.")
            break
        except Exception as e:
            print(f"Error in monitoring recorded audio: {e}")

def main():
    try:
        ensure_folder_exists("recorded_audio")
        ensure_folder_exists("transcriptions")
        ensure_folder_exists("clusters")

        print("Starting CCTV.py and record.py...")
        cctv_process = run_script("cctv.py")
        record_process = run_script("record.py")

        # Start monitoring for audio files
        monitor_thread = threading.Thread(target=monitor_recorded_audio, daemon=True)
        monitor_thread.start()

        print("Workflow is running. Press Ctrl+C to stop.")
        
        cctv_process.wait()
        record_process.wait()

    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except Exception as e:
        print(f"Error in main workflow: {e}")

if __name__ == "__main__":
    main()
