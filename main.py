import subprocess
import os
from utils import ensure_folder_exists

def run_script(script_name):
    """
    Run a Python script as a subprocess.
    """
    return subprocess.Popen(["python", script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def wait_for_completion(process):
    """
    Wait for a subprocess to finish and capture its output.
    """
    try:
        output, error = process.communicate()  # Waits for the process to complete
        if output:
            print(f"Output from {process.args[1]}:\n{output.decode()}")
        if error:
            print(f"Error from {process.args[1]}:\n{error.decode()}")
    except Exception as e:
        print(f"Error while waiting for {process.args[1]}: {e}")

def preprocess_audio(audio_file, preprocessed_audio):
    """
    Preprocess audio by running preprocess_audio.py script.
    """
    try:
        print("Preprocessing the audio...")
        preprocess_command = ["python", "preprocess_audio.py", audio_file, preprocessed_audio]
        result = subprocess.run(preprocess_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error during audio preprocessing: {e.stderr.decode()}")
    except Exception as e:
        print(f"Unexpected error during audio preprocessing: {e}")

def main():
    try:
        # Step 1: Ensure folders for videos and audio exist
        ensure_folder_exists("recorded_videos")
        ensure_folder_exists("recorded_audio")
        ensure_folder_exists("transcription")

        # Step 2: Start CCTV.py and record.py in parallel
        print("Starting CCTV.py and record.py...")
        cctv_process = run_script("cctv.py")
        record_process = run_script("record.py")

        # Step 3: Wait for both processes to complete
        print("Waiting for CCTV.py and record.py to finish...")
        wait_for_completion(cctv_process)
        wait_for_completion(record_process)

        # Step 4: Locate audio file saved by record.py
        audio_folder = "recorded_audio"
        latest_audio_file = None
        for file in os.listdir(audio_folder):
            if file.endswith(".wav") and not file.endswith("_preprocessed.wav"):
                latest_audio_file = os.path.join(audio_folder, file)

        if not latest_audio_file:
            print("Error: No audio file found. Cannot proceed.")
            return

        # Step 5: Preprocess the located audio
        preprocessed_audio_file = latest_audio_file.replace(".wav", "_preprocessed.wav")
        preprocess_audio(latest_audio_file, preprocessed_audio_file)

        # Step 6: Run transcription.py on the preprocessed audio
        print("Starting transcription.py...")
        transcription_command = ["python", "transcription.py", preprocessed_audio_file]
        result = subprocess.run(transcription_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())

        print("Workflow completed successfully!")

    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except Exception as e:
        print(f"Error in main workflow: {e}")


if __name__ == "__main__":
    main()
