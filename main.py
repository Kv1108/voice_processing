import subprocess
import os
import time

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

def main():
    try:
        # Step 1: Start CCTV.py and record.py in parallel
        print("Starting CCTV.py and record.py...")
        cctv_process = run_script("CCTV.py")
        record_process = run_script("record.py")

        # Step 2: Wait for both processes to complete
        print("Waiting for CCTV.py and record.py to finish...")
        wait_for_completion(cctv_process)
        wait_for_completion(record_process)

        # Step 3: Ensure the audio file exists before proceeding
        audio_file = "myrecording.wav"
        if not os.path.exists(audio_file):
            print(f"Error: Expected audio file '{audio_file}' not found. Cannot proceed.")
            return

        # Step 4: Preprocess the audio
        print("Preprocessing the audio...")
        preprocessed_audio = "preprocessed_audio.wav"
        preprocess_command = ["python", "preprocess_audio.py", audio_file, preprocessed_audio]
        subprocess.run(preprocess_command)

        # Step 5: Run transcription.py on the preprocessed audio
        print("Starting transcription.py...")
        transcription_command = ["python", "transcription.py", preprocessed_audio]
        subprocess.run(transcription_command)

        print("Workflow completed successfully!")

    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except Exception as e:
        print(f"Error in main workflow: {e}")

if __name__ == "__main__":
    main()
