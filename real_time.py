import tkinter as tk
from tkinter import messagebox
import subprocess
import os
from utils import ensure_folder_exists
from cluster import process_vad_and_clustering

# Function to run subprocesses
def run_subprocess(command):
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode(), result.stderr.decode()
    except subprocess.CalledProcessError as e:
        return e.stdout.decode(), e.stderr.decode()
    except Exception as e:
        return str(e), None

# Function to start the real-time processing
def start_processing():
    try:
        # Ensure necessary folders exist
        ensure_folder_exists("recorded_videos")
        ensure_folder_exists("recorded_audio")
        ensure_folder_exists("transcription")
        ensure_folder_exists("clusters")

        # Step 1: Run CCTV and Record
        output, error = run_subprocess(["python", "cctv.py"])
        if error:
            messagebox.showerror("Error", f"CCTV.py Error: {error}")
            return

        output, error = run_subprocess(["python", "record.py"])
        if error:
            messagebox.showerror("Error", f"Record.py Error: {error}")
            return

        # Step 2: Process the latest audio file
        audio_folder = "recorded_audio"
        latest_audio_file = None
        for file in os.listdir(audio_folder):
            if file.endswith(".wav") and not file.endswith("_preprocessed.wav"):
                latest_audio_file = os.path.join(audio_folder, file)

        if not latest_audio_file:
            messagebox.showerror("Error", "No audio file found!")
            return

        preprocessed_audio_file = latest_audio_file.replace(".wav", "_preprocessed.wav")
        preprocess_audio(latest_audio_file, preprocessed_audio_file)

        # Step 3: Perform Clustering
        process_vad_and_clustering(preprocessed_audio_file)

        # Step 4: Run transcription
        output, error = run_subprocess(["python", "transcription.py", preprocessed_audio_file])
        if error:
            messagebox.showerror("Error", f"Transcription Error: {error}")
            return

        messagebox.showinfo("Success", "Real-time processing completed successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error: {e}")

# GUI Setup
root = tk.Tk()
root.title("Real-time Audio Processing")
root.geometry("400x300")

# Add a label for the app
label = tk.Label(root, text="Real-time Audio Processing", font=("Arial", 16))
label.pack(pady=20)

# Add a Start button
start_button = tk.Button(root, text="Start Processing", font=("Arial", 14), command=start_processing)
start_button.pack(pady=20)

# Add a Quit button
quit_button = tk.Button(root, text="Quit", font=("Arial", 14), command=root.quit)
quit_button.pack(pady=10)

# Start the GUI loop
root.mainloop()
