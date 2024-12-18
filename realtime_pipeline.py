import os
import threading
import time
from utils import ensure_folder_exists
from audio_acquisition_preprocessing import preprocess_and_save_audio
from vad_silence_removal import process_vad
from speaker_separation import process_speaker_separation
from transcription_stage import process_transcription

# Global Variables
INPUT_AUDIO_FOLDER = "input_audio"            # Folder for incoming audio files
PROCESSED_AUDIO_FOLDER = "processed_audio"    # Processed audio after filtering
VAD_OUTPUT_FOLDER = "speech_segments"         # Folder for speech segments after VAD
SEPARATED_AUDIO_FOLDER = "speaker_audio"      # Folder for speaker-separated audio
TRANSCRIPTIONS_FOLDER = "transcriptions"      # Final transcription output
LOG_FOLDER = "logs"                           # Error logs folder
OUTPUT_SUMMARY_FILE = "output_summary.txt"    # Consolidated final transcription file

# Ensure required folders exist
ensure_folder_exists(INPUT_AUDIO_FOLDER)
ensure_folder_exists(PROCESSED_AUDIO_FOLDER)
ensure_folder_exists(VAD_OUTPUT_FOLDER)
ensure_folder_exists(SEPARATED_AUDIO_FOLDER)
ensure_folder_exists(TRANSCRIPTIONS_FOLDER)
ensure_folder_exists(LOG_FOLDER)

def log_error(message):
    log_file = os.path.join(LOG_FOLDER, "pipeline_errors.log")
    with open(log_file, "a") as file:
        file.write(f"[ERROR] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def consolidate_transcriptions(transcriptions_folder, output_file):
    with open(output_file, "w") as output:
        output.write("Consolidated Transcriptions\n")
        output.write("=" * 40 + "\n\n")
        for transcription_file in os.listdir(transcriptions_folder):
            if transcription_file.endswith(".txt"):
                speaker_name = transcription_file.replace("_transcription.txt", "")
                output.write(f"### {speaker_name} ###\n")
                with open(os.path.join(transcriptions_folder, transcription_file), "r") as file:
                    output.write(file.read() + "\n")
        print(f"Final transcription summary saved to: {output_file}")

def process_audio_pipeline(audio_file):
    try:
        print(f"Starting pipeline for: {audio_file}")
        base_name = os.path.splitext(os.path.basename(audio_file))[0]

        # Stage 1: Audio Preprocessing
        print("Step 1: Preprocessing Audio...")
        processed_audio_file = os.path.join(PROCESSED_AUDIO_FOLDER, f"{base_name}_processed.wav")
        preprocess_and_save_audio(audio_file, processed_audio_file)

        # Stage 2: VAD and Silence Removal
        print("Step 2: Performing VAD and Silence Removal...")
        process_vad(processed_audio_file)

        # Stage 3: Speaker Audio Separation
        print("Step 3: Separating Speakers...")
        for segment_file in os.listdir(VAD_OUTPUT_FOLDER):
            if segment_file.startswith(base_name) and segment_file.endswith(".wav"):
                segment_path = os.path.join(VAD_OUTPUT_FOLDER, segment_file)
                separate_speakers(segment_path, SEPARATED_AUDIO_FOLDER)

        # Stage 4: Transcription
        print("Step 4: Transcribing Separated Audio...")
        process_transcription(SEPARATED_AUDIO_FOLDER, TRANSCRIPTIONS_FOLDER)

        print(f"Pipeline completed for: {audio_file}")
    except Exception as e:
        error_message = f"Error processing {audio_file}: {e}"
        log_error(error_message)
        print(error_message)

def monitor_input_folder():
    print("Monitoring input folder for new audio files...")
    processed_files = set()

    while True:
        try:
            for audio_file in os.listdir(INPUT_AUDIO_FOLDER):
                if audio_file.endswith(".wav") and audio_file not in processed_files:
                    audio_path = os.path.join(INPUT_AUDIO_FOLDER, audio_file)
                    threading.Thread(target=process_audio_pipeline, args=(audio_path,)).start()
                    processed_files.add(audio_file)
        except Exception as e:
            log_error(f"Error monitoring input folder: {e}")
        time.sleep(5)  # Check for new files every 5 seconds

if __name__ == "__main__":
    monitor_input_folder()
