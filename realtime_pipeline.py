import os
import threading
import time
from utils import ensure_folder_exists
from audio_acquisition_preprocessing import preprocess_and_save_audio
from diarization_stage import perform_speaker_diarization
from audio_separation_stage import separate_speakers
from transcription_stage import process_transcription

# Global Variables
INPUT_AUDIO_FOLDER = "input_audio"            # Folder for incoming audio files
PROCESSED_AUDIO_FOLDER = "processed_audio"    # Processed audio after filtering
DIARIZED_FOLDER = "diarized_audio"            # Speaker segments for diarization
SEPARATED_AUDIO_FOLDER = "speaker_audio"      # Folder for speaker-separated audio
TRANSCRIPTIONS_FOLDER = "transcriptions"      # Final transcription output
LOG_FOLDER = "logs"                           # Error logs folder
OUTPUT_SUMMARY_FILE = "output_summary.txt"    # Consolidated final transcription file

# Ensure required folders exist
ensure_folder_exists(INPUT_AUDIO_FOLDER)
ensure_folder_exists(PROCESSED_AUDIO_FOLDER)
ensure_folder_exists(DIARIZED_FOLDER)
ensure_folder_exists(SEPARATED_AUDIO_FOLDER)
ensure_folder_exists(TRANSCRIPTIONS_FOLDER)
ensure_folder_exists(LOG_FOLDER)

def log_error(message):
    """Logs errors to a log file."""
    log_file = os.path.join(LOG_FOLDER, "pipeline_errors.log")
    with open(log_file, "a") as file:
        file.write(f"[ERROR] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def consolidate_transcriptions(transcriptions_folder, output_file):
    """Combines all transcriptions into a single summary file."""
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
    """
    Orchestrates the entire pipeline for a single audio file.
    Args:
        audio_file (str): Path to the input audio file.
    """
    try:
        print(f"Starting pipeline for: {audio_file}")
        base_name = os.path.splitext(os.path.basename(audio_file))[0]

        # Stage 1: Audio Preprocessing
        print("Step 1: Preprocessing Audio...")
        processed_audio_file = os.path.join(PROCESSED_AUDIO_FOLDER, f"{base_name}_processed.wav")
        preprocess_audio_file(audio_file, processed_audio_file)

        # Stage 2: Speaker Diarization
        print("Step 2: Performing Speaker Diarization...")
        diarized_output = os.path.join(DIARIZED_FOLDER, base_name)
        perform_speaker_diarization(processed_audio_file, diarized_output)

        # Stage 3: Speaker Audio Separation
        print("Step 3: Separating Speakers...")
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
     
