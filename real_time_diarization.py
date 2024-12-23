import os
import time
from speaker_separation import separate_speakers
from transcription_formatter import format_transcription

# Folder for audio files and diarization results
input_folder = "recorded_audio"
output_folder = "diarized_audio"
os.makedirs(output_folder, exist_ok=True)

def process_audio_chunk(audio_path, output_path):
    """
    Process an audio chunk for diarization and save the results.
    
    Parameters:
        audio_path (str): Path to the input audio file.
        output_path (str): Path to save the diarization results.
    """
    # Step 1: Separate speakers and get segments
    speaker_segments = separate_speakers(audio_path)

    # Step 2: Format and save transcription in conversation style
    if speaker_segments:
        format_transcription(speaker_segments, audio_path, output_path)
    else:
        print(f"No valid speaker segments for {audio_path}.")

def real_time_processing():
    """Main loop for real-time processing."""
    processed_files = set()
    while True:
        audio_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
        for audio_file in audio_files:
            if audio_file in processed_files:
                continue  # Skip already processed files
            input_path = os.path.join(input_folder, audio_file)
            process_audio_chunk(input_path, output_folder)
            processed_files.add(audio_file)
            os.remove(input_path)  # Clean up processed files
        time.sleep(1)  # Polling interval for new files

if __name__ == "__main__":
    real_time_processing()
