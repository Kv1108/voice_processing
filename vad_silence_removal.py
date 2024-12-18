import os
import librosa
import soundfile as sf
import numpy as np
from scipy.io.wavfile import write
from silero_vad import VoiceActivityDetector

# Global Variables
PROCESSED_AUDIO_FOLDER = "processed_audio"  # Input folder (from Stage 1)
VAD_OUTPUT_FOLDER = "speech_segments"       # Output folder for speech regions
SAMPLE_RATE = 16000                         # Required sample rate for VAD

def ensure_folder_exists(folder_path):
    """Ensure that the specified folder exists."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def load_audio(file_path, sample_rate=SAMPLE_RATE):
    """Load audio file and ensure correct sample rate."""
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio

def detect_speech_segments_silero(audio, sample_rate):
    """
    Detect speech segments using Silero VAD.
    Args:
        audio (numpy array): Audio signal.
        sample_rate (int): Sampling rate.
    Returns:
        List of (start, end) tuples of speech segments.
    """
    detector = VoiceActivityDetector(sample_rate=sample_rate)
    speech_probs = detector(audio)  # Returns a probability for each frame
    speech_segments = []

    # Convert frame-level probabilities to time segments
    frame_duration = detector.frame_duration_sec
    start, end = None, None

    for i, prob in enumerate(speech_probs):
        if prob > 0.5:  # Speech detected
            if start is None:
                start = i * frame_duration
        else:
            if start is not None:
                end = i * frame_duration
                speech_segments.append((start, end))
                start, end = None, None
    if start is not None:
        speech_segments.append((start, len(audio) / sample_rate))
    
    return speech_segments

def save_speech_segments(audio, speech_segments, sample_rate, output_folder, file_name):
    """
    Save speech segments as separate audio files.
    Args:
        audio (numpy array): Audio signal.
        speech_segments (list): List of (start, end) tuples.
        sample_rate (int): Sampling rate.
        output_folder (str): Folder to save speech segments.
        file_name (str): Original audio file name.
    """
    ensure_folder_exists(output_folder)
    for idx, (start, end) in enumerate(speech_segments):
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_audio = audio[start_sample:end_sample]
        output_path = os.path.join(output_folder, f"{file_name}_segment_{idx + 1}.wav")
        write(output_path, sample_rate, (segment_audio * 32767).astype(np.int16))
        print(f"Saved: {output_path}")

def process_vad(audio_file):
    """
    Perform VAD and silence removal on an audio file.
    Args:
        audio_file (str): Path to the input audio file.
    """
    print(f"Processing VAD for: {audio_file}")
    try:
        audio = load_audio(audio_file, SAMPLE_RATE)
        speech_segments = detect_speech_segments_silero(audio, SAMPLE_RATE)
        
        if not speech_segments:
            print("No speech detected.")
            return

        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        save_speech_segments(audio, speech_segments, SAMPLE_RATE, VAD_OUTPUT_FOLDER, file_name)
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

def process_all_files():
    """
    Process all preprocessed audio files for VAD and silence removal.
    """
    ensure_folder_exists(PROCESSED_AUDIO_FOLDER)
    ensure_folder_exists(VAD_OUTPUT_FOLDER)
    
    for audio_file in os.listdir(PROCESSED_AUDIO_FOLDER):
        audio_path = os.path.join(PROCESSED_AUDIO_FOLDER, audio_file)
        if audio_path.endswith(".wav"):
            try:
                process_vad(audio_path)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

if __name__ == "__main__":
    print("Starting VAD and Silence Removal...")
    process_all_files()
