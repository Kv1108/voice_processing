import os
import webrtcvad
import librosa
import soundfile as sf
import numpy as np
from scipy.io.wavfile import write

# Global Variables
PROCESSED_AUDIO_FOLDER = "processed_audio"  # Input folder (from Stage 1)
VAD_OUTPUT_FOLDER = "speech_segments"       # Output folder for speech regions
SAMPLE_RATE = 16000                         # Required sample rate for VAD
FRAME_DURATION = 30                         # Frame duration in ms (10, 20, or 30 ms)

def ensure_folder_exists(folder_path):
    """Ensure that the specified folder exists."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def load_audio(file_path, sample_rate=SAMPLE_RATE):
    """Load audio file and ensure correct sample rate."""
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio

def frame_generator(audio, frame_duration, sample_rate):
    """
    Generator function to yield audio frames for VAD processing.
    Args:
        audio (numpy array): Audio samples.
        frame_duration (int): Frame duration in milliseconds.
        sample_rate (int): Audio sample rate.
    """
    frame_size = int(sample_rate * (frame_duration / 1000))
    num_frames = len(audio) // frame_size
    for i in range(num_frames):
        yield audio[i * frame_size : (i + 1) * frame_size]

def detect_speech_segments(audio, vad, sample_rate, frame_duration):
    """
    Detect speech segments using WebRTC VAD.
    Args:
        audio (numpy array): Audio signal.
        vad (webrtcvad.Vad): VAD instance.
        sample_rate (int): Sampling rate.
        frame_duration (int): Duration of each frame.
    Returns:
        List of (start, end) tuples of speech segments.
    """
    frames = list(frame_generator(audio, frame_duration, sample_rate))
    is_speech = [vad.is_speech(frame.tobytes(), sample_rate) for frame in frames]
    
    segments = []
    start, end = None, None
    for i, speech in enumerate(is_speech):
        if speech and start is None:
            start = i
        if not speech and start is not None:
            end = i
            segments.append((start * frame_duration / 1000, end * frame_duration / 1000))
            start, end = None, None
    if start is not None:
        segments.append((start * frame_duration / 1000, len(audio) / sample_rate))
    return segments

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
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # Aggressiveness mode (0-3, 3 being most aggressive)
    
    audio = load_audio(audio_file, SAMPLE_RATE)
    speech_segments = detect_speech_segments(audio, vad, SAMPLE_RATE, FRAME_DURATION)
    
    if not speech_segments:
        print("No speech detected.")
        return

    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    save_speech_segments(audio, speech_segments, SAMPLE_RATE, VAD_OUTPUT_FOLDER, file_name)

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
