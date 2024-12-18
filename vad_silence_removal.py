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
FRAME_DURATION = 10                         # Frame duration in ms (10, 20, or 30 ms)

def ensure_folder_exists(folder_path):
    """
    Ensures that a folder exists, creating it if necessary.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def load_audio(file_path, sample_rate=SAMPLE_RATE):
    """
    Loads an audio file and resamples it to the desired sample rate.
    """
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio

def frame_generator(audio, frame_duration, sample_rate):
    """
    Yields frames of audio data, given the frame duration and sample rate.
    """
    frame_size = int(sample_rate * (frame_duration / 1000))
    num_frames = len(audio) // frame_size
    for i in range(num_frames):
        yield audio[i * frame_size : (i + 1) * frame_size]

def detect_speech_segments(audio, vad, sample_rate, frame_duration):
    """
    Detects speech segments using WebRTC VAD.
    Returns only meaningful segments with non-zero duration.
    """
    frame_size = int(sample_rate * (frame_duration / 1000))
    is_speech = []

    # Create frames and process each for speech detection
    for frame in frame_generator(audio, frame_duration, sample_rate):
        if len(frame) == frame_size:  # Ensure frame size matches expected duration
            is_speech.append(vad.is_speech(frame.tobytes(), sample_rate))
        else:
            is_speech.append(False)

    # Identify speech segments
    segments = []
    start, end = None, None
    for i, speech in enumerate(is_speech):
        if speech and start is None:  # Start of a speech segment
            start = i * frame_duration / 1000
        if not speech and start is not None:  # End of a speech segment
            end = i * frame_duration / 1000
            if end - start > 0.1:  # Keep only meaningful segments (>100ms)
                segments.append((start, end))
            start, end = None, None

    # Handle case where audio ends with speech
    if start is not None:
        segments.append((start, len(audio) / sample_rate))

    return segments

def save_speech_segments(audio, speech_segments, sample_rate, output_folder, file_name):
    """
    Saves detected speech segments as separate audio files.
    Filters out segments with zero duration or negligible duration.
    """
    ensure_folder_exists(output_folder)
    for idx, (start, end) in enumerate(speech_segments):
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_audio = audio[start_sample:end_sample]

        # Skip segments with negligible or zero duration
        if len(segment_audio) == 0:
            continue

        output_path = os.path.join(output_folder, f"{file_name}_segment_{idx + 1}.wav")
        write(output_path, sample_rate, (segment_audio * 32767).astype(np.int16))
        print(f"Saved: {output_path}")

def process_vad(audio_file):
    """
    Processes a single audio file to detect and save speech segments.
    """
    print(f"Processing VAD for: {audio_file}")
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # Aggressiveness mode (0-3, 3 being most aggressive)

    audio = load_audio(audio_file, SAMPLE_RATE)
    speech_segments = detect_speech_segments(audio, vad, SAMPLE_RATE, FRAME_DURATION)

    # Ensure meaningful segments are processed
    if not speech_segments:
        print("No meaningful speech detected.")
        return

    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    save_speech_segments(audio, speech_segments, SAMPLE_RATE, VAD_OUTPUT_FOLDER, file_name)

def process_all_files():
    """
    Processes all .wav files in the input folder and detects speech segments.
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
