import os
import numpy as np
import sounddevice as sd
import librosa
import noisereduce as nr
import soundfile as sf
from scipy.io.wavfile import write
import time
# Global Variables
AUDIO_OUTPUT_FOLDER = "processed_audio"
RAW_AUDIO_FOLDER = "raw_audio"
DURATION = 10  # Audio recording duration in seconds
SAMPLE_RATE = 16000  # Standard sample rate for speech processing


def ensure_folder_exists(folder_path):
    """Ensure that the specified folder exists."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    """
    Records audio in real-time from the microphone.
    Args:
        duration (int): Duration of recording in seconds.
        sample_rate (int): Sampling rate for recording.
    Returns:
        np.ndarray: Recorded audio signal.
    """
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return np.squeeze(audio)  # Remove single-dimensional entries


def apply_noise_reduction(audio, sr):
    print("Applying noise reduction...")
    reduced_audio = nr.reduce_noise(y=audio, sr=sr)
    print("Noise reduction applied.")
    return reduced_audio


def normalize_audio(audio):
    print("Normalizing audio...")
    max_val = np.max(np.abs(audio))
    normalized_audio = audio / max_val if max_val > 0 else audio
    print("Audio normalized.")
    return normalized_audio


def save_audio(audio, sample_rate, folder, file_name):
    ensure_folder_exists(folder)
    output_path = os.path.join(folder, file_name)
    sf.write(output_path, audio, sample_rate)
    print(f"Audio saved to {output_path}")
    return output_path


def preprocess_and_save_audio(audio, sample_rate, output_name):
    # Step 1: Noise Reduction
    audio = apply_noise_reduction(audio, sample_rate)

    # Step 2: Normalization
    audio = normalize_audio(audio)

    # Step 3: Save Processed Audio
    output_path = save_audio(audio, sample_rate, AUDIO_OUTPUT_FOLDER, output_name)
    return output_path


def main():
    ensure_folder_exists(RAW_AUDIO_FOLDER)
    ensure_folder_exists(AUDIO_OUTPUT_FOLDER)

    while True:
        try:
            print("\n Real-Time Audio Acquisition & Preprocessing")
            audio_signal = record_audio()

            # Save raw audio for backup
            raw_file_name = f"raw_audio_{int(time.time())}.wav"
            raw_path = save_audio(audio_signal, SAMPLE_RATE, RAW_AUDIO_FOLDER, raw_file_name)

            # Preprocess the audio and save it
            processed_file_name = f"processed_audio_{int(time.time())}.wav"
            processed_path = preprocess_and_save_audio(audio_signal, SAMPLE_RATE, processed_file_name)

            print(f"Audio processing complete. Processed file: {processed_path}")

            # Handoff: Returning processed path for pipeline integration
            return processed_path

        except KeyboardInterrupt:
            print("Stopping real-time audio acquisition...")
            break


if __name__ == "__main__":
    main()
