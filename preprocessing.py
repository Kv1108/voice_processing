import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter
import os
import datetime
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Band-pass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a

def apply_bandpass_filter(audio, sr, lowcut=300.0, highcut=3400.0):
    b, a = butter_bandpass(lowcut, highcut, sr, order=6)
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

# Volume adjustment
def ramp_up_volume(audio, target_db=-0.0):
    target_amplitude = 10 ** (target_db / 20.0)
    rms = np.sqrt(np.mean(audio**2))
    scale_factor = target_amplitude / rms if rms != 0 else 1
    audio_ramped = audio * scale_factor
    return audio_ramped

def boost_audio_volume(audio, boost_db=15):
    boost_factor = 10 ** (boost_db / 20.0)
    boosted_audio = audio * boost_factor
    return np.clip(boosted_audio, -1.0, 1.0)

# Noise reduction
def reduce_noise(audio, noise_estimation_samples=10000):
    """Subtracts a noise profile from the audio."""
    noise_profile = np.mean(audio[:noise_estimation_samples])  # Estimate noise
    reduced_audio = audio - noise_profile
    return reduced_audio

# Main preprocessing function
def preprocess_audio(input_file, output_directory, boost_db=10):
    try:
        # Load audio
        logging.info(f"Loading audio file: {input_file}")
        audio, sr = librosa.load(input_file, sr=None)

        # Preprocessing steps
        logging.info("Reducing noise...")
        audio = reduce_noise(audio)

        logging.info("Ramping up volume...")
        audio = ramp_up_volume(audio, target_db=-0.0)

        logging.info(f"Boosting volume by {boost_db} dB...")
        audio = boost_audio_volume(audio, boost_db=boost_db)

        logging.info("Applying band-pass filter (300 Hz - 3400 Hz)...")
        audio = apply_bandpass_filter(audio, sr)

        # Save preprocessed audio
        file_name = os.path.basename(input_file)
        file_root, file_ext = os.path.splitext(file_name)
        output_file = os.path.join(output_directory, f"{file_root}_preprocessed{file_ext}")

        sf.write(output_file, audio, sr)
        logging.info(f"Audio successfully preprocessed and saved to '{output_file}'")

        return output_file

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        return None

# Real-time preprocessing for long audio files
def preprocess_large_audio(input_file, output_directory, clip_duration=30*60, boost_db=10):
    try:
        # Load audio
        logging.info(f"Loading large audio file: {input_file}")
        audio, sr = librosa.load(input_file, sr=None)

        # Split into smaller clips
        total_samples = len(audio)
        clip_samples = clip_duration * sr
        num_clips = (total_samples + clip_samples - 1) // clip_samples

        logging.info(f"Audio will be split into {num_clips} segments of {clip_duration} seconds each.")

        for i in range(num_clips):
            start_sample = i * clip_samples
            end_sample = min((i + 1) * clip_samples, total_samples)
            audio_clip = audio[start_sample:end_sample]

            # Preprocess each segment
            logging.info(f"Processing segment {i + 1}/{num_clips}...")
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            segment_file = os.path.join(output_directory, f"segment_{timestamp}_{i + 1}.wav")
            sf.write(segment_file, audio_clip, sr)  # Save intermediate clip

            preprocess_audio(segment_file, output_directory, boost_db=boost_db)

        logging.info("All segments have been processed successfully.")

    except Exception as e:
        logging.error(f"Error during large file preprocessing: {e}")

# Ensure output directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Main entry point
if __name__ == "__main__":
    input_file = "path_to_input_audio.wav"  # Replace with actual input file
    output_directory = "preprocessed_audio"
    ensure_directory_exists(output_directory)

    logging.info("Starting preprocessing of large audio file...")
    preprocess_large_audio(input_file, output_directory)
    logging.info("Preprocessing complete.")