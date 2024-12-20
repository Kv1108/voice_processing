

import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter


# what sound to allow instructions
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a

# 300 and 3400 Hz of voice
def apply_bandpass_filter(audio, sr, lowcut=30.0, highcut=3400.0): 
    b, a = butter_bandpass(lowcut, highcut, sr, order=6)
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

# adjusts the loudness 
def ramp_up_volume(audio, target_db=-0.0): 
    target_amplitude = 10 ** (target_db / 20.0)
    rms = np.sqrt(np.mean(audio**2))
    scale_factor = target_amplitude / rms if rms != 0 else 1
    audio_ramped = audio * scale_factor
    return audio_ramped

# increases the volume by 15 db
def boost_audio_volume(audio, boost_db=25):
    boost_factor = 10 ** (boost_db / 20.0)
    boosted_audio = audio * boost_factor
    return np.clip(boosted_audio, -1.0, 1.0)  # Prevent clipping

def preprocess_audio(input_file, output_file, boost_db=10):
    try:
       
        audio, sr = librosa.load(input_file, sr=None)
        print("Applying noise reduction...")
        noise_profile = np.mean(audio[:1000])  # Estimate noise from the first 1000 samples
        audio = audio - noise_profile  # Subtract the noise profile

        print("Ramping up volume...")
        audio = ramp_up_volume(audio, target_db=-0.0)  

        print(f"Boosting volume by {boost_db} dB...")
        audio = boost_audio_volume(audio, boost_db=boost_db)

        print("Applying band-pass filter...")
        audio = apply_bandpass_filter(audio, sr)

        sf.write(output_file, audio, sr)
        print(f"Audio successfully preprocessed and saved to '{output_file}'")
    except Exception as e:
        print(f"Error during preprocessing: {e}")


