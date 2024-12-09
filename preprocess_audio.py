import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter

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

def ramp_up_volume(audio, target_db=-3.0):
    """
    Ramps up the volume of the audio until it reaches the target dB level.
    """
    # Convert the target volume level (dB) to linear amplitude
    target_amplitude = 10 ** (target_db / 20.0)
    
    # Find the current volume level (rms)
    rms = np.sqrt(np.mean(audio**2))
    
    # Calculate the factor by which to scale the audio
    scale_factor = target_amplitude / rms if rms != 0 else 1
    
    # Ramp up the volume of the audio
    audio_ramped = audio * scale_factor
    return audio_ramped

def preprocess_audio(input_file, output_file):
    try:
        # Load audio
        audio, sr = librosa.load(input_file, sr=None)

        # Step 1: Noise reduction (simple noise gate)
        print("Applying noise reduction...")
        noise_profile = np.mean(audio[:1000])  # Estimate noise from the first 1000 samples
        audio = audio - noise_profile  # Subtract the noise profile

        # Step 2: Volume normalization (with volume ramping)
        print("Ramping up volume...")
        audio = ramp_up_volume(audio, target_db=-3.0)  # Ramp up to -3 dB

        # Step 3: Apply band-pass filtering
        print("Applying band-pass filter...")
        audio = apply_bandpass_filter(audio, sr)

        # Save the processed audio
        sf.write(output_file, audio, sr)
        print(f"Audio successfully preprocessed and saved to '{output_file}'")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
