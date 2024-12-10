import os
import librosa
import soundfile as sf

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def preprocess_audio(input_path, output_path):
    try:
        audio, sr = librosa.load(input_path, sr=None)
        audio = librosa.effects.preemphasis(audio)  # Apply pre-emphasis
        sf.write(output_path, audio, sr)
        print(f"Preprocessed audio saved to: {output_path}")
    except Exception as e:
        print(f"Error in preprocessing audio: {e}")
