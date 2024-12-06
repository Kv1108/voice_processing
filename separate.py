import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import soundfile as sf

# Function to perform NMF-based voice separation
def nmf_voice_separation(audio_file, n_components=2):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    print("Audio loaded...")

    # Compute STFT to get the spectrogram
    S = np.abs(librosa.stft(y))
    print("Spectrogram computed...")

    # Apply NMF
    print("Applying NMF...")
    model = NMF(n_components=n_components, init='random', random_state=42, max_iter=500)
    W = model.fit_transform(S)  # Basis matrix (speakers)
    H = model.components_       # Activations (time-related data)

    # Reconstruct individual components
    print("Reconstructing sources...")
    sources = []
    for i in range(n_components):
        # Recreate the spectrogram for each component
        S_i = np.outer(W[:, i], H[i, :])
        # Inverse STFT to get back to audio
        y_i = librosa.istft(S_i * np.exp(1j * np.angle(librosa.stft(y))))
        sources.append(y_i)

    # Save each separated source
    for i, source in enumerate(sources):
        output_file = f"separated_speaker_{i + 1}.wav"
        sf.write(output_file, source, sr)
        print(f"Speaker {i + 1} audio saved as {output_file}")

# Main script
if __name__ == "__main__":
    # Path to the input audio file
    input_audio = "myrecording.wav"  # Replace with your audio file path

    # Perform voice separation
    nmf_voice_separation(input_audio, n_components=2)  # Adjust n_components based on expected speakers
