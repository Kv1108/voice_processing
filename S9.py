import torch
import torchaudio
import soundfile as sf
import os
from asteroid import ConvTasNet
from utils import ensure_folder_exists

# Folder setup for audio files
audio_folder = "recorded_audio"
separated_audio_folder = "separated_audio"
ensure_folder_exists(audio_folder)
ensure_folder_exists(separated_audio_folder)

# Define a function to load the pre-trained Conv-TasNet model from Asteroid
def load_pretrained_conv_tasnet_model():
    # Load the pre-trained model from Asteroid
    model = ConvTasNet.from_pretrained("mpariente/Conv-TasNet-WSJ0-2mix")
    model.eval()  # Set the model to evaluation mode
    return model

# Define a function to separate speakers using Conv-TasNet
def separate_speakers_conv_tasnet(input_path, output_folder):
    # Load the pre-trained Conv-TasNet model
    model = load_pretrained_conv_tasnet_model()

    # Load the audio file
    waveform, sample_rate = torchaudio.load(input_path)

    # Resample the audio to the model's expected sample rate (16 kHz)
    resample_rate = 16000
    if sample_rate != resample_rate:
        waveform = torchaudio.transforms.Resample(sample_rate, resample_rate)(waveform)

    # Apply Conv-TasNet to separate the sources
    with torch.no_grad():
        separated_sources = model.separate(waveform)

    # Save each separated source as a separate file
    for i, source in enumerate(separated_sources):
        # Convert the separated waveform back to numpy format
        output_path = os.path.join(output_folder, f"separated_speaker_{i + 1}.wav")
        sf.write(output_path, source.numpy().T, resample_rate)
        print(f"Saved separated speaker {i + 1} to {output_path}")

# Example Usage:

input_audio_path = "path_to_input_audio.wav"  # Specify the path to your recorded audio
output_folder = separated_audio_folder  # Folder where the separated audio will be saved

# Perform speaker separation
try:
    separate_speakers_conv_tasnet(input_audio_path, output_folder)
except Exception as e:
    print(f"Error during speaker separation: {e}")
