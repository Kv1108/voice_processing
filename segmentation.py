import os
import numpy as np
from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F

def ensure_folder_exists(folder_path):
    """Ensure a folder exists, and create it if it doesn't."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def fix_input_size(input_tensor, kernel_size):
    """
    Adjust the input size by padding it to ensure it fits the kernel size for pooling.
    
    Parameters:
        input_tensor (torch.Tensor): The input tensor to be padded.
        kernel_size (int): The size of the kernel for pooling.
    
    Returns:
        torch.Tensor: The padded input tensor.
    """
    # Ensure the input tensor is 2D
    if input_tensor.ndimension() == 1:
        input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension
    
    # Calculate the required padding to ensure output size is valid after pooling
    if input_tensor.size(1) < kernel_size:
        padding_size = kernel_size - input_tensor.size(1)
        left_padding = padding_size // 2
        right_padding = padding_size - left_padding
        input_tensor = F.pad(input_tensor, (left_padding, right_padding))  # Pad the input
    return input_tensor

def extract_segments_and_embeddings(audio_path, auth_token):
    """
    Extract speech segments and corresponding embeddings from audio.
    
    Parameters:
        audio_path (str): Path to the input audio file.
        auth_token (str): Hugging Face authentication token.
    
    Returns:
        embeddings (np.ndarray): Array of speaker embeddings.
        segments (list): List of (start, end) tuples for speech segments.
    """
    segmentation_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=auth_token)
    embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=auth_token)
    inference = Inference(embedding_model, window="whole")

    diarization_result = segmentation_pipeline(audio_path)
    embeddings = []
    segments = []

    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        # Crop the audio to the segment and generate embedding
        audio_embedding = inference.crop(audio_path, Segment(turn.start, turn.end))
        
        # Convert to tensor and ensure proper size
        audio_embedding_tensor = torch.tensor(audio_embedding)
        kernel_size = 2  # Set kernel size to a smaller value
        audio_embedding_tensor = fix_input_size(audio_embedding_tensor, kernel_size)
        
        embeddings.append(audio_embedding_tensor.numpy())
        segments.append((turn.start, turn.end))

    return np.array(embeddings), segments

def calculate_distances(embeddings):
    """
    Calculate cosine distances between all pairs of embeddings.
    
    Parameters:
        embeddings (np.ndarray): Array of embeddings.
    
    Returns:
        distances (np.ndarray): Pairwise cosine distances.
    """
    distances = cdist(embeddings, embeddings, metric="cosine")
    return distances

def get_latest_file(directory, suffix="_preprocessed.wav"):
    """
    Get the latest file in a directory with a given suffix.
    
    Parameters:
        directory (str): Path to the directory where the files are stored.
        suffix (str): Suffix of the file to search for.
    
    Returns:
        str: Path to the latest file with the given suffix.
    """
    # List all files in the directory that end with the specified suffix
    files = [f for f in os.listdir(directory) if f.endswith(suffix)]
    
    if not files:
        raise ValueError(f"No files with the suffix '{suffix}' found in directory '{directory}'")

    # Get the full file paths and sort them by modification time (most recent first)
    full_paths = [os.path.join(directory, f) for f in files]
    latest_file = max(full_paths, key=os.path.getmtime)
    
    return latest_file

# Example usage
audio_folder = "recorded_audio"
auth_token = "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"

# Ensure the audio folder exists
ensure_folder_exists(audio_folder)

# Get the latest file with the '_preprocessed.wav' suffix
audio_path = get_latest_file(audio_folder, suffix="_preprocessed.wav")

# If the file exists, proceed to extract embeddings and segments
embeddings, segments = extract_segments_and_embeddings(audio_path, auth_token)

# Calculate pairwise distances
distances = calculate_distances(embeddings)
print("Pairwise cosine distances:")
print(distances)
