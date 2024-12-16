import torch
import silero_vad

# Load Silero VAD model
model, utils = silero_vad.get_speech_timestamps, silero_vad.prepare_model
vad_model = utils(model_name='silero_vad', device='cpu')  # Using CPU for now


def run_vad(audio_chunk):
    """
    Run voice activity detection (VAD) on an audio chunk.
    
    Args:
        audio_chunk (numpy.ndarray): The audio data chunk to process.
        
    Returns:
        numpy.ndarray or None: Returns the audio chunk if speech is detected, otherwise None.
    """
    
    try:
        # Convert audio to the required format (PyTorch tensor)
        audio_tensor = torch.from_numpy(audio_chunk).float().squeeze()  # Remove extra dimensions if any
        
        # Ensure the audio tensor has the correct shape
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.mean(dim=1)  # Convert to mono by averaging channels
        
        # Normalize audio (if needed, depends on Silero model requirements)
        if audio_tensor.max() > 1.0:
            audio_tensor = audio_tensor / 32768.0  # Normalization for 16-bit PCM audio
        
        # Run Silero VAD to detect speech timestamps
        speech_timestamps = model(audio_tensor, vad_model, sampling_rate=16000)
        
        if speech_timestamps:
            start = speech_timestamps[0]['start']  # Start of detected speech in samples
            end = speech_timestamps[0]['end']  # End of detected speech in samples
            
            # Extract the audio chunk corresponding to the speech
            speech_audio = audio_chunk[start:end]
            return speech_audio
        else:
            return None  # No speech detected
    except Exception as e:
        print(f"Error running VAD: {e}")
        return None


if __name__ == "__main__":
    import numpy as np
    
    # Example usage with dummy audio data
    dummy_audio = np.random.rand(16000 * 3).astype(np.float32)  # 3 seconds of random audio data
    
    speech_audio = run_vad(dummy_audio)
    if speech_audio is not None:
        print("Speech detected in the audio chunk.")
    else:
        print("No speech detected.")
