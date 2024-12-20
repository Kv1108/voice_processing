import os
from pyannote.audio import Model, Inference
from pyannote.core import Segment
import librosa
import soundfile as sf

def set_huggingface_token():
    """Set the Hugging Face token in the environment."""
    token = "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"  # Your Hugging Face token (do not share this)
    if not token:
        raise RuntimeError("Hugging Face token is missing. Please set the HUGGING_FACE_TOKEN environment variable.")
    os.environ['HUGGING_FACE_TOKEN'] = token
    print(f"Hugging Face token set: {token[:5]}... (truncated for security)")

def load_segmentation_model():
    """Load the pyannote segmentation model using the authenticated token."""
    try:
        token = os.getenv('HUGGING_FACE_TOKEN')
        if not token:
            raise RuntimeError("Hugging Face token is missing. Please set the HUGGING_FACE_TOKEN environment variable.")
        
        print(f"Loading segmentation model with token: {token[:5]}... (truncated for security)")
        
        # Load the segmentation model
        model = Model.from_pretrained("pyannote/segmentation", use_auth_token=token)
        inference = Inference(model)
        print("Segmentation model loaded successfully.")
        return inference
    except Exception as e:
        print(f"Error loading segmentation model: {str(e)}")
        raise RuntimeError("Failed to load pyannote segmentation model. Check your Hugging Face token, model access, and network connectivity.") from e

def diarize_audio(inference, audio_file):
    """Diarize the audio file using the pyannote segmentation model."""
    try:
        print(f"Processing file: {audio_file}")
        
        # Perform speaker diarization on the entire file
        diarization_result = inference(audio_file)
        
        clustered_segments = {}
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            if speaker not in clustered_segments:
                clustered_segments[speaker] = []
            clustered_segments[speaker].append((turn.start, turn.end))
        
        print("Diarization result:", clustered_segments)
        return clustered_segments

    except Exception as e:
        print(f"Error during diarization: {str(e)}")
        raise RuntimeError("Error during diarization. Check your file path, segmentation model, and audio format.") from e

def preprocess_audio(input_audio_path, output_audio_path):
    """Preprocess audio before running the diarization."""
    try:
        print(f"Preprocessing audio: {input_audio_path}")
        # Load the audio file
        audio, sr = librosa.load(input_audio_path, sr=16000)  # Resample to 16kHz
        # Save the processed file to a new location
        sf.write(output_audio_path, audio, sr)
        print(f"Audio preprocessed and saved at: {output_audio_path}")
    except Exception as e:
        print(f"Error in preprocessing audio: {str(e)}")
        raise RuntimeError(f"Error in preprocessing audio file: {input_audio_path}") from e

def main():
    """Main function to run the diarization process."""
    try:
        # Step 1: Set the Hugging Face token
        set_huggingface_token()
        
        # Step 2: Load the pyannote segmentation model
        inference = load_segmentation_model()
        
        # Step 3: Specify the path to the audio file
        audio_directory = "recorded_audio"
        processed_audio_directory = "processed_audio"
        
        # Automatically select the most recent file in the recorded_audio folder
        audio_files = sorted([f for f in os.listdir(audio_directory) if f.endswith(".wav")], reverse=True)
        if not audio_files:
            raise RuntimeError(f"No audio files found in {audio_directory} directory.")
        
        latest_audio_file = os.path.join(audio_directory, audio_files[0])
        print(f"Processing latest audio file: {latest_audio_file}")
        
        # Step 4: Preprocess the audio (downsampling to 16kHz)
        processed_audio_file = os.path.join(processed_audio_directory, "processed_audio.wav")
        preprocess_audio(latest_audio_file, processed_audio_file)
        
        # Step 5: Run speaker diarization on the processed audio file
        clustered_segments = diarize_audio(inference, processed_audio_file)
        print("Clustered segments:", clustered_segments)
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
