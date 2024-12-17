# transcribe.py
import whisper
import os

# Load the Whisper model
model = whisper.load_model("base")

def transcribe_audio(audio_path):
    """
    This function takes the path to an audio file, transcribes it using the Whisper model,
    and returns the transcribed text.

    Parameters:
        audio_path (str): Path to the audio file to be transcribed.

    Returns:
        str: Transcribed text from the audio.
    """
    try:
        # Check if the file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"The audio file at {audio_path} was not found.")
        
        # Transcribe the audio
        result = model.transcribe(audio_path)
        
        # Get the transcribed text
        transcribed_text = result['text']
        
        # Return the transcribed text
        return transcribed_text
    
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return None
