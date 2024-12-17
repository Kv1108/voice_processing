import os
import librosa
import speech_recognition as sr
from utils import ensure_folder_exists

# Global Variables
SPEAKER_AUDIO_FOLDER = "speaker_audio"         # Input folder (from Stage 3)
TRANSCRIPTIONS_FOLDER = "transcriptions"       # Output folder for transcription results

def transcribe_with_timestamps(audio_file, recognizer):
    """
    Transcribes audio with timestamps using Google Speech Recognition.
    Args:
        audio_file (str): Path to audio file.
        recognizer: SpeechRecognition Recognizer instance.
    Returns:
        list: List of dictionaries containing 'start', 'end', and 'text' for each segment.
    """
    try:
        audio_data = librosa.load(audio_file, sr=None)
        sr_rate = audio_data[1]
        duration = librosa.get_duration(y=audio_data[0], sr=sr_rate)
        
        chunk_length = 10  # Length of each audio chunk in seconds
        segments = []
        recognizer_instance = sr.Recognizer()

        for start in range(0, int(duration), chunk_length):
            end = min(start + chunk_length, int(duration))
            with sr.AudioFile(audio_file) as source:
                audio = recognizer_instance.record(source, duration=chunk_length, offset=start)
            
            try:
                text = recognizer_instance.recognize_google(audio)
                segments.append({"start": start, "end": end, "text": text})
                print(f"Transcribed Segment [{start}-{end} sec]: {text}")
            except sr.UnknownValueError:
                print(f"Unable to transcribe audio from {start} to {end} seconds.")
            except sr.RequestError as e:
                print(f"Error with transcription service: {e}")
        
        return segments

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return []

def save_transcription(transcription_data, speaker_name, output_folder):
    """
    Save transcription results to a file with timestamps.
    Args:
        transcription_data (list): List of transcription segments with timestamps.
        speaker_name (str): Name of the speaker (e.g., speaker_1).
        output_folder (str): Folder to save the transcription file.
    """
    ensure_folder_exists(output_folder)
    output_file = os.path.join(output_folder, f"{speaker_name}_transcription.txt")

    with open(output_file, "w") as file:
        file.write("Speaker Transcription with Timestamps\n")
        file.write("=" * 40 + "\n")
        for segment in transcription_data:
            start_time = f"{segment['start']}s"
            end_time = f"{segment['end']}s"
            text = segment['text']
            file.write(f"[{start_time} - {end_time}] {text}\n")

    print(f"Transcription saved: {output_file}")

def process_transcription(input_folder, output_folder):
    """
    Process all speaker-separated audio files for transcription.
    Args:
        input_folder (str): Path to speaker-separated audio folder.
        output_folder (str): Path to save transcriptions.
    """
    recognizer = sr.Recognizer()
    ensure_folder_exists(output_folder)

    for audio_file in os.listdir(input_folder):
        if audio_file.endswith(".wav"):
            speaker_name = os.path.splitext(audio_file)[0]
            audio_path = os.path.join(input_folder, audio_file)
            print(f"Processing Transcription for: {speaker_name}")

            # Perform transcription with timestamps
            transcription_data = transcribe_with_timestamps(audio_path, recognizer)

            # Save transcription
            if transcription_data:
                save_transcription(transcription_data, speaker_name, output_folder)
            else:
                print(f"No transcribable data found for {speaker_name}.")

if __name__ == "__main__":
    print("Starting Transcription Stage...")
    process_transcription(SPEAKER_AUDIO_FOLDER, TRANSCRIPTIONS_FOLDER)
