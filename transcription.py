import speech_recognition as sr
import os
import sys

def ensure_folder_exists(folder_name):
    """
    Ensure that the specified folder exists; create it if it doesn't.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")

def transcribe_audio(audio_path):
    """
    Transcribes audio from a given file path using Google Speech Recognition.
    """
    recognizer = sr.Recognizer()

    # Load the audio file
    try:
        with sr.AudioFile(audio_path) as source:
            print("Processing audio file...")
            audio = recognizer.record(source)  # Read the entire audio file
    except FileNotFoundError:
        return f"Error: File '{audio_path}' not found."
    except Exception as e:
        return f"Error processing the audio file: {e}"

    try:
        print("Transcribing audio...")
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sr.UnknownValueError:
        return "Error: Unable to understand the audio."
    except sr.RequestError as e:
        return f"Error: Could not request results from the Google Speech Recognition service; {e}"

def save_transcription_to_file(audio_path, transcription, output_folder):
    """
    Saves the transcription to a text file in the specified output folder.
    """
    # Generate the transcription file path
    base_name = os.path.basename(audio_path)  # Extract file name from the path
    transcription_file_path = os.path.join(output_folder, f"{os.path.splitext(base_name)[0]}_transcription.txt")

    # Write the transcription to a text file
    try:
        with open(transcription_file_path, "w") as file:
            file.write(transcription)
        print(f"Transcription saved to '{transcription_file_path}'")
    except Exception as e:
        print(f"Error saving transcription: {e}")

def main():
    # Check if the audio file path is provided via command-line argument
    if len(sys.argv) != 2:
        print("Usage: python transcription.py <audio_file_path>")
        sys.exit(1)

    audio_file_path = sys.argv[1]

    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file '{audio_file_path}' does not exist.")
        sys.exit(1)

    # Create the transcriptions folder if it doesn't exist
    transcription_folder = "transcriptions"
    ensure_folder_exists(transcription_folder)

    # Transcribe the audio
    result = transcribe_audio(audio_file_path)
    print("\nTranscription:")
    print(result)

    # Save the transcription to the folder
    save_transcription_to_file(audio_file_path, result, transcription_folder)

if __name__ == "__main__":
    main()
