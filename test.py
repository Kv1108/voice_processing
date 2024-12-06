import speech_recognition as sr

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()

    # Load the audio file
    try:
        with sr.AudioFile(audio_path) as source:
            print("Processing audio file...")
            audio = recognizer.record(source)  # Read the entire audio file
    except FileNotFoundError:
        return f"Error: File '{audio_path}' not found."

    try:
        print("Transcribing audio...")
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sr.UnknownValueError:
        return "Error: Unable to understand the audio."
    except sr.RequestError as e:
        return f"Error: Could not request results from the Google Speech Recognition service; {e}"

def save_transcription_to_file(audio_path, transcription):
    # Generate file path for transcription
    transcription_file_path = f"{audio_path}_speakers_transcription.txt"
    
    # Write the transcription to a text file
    try:
        with open(transcription_file_path, "w") as file:
            file.write(transcription)
        print(f"Transcription saved as '{transcription_file_path}'")
    except Exception as e:
        print(f"Error saving transcription: {e}")

if __name__ == "__main__":
    # Path to your specific audio file
    audio_file_path = f"{timestamp}.wav"  

    # Call the transcription function
    result = transcribe_audio(audio_file_path)
    print("\nTranscription:")
    print(result)

    # Save the transcription to a text file
    save_transcription_to_file(audio_file_path, result)
