from flask import Flask, request, jsonify
from google.cloud import speech
import os
from pydub import AudioSegment
import io

# Initialize Flask app
app = Flask(__name__)

# Initialize Google Cloud Speech client
client = speech.SpeechClient()

# Route to handle file upload and transcription
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Get the uploaded audio file
    file = request.files['audio']
    audio_data = file.read()

    # Convert audio to required format (if necessary)
    # Here we assume the audio is in wav format, if it's in a different format
    # you can use libraries like pydub to convert to WAV before sending to API
    audio = AudioSegment.from_file(io.BytesIO(audio_data))
    audio = audio.set_channels(1).set_frame_rate(16000)  # Set mono channel and 16kHz sample rate
    wav_audio = io.BytesIO()
    audio.export(wav_audio, format='wav')
    wav_audio.seek(0)

    # Prepare the audio file for transcription
    audio = speech.RecognitionAudio(content=wav_audio.read())

    # Configure the recognizer
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    # Send the request for transcription
    response = client.recognize(config=config, audio=audio)

    # Get the transcription result
    transcription = ""
    for result in response.results:
        transcription += result.alternatives[0].transcript

    return jsonify({'transcription': transcription})

if __name__ == '__main__':
    app.run(debug=True)
