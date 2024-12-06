from google.cloud import speech
import os

# Initialize the Speech Client
client = speech.SpeechClient()

# Provide the path to your audio file
audio_file_path = 'path/to/your/audio/file.wav'

with open(audio_file_path, 'rb') as audio_file:
    content = audio_file.read()

audio = speech.RecognitionAudio(content=content)

# Configure the request (set language code to 'en-US' for English)
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="en-US",
)

# Send the request to the Speech-to-Text API
response = client.recognize(config=config, audio=audio)

# Print the transcription results
for result in response.results:
    print("Transcript: {}".format(result.alternatives[0].transcript))
