import pyaudio
import wave
import threading

# Audio recording function
def record_audio(filename):
    print("Starting audio recording...")

    audio = pyaudio.PyAudio()

    # Open the audio stream
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    frames = []

    try:
        print("Recording... Press Ctrl+C to stop.")
        while True:
            # Read data from the stream
            data = stream.read(1024)
            frames.append(data)

    except KeyboardInterrupt:
        print("Recording stopped by user.")
    except Exception as e:
        print(f"Error while recording audio: {e}")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the audio to a file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))

    print(f"Audio recording saved as {filename}")

# Function to start the audio recording in a thread
def start_audio_recording_thread(filename):
    audio_thread = threading.Thread(target=record_audio, args=(filename,))
    audio_thread.start()
    return audio_thread

# Main function to run the audio recording process
if __name__ == "__main__":
    # Define the filename with timestamp
    filename = "myrecording.wav"

    # Start audio recording in a separate thread
    audio_thread = start_audio_recording_thread(filename)

    # Join the thread to wait for the audio recording to finish
    audio_thread.join()

    print("Audio recording finished.")
