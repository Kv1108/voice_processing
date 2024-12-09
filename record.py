import pyaudio
import wave
from preprocess_audio import preprocess_audio  # Import the preprocessing module

# Function to record audio
def record_audio(output_path="myrecording.wav", duration=None):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    frames = []
    print("Recording... Press Ctrl+C to stop.")

    try:
        if duration:
            for _ in range(0, int(44100 / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)
        else:
            while True:
                data = stream.read(1024)
                frames.append(data)
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    # Save the raw recording
    raw_audio_file = output_path
    sound_file = wave.open(raw_audio_file, "wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    sound_file.setframerate(44100)
    sound_file.writeframes(b''.join(frames))
    sound_file.close()

    print(f"Audio saved to '{raw_audio_file}'")
    return raw_audio_file


if __name__ == "__main__":
    # Step 1: Record audio
    raw_audio_file = record_audio(output_path="myrecording.wav")

    # Step 2: Preprocess the recorded audio
    preprocessed_audio_file = "preprocessed_audio.wav"
    print("Preprocessing the audio for better quality...")
    preprocess_audio(raw_audio_file, preprocessed_audio_file)

    print(f"Preprocessed audio saved to '{preprocessed_audio_file}'")
