import pyaudio
import wave
import datetime
import os
import time
import keyboard  # For listening to key presses
from utils import ensure_folder_exists
import audioop  # For detecting silence

# Detection variables
recording = False
recording_stopped_time = None
SILENCE_THRESHOLD = 500  # Adjust based on environment (lower = more sensitive to sound)
SILENCE_TIMEOUT = 2  # Stop recording after 2 seconds of silence

# Folder setup for audio files
audio_folder = "recorded_audio"
ensure_folder_exists(audio_folder)

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

frames = []
print("Audio recording ready... Press 'q' to stop manually.")

try:
    while True:
        # Read audio input
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

        # Check for silence
        rms = audioop.rms(data, 2)  # Calculate root mean square (RMS) value
        is_silence = rms < SILENCE_THRESHOLD

        if not is_silence:  # Audio detected
            if not recording:
                # Start recording
                recording = True
                timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                audio_path = os.path.join(audio_folder, f"{timestamp}.wav")
                print(f"Started Recording: {audio_path}")
        else:  # Silence detected
            if recording:
                if recording_stopped_time is None:
                    # Silence detected for the first time, start the timer
                    recording_stopped_time = time.time()
                elif time.time() - recording_stopped_time >= SILENCE_TIMEOUT:
                    # Silence has persisted for longer than the timeout
                    # Stop recording and save the audio
                    recording = False
                    recording_stopped_time = None

                    sound_file = wave.open(audio_path, "wb")
                    sound_file.setnchannels(1)
                    sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                    sound_file.setframerate(44100)
                    sound_file.writeframes(b"".join(frames))
                    sound_file.close()
                    print(f"Stopped Recording! Saved to: {audio_path}")

                    # Clear frames for the next recording
                    frames = []
            else:
                recording_stopped_time = None  # Reset if not currently recording

        # Stop recording manually by pressing 'q'
        if keyboard.is_pressed('q'):
            print("Manual stop detected. Finalizing...")
            if recording:
                # Save the current recording if active
                sound_file = wave.open(audio_path, "wb")
                sound_file.setnchannels(1)
                sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                sound_file.setframerate(44100)
                sound_file.writeframes(b"".join(frames))
                sound_file.close()
                print(f"Stopped Recording! Saved to: {audio_path}")
            break

except KeyboardInterrupt:
    print("Recording stopped manually via Ctrl+C.")

# Cleanup
stream.stop_stream()
stream.close()
audio.terminate()
