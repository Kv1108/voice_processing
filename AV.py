import cv2
import threading
import pyaudio
import wave
import datetime

# Video recording function
def record_video(video_filename):
    cap = cv2.VideoCapture(0)
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_filename, fourcc, 20, frame_size)

    while True:
        _, frame = cap.read()
        out.write(frame)
        cv2.imshow("Video Recording", frame)
        
        if cv2.waitKey(1) == ord('q'):  # Stop on 'q' key
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Audio recording function
def record_audio(audio_filename):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    frames = []

    while True:
        data = stream.read(1024)
        frames.append(data)
        
        # For demonstration, stop after 10 seconds
        if len(frames) > 10:  
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(audio_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))

# Main function to start both recordings with the same timestamped filename
def main():
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    video_filename = f"{timestamp}.mp4"
    audio_filename = f"{timestamp}.wav"

    # Start video and audio recording in separate threads
    video_thread = threading.Thread(target=record_video, args=(video_filename,))
    audio_thread = threading.Thread(target=record_audio, args=(audio_filename,))
    
    video_thread.start()
    audio_thread.start()

    # Wait for both threads to finish
    video_thread.join()
    audio_thread.join()

    print("Recording completed.")

if __name__ == "__main__":
    main()
