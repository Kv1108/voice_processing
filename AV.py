import cv2
import pyaudio
import wave
import os
import datetime
import threading
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Constants
VIDEO_FOLDER = "recorded_videos"
AUDIO_FOLDER = "recorded_audio"
CLIP_DURATION = 30  # seconds
FRAME_RATE = 20
AUDIO_RATE = 44100
CHUNK = 1024
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1

# Ensure necessary folders exist
def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

ensure_folder_exists(VIDEO_FOLDER)
ensure_folder_exists(AUDIO_FOLDER)

# Flag for stopping recording
stop_event = threading.Event()

def record_video():
    logging.info("Video recording started.")
    cap = cv2.VideoCapture(0)
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    while not stop_event.is_set():
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        video_path = os.path.join(VIDEO_FOLDER, f"{timestamp}.mp4")
        out = cv2.VideoWriter(video_path, fourcc, FRAME_RATE, frame_size)
        logging.info(f"Video segment recording: {video_path}")

        start_time = datetime.datetime.now()
        while (datetime.datetime.now() - start_time).seconds < CLIP_DURATION:
            ret, frame = cap.read()
            if not ret:
                logging.error("Error: Unable to read video frame.")
                break
            out.write(frame)
            cv2.imshow("Recording Video", frame)

            if cv2.waitKey(1) == ord("q") or stop_event.is_set():
                stop_event.set()
                break

        out.release()
        logging.info(f"Video segment saved: {video_path}")

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Video recording stopped.")

def record_audio():
    logging.info("Audio recording started.")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True, frames_per_buffer=CHUNK)

    while not stop_event.is_set():
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        audio_path = os.path.join(AUDIO_FOLDER, f"{timestamp}.wav")
        logging.info(f"Audio segment recording: {audio_path}")

        frames = []
        start_time = datetime.datetime.now()
        while (datetime.datetime.now() - start_time).seconds < CLIP_DURATION:
            if stop_event.is_set():
                break
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        # Save audio to file
        with wave.open(audio_path, "wb") as sound_file:
            sound_file.setnchannels(AUDIO_CHANNELS)
            sound_file.setsampwidth(audio.get_sample_size(AUDIO_FORMAT))
            sound_file.setframerate(AUDIO_RATE)
            sound_file.writeframes(b"".join(frames))
        logging.info(f"Audio segment saved: {audio_path}")

    stream.stop_stream()
    stream.close()
    audio.terminate()
    logging.info("Audio recording stopped.")

def start_av_recording():
    try:
        video_thread = threading.Thread(target=record_video)
        audio_thread = threading.Thread(target=record_audio)

        video_thread.start()
        audio_thread.start()

        logging.info("Press 'q' to stop recording.")

        while not stop_event.is_set():
            if cv2.waitKey(1) == ord("q"):
                stop_event.set()

        video_thread.join()
        audio_thread.join()
    except KeyboardInterrupt:
        logging.warning("Recording interrupted manually.")
        stop_event.set()
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    logging.info("Starting Real-Time Audio-Video Recording...")
    start_av_recording()
    logging.info("Recording session complete.")
