import os
import threading
import soundfile as sf
import librosa
import speech_recognition as sr
import noisereduce as nr

def ensure_folder_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")

def reduce_noise(input_audio_path, output_audio_path):
    signal, sr = librosa.load(input_audio_path, sr=None)
    reduced_signal = nr.reduce_noise(y=signal, sr=sr, prop_decrease=0.8)
    sf.write(output_audio_path, reduced_signal, sr)
    print(f"Noise-reduced audio saved at: {output_audio_path}")

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            print(f"Processing audio file: {audio_path}")
            audio = recognizer.record(source)
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

def save_transcription(audio_path, transcription, speaker_label, output_folder):
    ensure_folder_exists(output_folder)
    base_name = os.path.basename(audio_path)
    transcription_file = os.path.join(output_folder, f"{os.path.splitext(base_name)[0]}_transcription.txt")
    try:
        with open(transcription_file, "a") as file:
            file.write(f"{speaker_label}: {transcription}\n")
        print(f"Transcription saved to '{transcription_file}'")
        return transcription_file
    except Exception as e:
        print(f"Error saving transcription: {e}")
        return None

def process_and_transcribe_cluster(cluster_audio_path, speaker_label, output_folder):
    reduced_audio_path = cluster_audio_path.replace(".wav", "_noise_reduced.wav")
    reduce_noise(cluster_audio_path, reduced_audio_path)
    transcription = transcribe_audio(reduced_audio_path)
    save_transcription(reduced_audio_path, transcription, speaker_label, output_folder)

def process_clusters_in_parallel(clusters_folder, output_folder):
    ensure_folder_exists(output_folder)
    cluster_files = [f for f in os.listdir(clusters_folder) if f.endswith(".wav")]

    threads = []
    for idx, cluster_file in enumerate(cluster_files):
        speaker_label = f"Person {idx + 1}"
        cluster_audio_path = os.path.join(clusters_folder, cluster_file)
        thread = threading.Thread(target=process_and_transcribe_cluster, args=(cluster_audio_path, speaker_label, output_folder))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("All clusters processed and transcribed.")

if __name__ == "__main__":
    clusters_folder = "clusters"
    transcription_output_folder = "transcriptions"

    ensure_folder_exists(clusters_folder)
    ensure_folder_exists(transcription_output_folder)

    print("Starting transcription for clustered audio files...")
    process_clusters_in_parallel(clusters_folder, transcription_output_folder)
    print("Transcription completed.")
