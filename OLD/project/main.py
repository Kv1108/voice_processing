import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import threading

from vad import run_vad  # Function from vad.py to handle voice activity detection
from transcribe import transcribe_audio  # Function from transcribe.py to handle Whisper transcription
from diarize import diarize_audio  # Function from diarize.py to handle speaker diarization
from database import store_transcription  # Function from database.py to store transcription in MySQL

# Global Variables
AUDIO_QUEUE = queue.Queue()  # Queue to store live audio chunks
IS_RECORDING = False  # Flag to control live audio recording

# Streamlit UI Setup
st.title("🎙️ Real-Time Speaker Identification and Transcription")
st.sidebar.header("Controls")

# Record Button
if st.sidebar.button("Start Recording"):
    IS_RECORDING = True
    st.session_state["recording"] = True
    st.sidebar.write("Recording in progress...")

# Stop Button
if st.sidebar.button("Stop Recording"):
    IS_RECORDING = False
    st.session_state["recording"] = False
    st.sidebar.write("Recording stopped.")

# Audio Stream Callback (captures real-time audio and adds it to the queue)
def audio_callback(indata, frames, time, status):
    if IS_RECORDING:
        AUDIO_QUEUE.put(indata.copy())  # Copy audio data to avoid data overwrite

# Function to Start Audio Stream
def start_audio_stream():
    stream = sd.InputStream(
        samplerate=16000, 
        channels=1, 
        dtype='float32', 
        callback=audio_callback
    )
    with stream:
        while IS_RECORDING:
            pass  # Keep the stream alive

# Function to Process Audio in the Queue (runs VAD, transcription, and diarization)
def process_audio():
    st.markdown("### Transcription Output")
    
    while True:
        if not AUDIO_QUEUE.empty():
            audio_chunk = AUDIO_QUEUE.get()  # Get the latest audio chunk
            
            # 1. Voice Activity Detection (VAD)
            vad_audio = run_vad(audio_chunk)  # Pass the audio chunk to VAD logic
            if vad_audio is None:
                continue  # If no voice detected, continue
            
            # 2. Transcription
            transcription = transcribe_audio(vad_audio)  # Transcribe using Whisper
            
            # 3. Speaker Diarization
            diarized_output = diarize_audio(vad_audio)  # Get speaker labels using pyannote-audio
            
            # 4. Display the transcription with speaker information
            for segment in diarized_output:
                speaker_label = segment['speaker']
                text = segment['text']
                start_time = segment['start_time']
                end_time = segment['end_time']
                
                st.write(f"**[{speaker_label}]** ({start_time} - {end_time}): {text}")
                
                # 5. Store the transcription to the database
                store_transcription(speaker_label, text, start_time, end_time)

# Thread for Real-Time Audio Processing
if "recording" in st.session_state and st.session_state["recording"]:
    audio_thread = threading.Thread(target=start_audio_stream)
    process_thread = threading.Thread(target=process_audio)
    
    if not audio_thread.is_alive():
        audio_thread.start()
    if not process_thread.is_alive():
        process_thread.start()
