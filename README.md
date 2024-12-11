# Voice Processing Project - README

## Introduction
This project processes audio files to identify the number of speakers, cluster audio segments by speaker, and generate a transcription file distinguishing speech by speakers. It integrates audio processing, speaker clustering, and transcription functionalities in a streamlined pipeline.

---

## Features
- Automatic detection of speakers in an audio file.
- Clustering of audio segments using K-means clustering.
- Transcription of clustered audio segments per speaker.
- Automated workflow with modular scripts.

---

## File Structure
```
voice_processing/
├── main.py                # Orchestrates the entire workflow
├── cluster.py             # Detects and clusters speakers in the audio file
├── separate2.py           # Transcribes clustered audio segments by speaker
├── recorded_audio/        # Folder containing recorded audio files
├── transcription.txt      # Output file containing speaker-wise transcription
├── cluster_results.json   # Stores clustering data for speaker segments
```

---

## Requirements
- Python 3.12.2 or higher
- Required libraries:
  - `librosa`
  - `speech_recognition`
  - `scikit-learn`
  - `numpy`
  - `json`

---

## Setup Instructions
1. Clone the repository to your local machine:
   ```bash
   git clone <repository_url>
   cd voice_processing
   ```

2. Ensure you have Python 3.12.2 installed.

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your audio files in the `recorded_audio/` folder.

---

## How to Run
1. Run the main script to execute the entire workflow:
   ```bash
   python main.py
   ```

2. The output transcription will be saved in `transcription.txt`.

---

## Expected Output
The transcription file will look like this:
```
Speaker 1: Hello, how are you?
Speaker 2: I'm fine, thank you.
Speaker 3: Let's start the meeting.
```

---

## Known Issues
- Accuracy may vary depending on audio quality and speaker overlap.
- Ensure clear and noise-free audio for best results.

---

## Contributions
Contributions are welcome! Please fork the repository and submit a pull request.

---

## License
This project is licensed under the MIT License.

---

&copy; 2024 Voice Processing Team. All rights reserved.
