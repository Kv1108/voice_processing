import torch
from speechbrain.pretrained import SpeakerRecognition

def compare_speakers_advanced(audio_path1, audio_path2):
    """
    Compare two audio files to check if they belong to the same speaker using SpeechBrain.
    
    :param audio_path1: Path to the first audio file
    :param audio_path2: Path to the second audio file
    :return: Similarity score (higher means more likely the same person)
    """
    # Load pre-trained speaker recognition model
    model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")
    
    # Compute similarity score between the two audio files
    score, prediction = model.verify_files(audio_path1, audio_path2)
    
    return score.item(), prediction  # Convert to scalar and return

# Example usage
audio_file1 = "SPEAKER_05.wav"  # Replace with the first audio file path
audio_file2 = "SPEAKER_06.wav"  # Replace with the second audio file path

similarity_score, is_same_speaker = compare_speakers_advanced(audio_file1, audio_file2)

if is_same_speaker:
    print(f"The voices are from the same person (score: {similarity_score:.4f}).")
else:
    print(f"The voices are from different people (score: {similarity_score:.4f}).")
