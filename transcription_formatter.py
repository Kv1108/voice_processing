import os

def format_transcription(speaker_segments, audio_path, output_folder):
    """
    Generate a conversation-style transcription based on speaker segments.
    
    Parameters:
        speaker_segments (list): List of (speaker_id, start, end, content) tuples.
            - speaker_id (int): ID of the speaker.
            - start (float): Start time of the segment.
            - end (float): End time of the segment.
            - content (str): Transcribed text of the segment.
        audio_path (str): Path to the input audio file.
        output_folder (str): Folder to save the formatted transcription.
    """
    # Sort segments by start time to ensure chronological order
    speaker_segments = sorted(speaker_segments, key=lambda x: x[1])

    # Create conversation-style transcription
    conversation_text = []
    for speaker_id, start, end, content in speaker_segments:
        speaker_label = f"Person {speaker_id}"
        conversation_text.append(f"{speaker_label}: {content}")

    # Save to file
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, os.path.basename(audio_path).replace(".wav", "_conversation.txt"))
    with open(output_file, "w") as f:
        f.write("\n".join(conversation_text))
    print(f"Conversation-style transcription saved to {output_file}")

# Example Usage
if __name__ == "__main__":
    # Sample speaker_segments
    speaker_segments = [
        (1, 0.00, 2.50, "Hello, how are you?"),
        (2, 3.00, 4.50, "I'm good, thank you."),
        (1, 5.00, 7.00, "That's great to hear!"),
    ]
    audio_file = "example.wav"
    output_dir = "formatted_transcriptions"

    format_transcription(speaker_segments, audio_file, output_dir)
