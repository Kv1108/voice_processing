from scipy.spatial.distance import cdist
import numpy as np

# Global speaker memory
speaker_memory = []
speaker_labels = []

# def assign_consistent_ids(embeddings, threshold=0.3):
#     """
#     Assign consistent speaker IDs by comparing with known speaker memory.
    
#     Parameters:
#         embeddings (np.ndarray): Array of speaker embeddings.
#         threshold (float): Cosine distance threshold for matching speakers.

#     Returns:
#         assigned_labels (list): List of consistent speaker IDs.
#     """
#     global speaker_memory, speaker_labels

#     if not speaker_memory:
#         # Initialize memory with the first embeddings
#         speaker_memory = embeddings
#         speaker_labels = list(range(len(embeddings)))
#         return speaker_labels

#     distances = cdist(embeddings, speaker_memory, metric="cosine")
#     assigned_labels = []

#     for i, dist in enumerate(distances):
#         if dist.min() < threshold:
#             # Match with the closest known speaker
#             assigned_labels.append(speaker_labels[np.argmin(dist)])
#         else:
#             # Add a new speaker to memory
#             new_label = max(speaker_labels) + 1
#             speaker_memory = np.vstack([speaker_memory, embeddings[i]])
#             speaker_labels.append(new_label)
#             assigned_labels.append(new_label)

#     return assigned_labels
def assign_consistent_ids(embeddings, speaker_memory, speaker_labels, threshold=0.3):
    distances = cdist(embeddings, speaker_memory, metric="cosine")
    assigned_labels = []

    for i, dist in enumerate(distances):
        if dist.min() < threshold:
            assigned_labels.append(speaker_labels[np.argmin(dist)])
        else:
            new_label = max(speaker_labels) + 1 if speaker_labels else 0
            speaker_memory = np.vstack([speaker_memory, embeddings[i]]) if speaker_memory.size else embeddings[i]
            speaker_labels.append(new_label)
            assigned_labels.append(new_label)

    return assigned_labels, speaker_memory, speaker_labels
