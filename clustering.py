from sklearn.cluster import AgglomerativeClustering

# def cluster_embeddings(embeddings, n_speakers=None):
#     """
#     Perform clustering on speaker embeddings to identify speaker groups.
    
#     Parameters:
#         embeddings (np.ndarray): Array of speaker embeddings.
#         n_speakers (int, optional): Number of speakers in the audio.
#             If not provided, the clustering will infer the number of clusters.

#     Returns:
#         cluster_labels (list): List of cluster IDs for each embedding.
#     """
#     if n_speakers is None:
#         n_speakers = 2  # Default to 2 speakers if not provided; modify as needed

#     clustering_model = AgglomerativeClustering(n_clusters=n_speakers, affinity="cosine", linkage="average")
#     cluster_labels = clustering_model.fit_predict(embeddings)
#     return cluster_labels
def cluster_embeddings(embeddings, n_speakers=None):
    if embeddings.shape[0] == 0:
        print("No embeddings to cluster.")
        return []
    if n_speakers is None:
        n_speakers = 2  # Default to 2; adapt for dynamic clustering if needed.
    clustering_model = AgglomerativeClustering(n_clusters=n_speakers, affinity="cosine", linkage="average")
    return clustering_model.fit_predict(embeddings)
