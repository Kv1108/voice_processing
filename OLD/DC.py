import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Parameters
EMBEDDING_DIM = 20
HIDDEN_DIM = 128
NUM_SPEAKERS = 2
BATCH_SIZE = 16
EPOCHS = 10

# Generate Synthetic Data (Replace with your own dataset)
def generate_synthetic_data(num_samples=100, duration=2.0, sr=8000):
    mixtures, sources = [], []
    for _ in range(num_samples):
        s1 = np.random.uniform(-1, 1, int(duration * sr))
        s2 = np.random.uniform(-1, 1, int(duration * sr))
        mixture = s1 + s2
        mixtures.append(mixture)
        sources.append([s1, s2])
    return mixtures, sources

# BLSTM Embedding Network
class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(EmbeddingNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        embeddings = self.fc(lstm_out)
        embeddings = embeddings / (torch.norm(embeddings, dim=-1, keepdim=True) + 1e-8)  # Normalize embeddings
        return embeddings

# Affinity Loss for Clustering
class AffinityLoss(nn.Module):
    def __init__(self):
        super(AffinityLoss, self).__init__()

    def forward(self, embeddings, affinity_matrix):
        embeddings = embeddings.view(-1, embeddings.size(-1))
        embedding_similarity = torch.matmul(embeddings, embeddings.t())
        loss = torch.mean((embedding_similarity - affinity_matrix) ** 2)
        return loss

# Prepare Data (Spectrogram Conversion)
def preprocess_audio(audio, sr=8000, n_fft=256, hop_length=128):
    spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(spectrogram).T
    return magnitude

# Training Function
def train_model(model, loss_fn, optimizer, data_loader, device):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for mixture, affinity_matrix in data_loader:
            mixture = mixture.to(device)
            affinity_matrix = affinity_matrix.to(device)

            optimizer.zero_grad()
            embeddings = model(mixture)
            loss = loss_fn(embeddings, affinity_matrix)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")

# Clustering and Separation
def separate_sources(model, mixture, num_speakers, device):
    model.eval()
    with torch.no_grad():
        mixture = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0).to(device)
        embeddings = model(mixture)
        embeddings = embeddings.squeeze(0).cpu().numpy()
        kmeans = KMeans(n_clusters=num_speakers)
        labels = kmeans.fit_predict(embeddings)
    return labels

# Main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate synthetic dataset
    mixtures, sources = generate_synthetic_data()
    spectrograms = [preprocess_audio(m) for m in mixtures]

    # Create Affinity Matrices (dummy ground truth for demo purposes)
    affinity_matrices = [np.eye(len(spectrogram)) for spectrogram in spectrograms]

    # Convert to PyTorch Datasets and DataLoader
    dataset = [(torch.tensor(spectrogram, dtype=torch.float32),
                torch.tensor(affinity, dtype=torch.float32))
               for spectrogram, affinity in zip(spectrograms, affinity_matrices)]
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model, Loss, and Optimizer
    input_dim = spectrograms[0].shape[1]
    model = EmbeddingNetwork(input_dim, HIDDEN_DIM, EMBEDDING_DIM).to(device)
    loss_fn = AffinityLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train Model
    train_model(model, loss_fn, optimizer, data_loader, device)

    # Test Separation
    test_mixture = preprocess_audio(mixtures[0])
    separated_labels = separate_sources(model, test_mixture, NUM_SPEAKERS, device)

    # Visualize Clustering Results
    plt.imshow(separated_labels.reshape(test_mixture.shape[0], -1).T, aspect='auto')
    plt.title("Clustered Time-Frequency Points")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()
