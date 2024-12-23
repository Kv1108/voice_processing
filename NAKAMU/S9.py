import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder network
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # Mean
        self.fc22 = nn.Linear(400, latent_dim)  # Log variance

        # Decoder network
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28 * 28)
    
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3)).view(-1, 28 * 28)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28 * 28))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction='sum')
    
    # KL divergence loss
    # KL divergence between learned distribution and standard normal distribution
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # where mu and logvar are the parameters of the Gaussian distribution
    # for each dimension of the latent space.
    # See VAE paper for more details.
    # https://arxiv.org/pdf/1312.6114.pdf
    # Equation (4) in the paper.
    
    # You can find the log variance and mean from the encoder network
    # KL divergence loss term
    MSE = torch.sum(logvar.exp() + mu.pow(2) - 1. - logvar)
    return BCE + MSE

# Prepare MNIST dataset and DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=128, shuffle=True)

# Instantiate the model, optimizer, and loss function
model = VAE(latent_dim=20)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
def train(model, train_loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch+1}, Average Loss: {train_loss / len(train_loader.dataset)}')

# Train the VAE
train(model, train_loader, optimizer, epochs=10)

# Generate new samples using the trained VAE
def generate_samples(model, n_samples=10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        samples = model.decode(z)
        samples = samples.view(-1, 1, 28, 28)
        return samples

# Generate and visualize samples
samples = generate_samples(model)
grid = torchvision.utils.make_grid(samples, nrow=5, padding=2)
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.axis('off')
plt.show()
