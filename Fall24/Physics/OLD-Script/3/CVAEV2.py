import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define variables (soft-coded at the beginning)
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/generated_cos3d_check.csv'
#FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/random_cos3d_10000.csv'
LATENT_DIM = 32  # Latent space dimension
POSITION_DIM = 9  # Dimension of position
MOMENTA_DIM = 9  # Dimension of momenta
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
EPOCHS = 100
BATCH_SIZE = 512  # Batch size for training
LEARNING_RATE = 0.01
PATIENCE = 20
MIN_DELTA = 1e-3
GPU_INDEX = 2  # Specify which GPU to use (in this case, cuda:2)

# Encoder and Decoder architecture configuration
ENCODER_LAYERS = [512, 256]  # Encoder hidden layers
DECODER_LAYERS = [256, 512]  # Decoder hidden layers

# Set device to use cuda:2 explicitly
#device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
else:
    print("CUDA is not available. Using CPU.")

# Load and preprocess data
data = pd.read_csv(FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Normalize the data using MinMaxScaler
position_scaler = MinMaxScaler()
momenta_scaler = MinMaxScaler()

position_normalized = position_scaler.fit_transform(position)
momenta_normalized = momenta_scaler.fit_transform(momenta)

# Split data into train and test sets
train_position, test_position, train_momenta, test_momenta = train_test_split(
    position_normalized, momenta_normalized, test_size=TEST_RATIO, random_state=42
)

# Convert to PyTorch tensors
train_position = torch.FloatTensor(train_position).to(device)
train_momenta = torch.FloatTensor(train_momenta).to(device)
test_position = torch.FloatTensor(test_position).to(device)
test_momenta = torch.FloatTensor(test_momenta).to(device)

# Define the encoder (for learning latent representation from position and momenta)
class Encoder(nn.Module):
    def __init__(self, position_dim, momenta_dim, latent_dim, hidden_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(position_dim + momenta_dim, hidden_layers[0]))
        self.layers.append(nn.LeakyReLU())

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.layers.append(nn.LeakyReLU())

        # Output layers for mean and log variance
        self.fc_mean = nn.Linear(hidden_layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[-1], latent_dim)

    def forward(self, position, momenta):
        x = torch.cat([position, momenta], dim=1)
        for layer in self.layers:
            x = layer(x)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_logvar(x)
        return z_mean, z_log_var

# Define the decoder (for predicting position from latent space and momenta)
class Decoder(nn.Module):
    def __init__(self, latent_dim, momenta_dim, output_dim, hidden_layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(latent_dim + momenta_dim, hidden_layers[0]))
        self.layers.append(nn.LeakyReLU())

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.layers.append(nn.LeakyReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, z, momenta):
        x = torch.cat([z, momenta], dim=1)
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)

# Define the CVAE model
class CVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, position, momenta):
        mu, logvar = self.encoder(position, momenta)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, momenta), mu, logvar

# Loss function
def loss_function(recon_position, position, mu, logvar):
    MSE = nn.functional.mse_loss(recon_position, position, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

# Build the CVAE model
encoder = Encoder(POSITION_DIM, MOMENTA_DIM, LATENT_DIM, ENCODER_LAYERS).to(device)
decoder = Decoder(LATENT_DIM, MOMENTA_DIM, POSITION_DIM, DECODER_LAYERS).to(device)
cvae = CVAE(encoder, decoder).to(device)

# Optimizer
optimizer = optim.Adam(cvae.parameters(), lr=LEARNING_RATE)

# Create DataLoader
train_dataset = TensorDataset(train_position, train_momenta)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
train_losses = []
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    cvae.train()
    train_loss = 0
    for batch_idx, (position, momenta) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_position, mu, logvar = cvae(position, momenta)
        loss = loss_function(recon_position, position, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    print(f'Epoch {epoch + 1}, Train loss: {train_loss:.4f}')

    if train_loss < best_val_loss - MIN_DELTA:
        best_val_loss = train_loss
        patience_counter = 0
        torch.save(cvae.state_dict(), 'cvae_model_weights.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'Early stopping after {epoch + 1} epochs')
            break

# Load the best model
cvae.load_state_dict(torch.load('cvae_model_weights.pth'))

# Evaluate on test set
cvae.eval()
with torch.no_grad():
    # Sample from the learned latent distribution
    mu, logvar = cvae.encoder(train_position, train_momenta)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std

    # Randomly select a subset of z that matches the size of test_momenta
    indices = torch.randperm(z.size(0))[:test_momenta.size(0)]
    z_subset = z[indices]

    # Predict position using the sampled latent representation subset and test momenta
    predicted_position = cvae.decoder(z_subset, test_momenta)

    # Reshape the predicted position to match the original shape
    predicted_position = predicted_position.view(-1, POSITION_DIM)


# Denormalize the predictions and true positions
predicted_position_denorm = position_scaler.inverse_transform(predicted_position.cpu().numpy())
true_position_denorm = position_scaler.inverse_transform(test_position.cpu().numpy())

# Calculate evaluation metrics
mse = np.mean((true_position_denorm - predicted_position_denorm) ** 2)
mae = np.mean(np.abs(true_position_denorm - predicted_position_denorm))

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Save the predicted positions
np.savetxt('predicted_positions.csv', predicted_position_denorm, delimiter=',')
print("Predicted positions have been saved as 'predicted_positions.csv'")

# Plot learning curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('learning_curve.png')
plt.close()

print("Learning curve has been saved as 'learning_curve.png'")