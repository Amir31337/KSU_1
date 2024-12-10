#FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/1M.csv' : Test MSE: 1.0610 Test MAE: 0.8220
#FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/random_cos3d_10000.csv': Test MSE: 0.1786 Test MAE: 0.3510
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Clear GPU memory
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define variables
FILEPATH = 'cei_traning_orient_1.csv'
#random_cos3d_10000.csv
#generated_cos3d_check.csv
#cei_traning_orient_1.csv
LATENT_DIM = 2048
INPUT_DIM = 9  # Dimension of position
OUTPUT_DIM = 9  # Dimension of momenta
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
EPOCHS = 1000
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
PATIENCE = 10
MIN_DELTA = 1e-5
NUM_DIVERSE_SAMPLES = 1000  # Number of diverse latent samples to store

# Define AUTOENCODER_LAYERS
AUTOENCODER_LAYERS = [1024, 512]

# Load and preprocess data
data = pd.read_csv(FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Normalize the position using StandardScaler
position_scaler = StandardScaler()
momenta_scaler = StandardScaler()
position_normalized = position_scaler.fit_transform(position)
momenta_normalized = momenta_scaler.fit_transform(momenta)

# Split data into train and test sets
train_position, test_position, train_momenta, test_momenta = train_test_split(
    position_normalized, momenta_normalized, test_size=TEST_RATIO, random_state=42
)

# Convert to PyTorch tensors and move to device
train_position = torch.FloatTensor(train_position).to(device)
test_position = torch.FloatTensor(test_position).to(device)
train_momenta = torch.FloatTensor(train_momenta).to(device)
test_momenta = torch.FloatTensor(test_momenta).to(device)

# Define the CVAE model
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_layers):
        super(CVAE, self).__init__()
        
        # Encoder: Encodes positions X into latent mean and variance
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.ELU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ELU()
        )
        self.fc_mu = nn.Linear(hidden_layers[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[1], latent_dim)
        
        # Decoder: Decodes latent Z conditioned on momenta Y
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_layers[1]),
            nn.ELU(),
            nn.Linear(hidden_layers[1], hidden_layers[0]),
            nn.ELU(),
            nn.Linear(hidden_layers[0], input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        combined = torch.cat((z, condition), dim=1)
        return self.decoder(combined)

    def forward(self, x, condition):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, condition)
        return recon_x, mu, logvar

# Loss functions for CVAE
def cvae_loss_fn(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence

# Build the CVAE model
cvae = CVAE(INPUT_DIM, LATENT_DIM, OUTPUT_DIM, AUTOENCODER_LAYERS).to(device)

# Optimizer
cvae_optimizer = optim.Adam(cvae.parameters(), lr=LEARNING_RATE)

# Create DataLoader
train_dataset = TensorDataset(train_position, train_momenta)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop for CVAE
cvae_losses = []
best_cvae_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    cvae.train()
    cvae_loss_epoch = 0

    for batch_idx, (position, momenta) in enumerate(train_loader):
        cvae_optimizer.zero_grad()
        recon_position, mu, logvar = cvae(position, momenta)
        loss = cvae_loss_fn(recon_position, position, mu, logvar)
        loss.backward()
        cvae_optimizer.step()
        cvae_loss_epoch += loss.item()

    cvae_loss_epoch /= len(train_loader)
    cvae_losses.append(cvae_loss_epoch)

    print(f'Epoch {epoch+1}, CVAE loss: {cvae_loss_epoch:.4f}')

    if cvae_loss_epoch < best_cvae_loss - MIN_DELTA:
        best_cvae_loss = cvae_loss_epoch
        patience_counter = 0
        torch.save(cvae.state_dict(), 'cvae_weights_CINNV4.pth')
        print("Saved best CVAE model.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'Early stopping CVAE training after {epoch+1} epochs')
            break

# Save the best latent distribution parameters
cvae.eval()
with torch.no_grad():
    latent_mu, latent_logvar = cvae.encode(train_position)
    torch.save((latent_mu, latent_logvar), 'cvae_latent_params_CINNV4.pt')

# Testing phase (No encoder on test data)
print("Evaluating on test set...")
with torch.no_grad():
    latent_mu, latent_logvar = torch.load('cvae_latent_params_CINNV4.pt', weights_only=True)
    z_sample = cvae.reparameterize(latent_mu, latent_logvar)
    all_predicted_positions = []

    for i in range(len(test_momenta)):
        momenta = test_momenta[i].unsqueeze(0)
        predicted_position = cvae.decode(z_sample[i].unsqueeze(0), momenta)
        all_predicted_positions.append(predicted_position)

# Concatenate predicted positions into a single tensor
all_predicted_positions = torch.cat(all_predicted_positions, dim=0)

# Inverse transform the predicted and actual positions
all_predicted_positions_inverse = position_scaler.inverse_transform(all_predicted_positions.cpu().numpy())
test_position_inverse = position_scaler.inverse_transform(test_position.cpu().numpy())

# Calculate evaluation metrics using the inverse transformed positions
mse = np.mean((all_predicted_positions_inverse - test_position_inverse) ** 2)
mae = np.mean(np.abs(all_predicted_positions_inverse - test_position_inverse))

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Visualize CVAE losses
plt.figure(figsize=(6, 4))
plt.plot(cvae_losses)
plt.title('CVAE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('cvae_loss_plot_CINNV4.png')
plt.close()

# Save results
results = {
    'cvae_losses': cvae_losses,
    'test_mse': mse,
    'test_mae': mae
}

torch.save(results, 'cvae_results_CINNV4.pt')
print("Results saved.")
