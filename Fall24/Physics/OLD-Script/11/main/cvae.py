'''
Epoch 1, CVAE loss: 2.5584, Val loss: 0.9516
Saved best CVAE model.
Epoch 2, CVAE loss: 1.1074, Val loss: 0.9500
Saved best CVAE model.
Epoch 3, CVAE loss: 1.0833, Val loss: 0.9425
Saved best CVAE model.
Epoch 4, CVAE loss: 1.0827, Val loss: 0.9463
Epoch 5, CVAE loss: 1.0836, Val loss: 0.9445
Epoch 6, CVAE loss: 1.0782, Val loss: 0.9404
Saved best CVAE model.
Epoch 7, CVAE loss: 1.0895, Val loss: 0.9400
Epoch 8, CVAE loss: 1.0820, Val loss: 0.9400
Epoch 9, CVAE loss: 1.0829, Val loss: 0.9424
Epoch 10, CVAE loss: 1.0806, Val loss: 0.9426
Epoch 11, CVAE loss: 1.0783, Val loss: 0.9379
Saved best CVAE model.
Epoch 12, CVAE loss: 1.0755, Val loss: 0.9323
Saved best CVAE model.
Epoch 13, CVAE loss: 1.0749, Val loss: 0.9379
Epoch 14, CVAE loss: 1.0726, Val loss: 0.9332
Epoch 15, CVAE loss: 1.0712, Val loss: 0.9372
Epoch 16, CVAE loss: 1.0704, Val loss: 0.9350
Epoch 17, CVAE loss: 1.0708, Val loss: 0.9335
Early stopping CVAE training after 17 epochs
Evaluating on test set...
Test MSE: 0.7312
Mean Relative Error: 0.7033
Results saved.
'''
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
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
LATENT_DIM = 1920
INPUT_DIM = 9  # Dimension of position
OUTPUT_DIM = 9  # Dimension of momenta
EPOCHS = 182
BATCH_SIZE = 128
LEARNING_RATE = 5.772971044368426e-04
PATIENCE = 5
MIN_DELTA = 1e-10

# Define AUTOENCODER_LAYERS
AUTOENCODER_LAYERS = [1152,1152]

# Load and preprocess data
data = pd.read_csv(FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Normalize the position using StandardScaler, don't normalize momenta
position_scaler = StandardScaler()
position_normalized = position_scaler.fit_transform(position)
momenta_normalized = momenta

# Split data into train, validation, and test sets
train_val_position, test_position, train_val_momenta, test_momenta = train_test_split(
    position_normalized, momenta_normalized, test_size=0.15, random_state=42
)
train_position, val_position, train_momenta, val_momenta = train_test_split(
    train_val_position, train_val_momenta, test_size=0.1765, random_state=42
)
# The second split ensures the validation set is 15% of the total data

# Convert to PyTorch tensors and move to device
train_position = torch.FloatTensor(train_position).to(device)
val_position = torch.FloatTensor(val_position).to(device)
test_position = torch.FloatTensor(test_position).to(device)
train_momenta = torch.FloatTensor(train_momenta).to(device)
val_momenta = torch.FloatTensor(val_momenta).to(device)
test_momenta = torch.FloatTensor(test_momenta).to(device)

# Define the CVAE model
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_layers):
        super(CVAE, self).__init__()
        print(hidden_layers)
        # Encoder: Encodes positions X into latent mean and variance
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_layers[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[1], latent_dim)
        
        # Decoder: Decodes latent Z conditioned on momenta Y
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], hidden_layers[0]),
            nn.ReLU(),
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
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_divergence

# Build the CVAE model
cvae = CVAE(INPUT_DIM, LATENT_DIM, OUTPUT_DIM, AUTOENCODER_LAYERS).to(device)

# Optimizer
cvae_optimizer = optim.Adam(cvae.parameters(), lr=LEARNING_RATE)

# Create DataLoader
train_dataset = TensorDataset(train_position, train_momenta)
val_dataset = TensorDataset(val_position, val_momenta)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training loop for CVAE
cvae_train_losses = []
cvae_val_losses = []
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
    cvae_train_losses.append(cvae_loss_epoch)

    # Validation phase
    cvae.eval()
    cvae_val_loss_epoch = 0
    with torch.no_grad():
        for batch_idx, (position, momenta) in enumerate(val_loader):
            recon_position, mu, logvar = cvae(position, momenta)
            loss = cvae_loss_fn(recon_position, position, mu, logvar)
            cvae_val_loss_epoch += loss.item()
    cvae_val_loss_epoch /= len(val_loader)
    cvae_val_losses.append(cvae_val_loss_epoch)

    print(f'Epoch {epoch+1}, CVAE loss: {cvae_loss_epoch:.4f}, Val loss: {cvae_val_loss_epoch:.4f}')

    # Early stopping based on validation loss
    if cvae_val_loss_epoch < best_cvae_loss - MIN_DELTA:
        best_cvae_loss = cvae_val_loss_epoch
        patience_counter = 0
        torch.save(cvae.state_dict(), 'cvae_weights_CINNV4.pth')
        print("Saved best CVAE model.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'Early stopping CVAE training after {epoch+1} epochs')
            break

# Save the best latent distribution parameters
cvae.load_state_dict(torch.load('cvae_weights_CINNV4.pth', weights_only=True))
cvae.eval()
with torch.no_grad():
    latent_mu, latent_logvar = cvae.encode(train_position)
    latent_z = cvae.reparameterize(latent_mu, latent_logvar)
    latent_z_np = latent_z.cpu().numpy()
    latent_z_mean = np.mean(latent_z_np, axis=0)
    latent_z_std = np.std(latent_z_np, axis=0)
    # Save the latent distribution parameters
    np.save('latent_z_mean.npy', latent_z_mean)
    np.save('latent_z_std.npy', latent_z_std)

# Testing phase (No encoder on test data)
print("Evaluating on test set...")
latent_z_mean = np.load('latent_z_mean.npy')
latent_z_std = np.load('latent_z_std.npy')

latent_z_mean = torch.tensor(latent_z_mean).to(device)
latent_z_std = torch.tensor(latent_z_std).to(device)

with torch.no_grad():
    cvae.eval()
    z_sample = torch.randn(len(test_momenta), LATENT_DIM).to(device) * latent_z_std + latent_z_mean
    all_predicted_positions = []

    for i in range(len(test_momenta)):
        momenta = test_momenta[i].unsqueeze(0)
        z = z_sample[i].unsqueeze(0)
        predicted_position = cvae.decode(z, momenta)
        all_predicted_positions.append(predicted_position)

# Concatenate predicted positions into a single tensor
all_predicted_positions = torch.cat(all_predicted_positions, dim=0)

# Inverse transform the predicted and actual positions
all_predicted_positions_inverse = position_scaler.inverse_transform(all_predicted_positions.cpu().numpy())
test_position_inverse = position_scaler.inverse_transform(test_position.cpu().numpy())

# Calculate evaluation metrics using the inverse transformed positions
mse = np.mean((all_predicted_positions_inverse - test_position_inverse) ** 2)
mae = np.mean(np.abs(all_predicted_positions_inverse - test_position_inverse))

# Calculate relative error
epsilon = 1e-8
relative_errors = np.abs(all_predicted_positions_inverse - test_position_inverse) / (np.abs(test_position_inverse) + epsilon)
mean_relative_error = np.mean(relative_errors)

print(f"Test MSE: {mse:.4f}")
#print(f"Test MAE: {mae:.4f}")
print(f"Mean Relative Error: {mean_relative_error:.4f}")

# Visualize CVAE losses for first 10 epochs
plt.figure(figsize=(6, 4))
plt.plot(range(1, min(11, len(cvae_train_losses)+1)), cvae_train_losses[:10], label='Training Loss')
plt.plot(range(1, min(11, len(cvae_val_losses)+1)), cvae_val_losses[:10], label='Validation Loss')
plt.title('CVAE Loss (First 10 Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('cvae_loss_plot_first10_CINNV4.png')
plt.close()

# Visualize CVAE losses for remaining epochs
if len(cvae_train_losses) > 10:
    plt.figure(figsize=(6, 4))
    plt.plot(range(11, len(cvae_train_losses)+1), cvae_train_losses[10:], label='Training Loss')
    plt.plot(range(11, len(cvae_val_losses)+1), cvae_val_losses[10:], label='Validation Loss')
    plt.title('CVAE Loss (Epochs 11 Onwards)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cvae_loss_plot_rest_CINNV4.png')
    plt.close()

# Save results
results = {
    'cvae_train_losses': cvae_train_losses,
    'cvae_val_losses': cvae_val_losses,
    'test_mse': mse,
    'test_mae': mae,
    'mean_relative_error': mean_relative_error
}

torch.save(results, 'cvae_results_CINNV4.pt')
print("Results saved.")
