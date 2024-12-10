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

# --------------------- Hyperparameters ---------------------
# Define variables
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
LATENT_DIM = 2048
INPUT_DIM = 9  # Dimension of position
OUTPUT_DIM = 9  # Dimension of momenta
EPOCHS = 25
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
PATIENCE = 10
MIN_DELTA = 1e-5

# Define AUTOENCODER_LAYERS
AUTOENCODER_LAYERS = [1024, 512]

# Paths for saving models and statistics
CVAE_WEIGHTS_PATH = 'cvae_weights_CINNV4.pth'
LATENT_STATS_PATH = 'latent_z_stats.npy'
RESULTS_PATH = 'cvae_results_CINNV4.pt'
# -------------------------------------------------------------

# --------------------- Data Loading & Preprocessing --------------------------
# Load data
data = pd.read_csv(FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Split data into train (70%), validation (15%), and test (15%)
train_val_position, test_position, train_val_momenta, test_momenta = train_test_split(
    position, momenta, test_size=0.15, random_state=42
)
train_position, val_position, train_momenta, val_momenta = train_test_split(
    train_val_position, train_val_momenta, test_size=0.1765, random_state=42
)
# The second split ensures the validation set is ~15% of the total data

# Initialize scalers
position_scaler = StandardScaler()
momenta_scaler = StandardScaler()

# Fit scalers on training data only
train_position_normalized = position_scaler.fit_transform(train_position)
train_momenta_normalized = momenta_scaler.fit_transform(train_momenta)

# Transform validation and test data using the fitted scalers
val_position_normalized = position_scaler.transform(val_position)
val_momenta_normalized = momenta_scaler.transform(val_momenta)

test_position_normalized = position_scaler.transform(test_position)
test_momenta_normalized = momenta_scaler.transform(test_momenta)

# Convert to PyTorch tensors and move to device
train_position_tensor = torch.FloatTensor(train_position_normalized).to(device)
val_position_tensor = torch.FloatTensor(val_position_normalized).to(device)
test_position_tensor = torch.FloatTensor(test_position_normalized).to(device)

train_momenta_tensor = torch.FloatTensor(train_momenta_normalized).to(device)
val_momenta_tensor = torch.FloatTensor(val_momenta_normalized).to(device)
test_momenta_tensor = torch.FloatTensor(test_momenta_normalized).to(device)

# Create DataLoaders
train_dataset = TensorDataset(train_position_tensor, train_momenta_tensor)
val_dataset = TensorDataset(val_position_tensor, val_momenta_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# ------------------------------------------------------------------------------

# --------------------- Model Definition ----------------------
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
# -------------------------------------------------------------

# --------------------- Loss Function -------------------------
def cvae_loss_fn(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_divergence
# -------------------------------------------------------------

# --------------------- Training Setup -------------------------
# Build the CVAE model
cvae = CVAE(INPUT_DIM, LATENT_DIM, OUTPUT_DIM, AUTOENCODER_LAYERS).to(device)

# Optimizer
cvae_optimizer = optim.Adam(cvae.parameters(), lr=LEARNING_RATE)

# Training loop for CVAE
cvae_train_losses = []
cvae_val_losses = []
best_cvae_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    # Training phase
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

    print(f'Epoch {epoch+1}, CVAE Loss: {cvae_loss_epoch:.4f}, Val Loss: {cvae_val_loss_epoch:.4f}')

    # Early stopping based on validation loss
    if cvae_val_loss_epoch < best_cvae_loss - MIN_DELTA:
        best_cvae_loss = cvae_val_loss_epoch
        patience_counter = 0
        torch.save(cvae.state_dict(), CVAE_WEIGHTS_PATH)
        print("Saved best CVAE model.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'Early stopping CVAE training after {epoch+1} epochs')
            break

# Load the best model
cvae.load_state_dict(torch.load(CVAE_WEIGHTS_PATH))
cvae.eval()
# -------------------------------------------------------------

# --------------------- Compute Latent Statistics ----------------------
# Compute the mean and std of the latent space from the training data
with torch.no_grad():
    latent_mu, latent_logvar = cvae.encode(train_position_tensor)
    latent_z = cvae.reparameterize(latent_mu, latent_logvar)
    latent_z_np = latent_z.cpu().numpy()
    latent_z_mean = np.mean(latent_z_np, axis=0)
    latent_z_std = np.std(latent_z_np, axis=0)
    # Save the latent distribution parameters
    np.save(LATENT_STATS_PATH, {'mean': latent_z_mean, 'std': latent_z_std})

print(f'Latent statistics saved to {LATENT_STATS_PATH}')
# ----------------------------------------------------------------------

# --------------------- Testing Phase --------------------------
print("Evaluating on test set...")
# Load the latent statistics
latent_z_mean = np.load('latent_z_mean.npy')
latent_z_std = np.load('latent_z_std.npy')

# Convert to torch tensors
latent_z_mean_tensor = torch.tensor(latent_z_mean).to(device)
latent_z_std_tensor = torch.tensor(latent_z_std).to(device)

# Number of test samples
num_test_samples = test_momenta_tensor.size(0)

with torch.no_grad():
    cvae.eval()
    # Sample z for all test samples in a single batch
    z_sample = torch.randn(num_test_samples, LATENT_DIM).to(device) * latent_z_std_tensor + latent_z_mean_tensor
    # Decode all z and y at once for efficiency
    predicted_positions = cvae.decode(z_sample, test_momenta_tensor)
    
    # Detach the predictions and move to CPU
    predicted_positions_np = predicted_positions.detach().cpu().numpy()
    actual_positions_np = test_position_tensor.cpu().numpy()  # Assuming test_position_tensor does not require grad

# Inverse transform the predicted and actual positions
predicted_positions_inverse = position_scaler.inverse_transform(predicted_positions_np)
actual_positions_inverse = position_scaler.inverse_transform(actual_positions_np)

# Calculate evaluation metrics using the inverse transformed positions
mse = np.mean((predicted_positions_inverse - actual_positions_inverse) ** 2)
mae = np.mean(np.abs(predicted_positions_inverse - actual_positions_inverse))

# Calculate relative error
epsilon = 1e-8
relative_errors = np.abs(predicted_positions_inverse - actual_positions_inverse) / (np.abs(actual_positions_inverse) + epsilon)
mean_relative_error = np.mean(relative_errors)

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Mean Relative Error: {mean_relative_error:.4f}")
# -------------------------------------------------------------


# --------------------- Plot Loss Curves ----------------------
plt.figure(figsize=(10,5))
plt.plot(range(1, len(cvae_train_losses)+1), cvae_train_losses, label='Training Loss')
plt.plot(range(1, len(cvae_val_losses)+1), cvae_val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('CVAE Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cvae_loss_plot.png')
plt.close()

# Optionally, plot only the first 10 epochs and the rest separately
plt.figure(figsize=(6, 4))
plt.plot(range(1, min(11, len(cvae_train_losses)+1)), cvae_train_losses[:10], label='Training Loss')
plt.plot(range(1, min(11, len(cvae_val_losses)+1)), cvae_val_losses[:10], label='Validation Loss')
plt.title('CVAE Loss (First 10 Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('cvae_loss_plot_first10.png')
plt.close()

if len(cvae_train_losses) > 10:
    plt.figure(figsize=(6, 4))
    plt.plot(range(11, len(cvae_train_losses)+1), cvae_train_losses[10:], label='Training Loss')
    plt.plot(range(11, len(cvae_val_losses)+1), cvae_val_losses[10:], label='Validation Loss')
    plt.title('CVAE Loss (Epochs 11 Onwards)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cvae_loss_plot_rest.png')
    plt.close()
# -------------------------------------------------------------

# --------------------- Save the Model and Results -------------------------
# Save results
results = {
    'cvae_train_losses': cvae_train_losses,
    'cvae_val_losses': cvae_val_losses,
    'test_mse': mse,
    'test_mae': mae,
    'mean_relative_error': mean_relative_error
}

torch.save(results, RESULTS_PATH)
print("Results saved.")
# -------------------------------------------------------------
