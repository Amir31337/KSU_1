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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define variables
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/generated_cos3d_check.csv'
LATENT_DIM = 256
INPUT_DIM = 9  # Dimension of position
OUTPUT_DIM = 9  # Dimension of momenta
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 0.001
PATIENCE = 10
MIN_DELTA = 1e-3
NUM_DIVERSE_SAMPLES = 1000  # Number of diverse latent samples to store

# Architecture configuration
AUTOENCODER_LAYERS = [512, 256]
CINN_LAYERS = [256, 512, 256]

# Load and preprocess data
data = pd.read_csv(FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Normalize the position using MinMaxScaler
position_scaler = MinMaxScaler()
position_normalized = position_scaler.fit_transform(position)

# Split data into train and test sets
train_position, test_position, train_momenta, test_momenta = train_test_split(
    position_normalized, momenta, test_size=TEST_RATIO, random_state=42
)

# Convert to PyTorch tensors and move to device
train_position = torch.FloatTensor(train_position).to(device)
test_position = torch.FloatTensor(test_position).to(device)
train_momenta = torch.FloatTensor(train_momenta).to(device)
test_momenta = torch.FloatTensor(test_momenta).to(device)

# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_layers):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.ELU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ELU(),
            nn.Linear(hidden_layers[1], latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_layers[1]),
            nn.ELU(),
            nn.Linear(hidden_layers[1], hidden_layers[0]),
            nn.ELU(),
            nn.Linear(hidden_layers[0], input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z

# Define the CINN model
class CINN(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_layers, output_dim):
        super(CINN, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim + condition_dim, hidden_layers[0]))
        self.layers.append(nn.ELU())

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.ELU())

        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, x, condition):
        combined = torch.cat((x, condition), dim=1)
        for layer in self.layers:
            combined = layer(combined)
        return combined

# Loss functions
def autoencoder_loss_fn(recon_x, x):
    return nn.functional.mse_loss(recon_x, x)

def cinn_loss_fn(pred_position, true_position):
    return nn.functional.mse_loss(pred_position, true_position)

# Build the models and move them to device
autoencoder = Autoencoder(INPUT_DIM, LATENT_DIM, AUTOENCODER_LAYERS).to(device)
cinn = CINN(LATENT_DIM, OUTPUT_DIM, CINN_LAYERS, INPUT_DIM).to(device)

# Optimizers
autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
cinn_optimizer = optim.Adam(cinn.parameters(), lr=LEARNING_RATE)

# Create DataLoaders
train_dataset = TensorDataset(train_position, train_momenta)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop for autoencoder
autoencoder_losses = []
autoencoder_recon_errors = []
cinn_losses = []

# Train autoencoder
print("Training autoencoder...")
best_ae_val_loss = float('inf')
ae_patience_counter = 0
for epoch in range(EPOCHS):
    autoencoder.train()
    autoencoder_loss_epoch = 0
    autoencoder_recon_error_epoch = 0

    for batch_idx, (position, _) in enumerate(train_loader):
        autoencoder_optimizer.zero_grad()
        recon_position, _ = autoencoder(position)
        autoencoder_loss = autoencoder_loss_fn(recon_position, position)
        autoencoder_loss.backward()
        autoencoder_optimizer.step()
        autoencoder_loss_epoch += autoencoder_loss.item()
        autoencoder_recon_error_epoch += torch.mean(torch.abs(recon_position - position)).item()

    autoencoder_loss_epoch /= len(train_loader)
    autoencoder_recon_error_epoch /= len(train_loader)
    autoencoder_losses.append(autoencoder_loss_epoch)
    autoencoder_recon_errors.append(autoencoder_recon_error_epoch)

    print(f'Epoch {epoch+1}, Autoencoder loss: {autoencoder_loss_epoch:.4f}, Recon error: {autoencoder_recon_error_epoch:.4f}')

    if autoencoder_loss_epoch < best_ae_val_loss - MIN_DELTA:
        best_ae_val_loss = autoencoder_loss_epoch
        ae_patience_counter = 0
        torch.save(autoencoder.state_dict(), 'autoencoder_weights_CINNV4.pth')
        print("Saved best autoencoder model.")
    else:
        ae_patience_counter += 1
        if ae_patience_counter >= PATIENCE:
            print(f'Early stopping autoencoder training after {epoch+1} epochs')
            break

# Generate and store latent z from training set
print("Generating diverse latent representations for training data...")
autoencoder.eval()
diverse_indices = torch.randperm(len(train_position))[:NUM_DIVERSE_SAMPLES]
diverse_positions = train_position[diverse_indices]
with torch.no_grad():
    _, diverse_z = autoencoder(diverse_positions)
torch.save(diverse_z, 'diverse_z_CINNV4.pt')

# Train CINN using the stored z values
print("Training CINN...")
best_cinn_loss = float('inf')
cinn_patience_counter = 0
for epoch in range(EPOCHS):
    cinn.train()
    cinn_loss_epoch = 0

    for position_batch, momenta_batch in train_loader:
        cinn_optimizer.zero_grad()
        # Randomly select latent representations for this batch
        batch_z = diverse_z[torch.randint(0, NUM_DIVERSE_SAMPLES, (len(position_batch),))]
        pred_position = cinn(batch_z, momenta_batch)
        cinn_loss = cinn_loss_fn(pred_position, position_batch)
        cinn_loss.backward()
        cinn_optimizer.step()
        cinn_loss_epoch += cinn_loss.item()

    cinn_loss_epoch /= len(train_loader)
    cinn_losses.append(cinn_loss_epoch)

    print(f'Epoch {epoch+1}, CINN loss: {cinn_loss_epoch:.4f}')

    if cinn_loss_epoch < best_cinn_loss - MIN_DELTA:
        best_cinn_loss = cinn_loss_epoch
        cinn_patience_counter = 0
        torch.save(cinn.state_dict(), 'cinn_weights_CINNV4.pth')
        print("Saved best CINN model.")
    else:
        cinn_patience_counter += 1
        if cinn_patience_counter >= PATIENCE:
            print(f'Early stopping CINN training after {epoch+1} epochs')
            break

# Load the best models and diverse_z for evaluation
autoencoder.load_state_dict(torch.load('autoencoder_weights_CINNV4.pth', map_location=device))
cinn.load_state_dict(torch.load('cinn_weights_CINNV4.pth', map_location=device))
diverse_z = torch.load('diverse_z_CINNV4.pt')  # Load the stored latent z from training phase

# Evaluate on test set
print("Evaluating on test set...")
cinn.eval()
all_predicted_positions = []
all_true_positions = []

with torch.no_grad():
    for i in range(len(test_momenta)):
        # Ensure z is loaded from saved latent representations (no new latent z should be created here)
        z = diverse_z[torch.randint(0, NUM_DIVERSE_SAMPLES, (1,))]
        momenta = test_momenta[i].unsqueeze(0)
        predicted_position = cinn(z, momenta)
        all_predicted_positions.append(predicted_position)
        all_true_positions.append(test_position[i].unsqueeze(0))

# Concatenate all batch results
all_predicted_positions = torch.cat(all_predicted_positions, dim=0)
all_true_positions = torch.cat(all_true_positions, dim=0)

# Calculate evaluation metrics
mse = nn.functional.mse_loss(all_predicted_positions, all_true_positions).item()
mae = nn.functional.l1_loss(all_predicted_positions, all_true_positions).item()

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Visualize results
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(autoencoder_losses)
plt.title('Autoencoder Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(132)
plt.plot(autoencoder_recon_errors)
plt.title('Autoencoder Reconstruction Error')
plt.xlabel('Epoch')
plt.ylabel('Error')

plt.subplot(133)
plt.plot(cinn_losses)
plt.title('CINN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.savefig('training_plots_CINNV4.png')
plt.close()

# Save results
results = {
    'autoencoder_losses': autoencoder_losses,
    'autoencoder_recon_errors': autoencoder_recon_errors,
    'cinn_losses': cinn_losses,
    'test_mse': mse,
    'test_mae': mae
}

torch.save(results, 'results_CINNV4.pt')
print("Results saved.")
