import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
)
import matplotlib.pyplot as plt
import json
import torchsummary
import torchinfo
from torch.cuda.amp import GradScaler, autocast  # Corrected import
import joblib  # For saving and loading scalers
import multiprocessing  # For dynamic num_workers setting

# ---------------------------
# 1. Configuration Section
# ---------------------------

# Define the path to your data file
DATA_PATH = 'sim_million_orient.csv'

# Define the directory where all output files will be saved
SAVE_DIR = '/workspace/nf/sim/opt'

# Create SAVE_DIR if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Define file names
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'best_model.pth')
DETAILED_RESULTS_PATH = os.path.join(SAVE_DIR, 'detailed_results.json')
FIRST_EPOCH_PLOT_PATH = os.path.join(SAVE_DIR, 'first.png')
REST_EPOCH_PLOT_PATH = os.path.join(SAVE_DIR, 'rest.png')
LATENT_STATS_PATH = os.path.join(SAVE_DIR, 'latent_stats.pt')
MOMENTA_SCALER_PATH = os.path.join(SAVE_DIR, 'momenta_scaler.pkl')
POSITION_SCALER_PATH = os.path.join(SAVE_DIR, 'position_scaler.pkl')
RESULTS_PATH = os.path.join(SAVE_DIR, 'results.json')

# ---------------------------
# 2. Hyperparameters and Settings
# ---------------------------

# Hyperparameters
LATENT_DIM = 256
EPOCHS = 50
BATCH_SIZE = 4096  # Increased batch size
LEARNING_RATE = 1e-4
PATIENCE = 20
MIN_DELTA = 1e-3
activation_name = 'Sigmoid'  # Choose from supported_activation_functions
position_norm_method = 'MinMaxScaler'  # Choose from supported_normalization_methods
momenta_norm_method = 'MinMaxScaler'  # Choose from supported_normalization_methods
use_l1 = True
L1_LAMBDA = 1e-3  # Low value for L1 regularization
use_l2 = True  # Flag to enable L2 regularization
L2_LAMBDA = 1e-3  # Low value for L2 regularization (weight_decay)
BETA = 0.01  # Low value for Beta parameter
num_hidden_layers = 2
hidden_layer_size = 64

# ---------------------------
# 3. Device Configuration
# ---------------------------

# Enable cuDNN benchmark for performance optimization
torch.backends.cudnn.benchmark = True

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Determine the device type dynamically
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------
# 4. Data Loading and Preprocessing
# ---------------------------

# Load data
data = pd.read_csv(DATA_PATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Split data into train, validation, and test sets (70%, 15%, 15%)
train_position, temp_position, train_momenta, temp_momenta = train_test_split(
    position, momenta, test_size=0.3, random_state=42, shuffle=True
)

val_position, test_position, val_momenta, test_momenta = train_test_split(
    temp_position, temp_momenta, test_size=0.5, random_state=42, shuffle=True
)

# Convert to PyTorch tensors (keep data on CPU)
train_position = torch.FloatTensor(train_position)
val_position = torch.FloatTensor(val_position)
test_position = torch.FloatTensor(test_position)
train_momenta = torch.FloatTensor(train_momenta)
val_momenta = torch.FloatTensor(val_momenta)
test_momenta = torch.FloatTensor(test_momenta)

# ---------------------------
# 5. Activation Functions and Normalization Methods
# ---------------------------

# Define supported activation functions
supported_activation_functions = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'ELU': nn.ELU(),
    'SELU': nn.SELU(),
    'PReLU': nn.PReLU(),
    'ReLU6': nn.ReLU6(),
    'GELU': nn.GELU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
}

# Activation function
activation_function = supported_activation_functions.get(activation_name, nn.ReLU())
if activation_name not in supported_activation_functions:
    print(f"Unsupported activation '{activation_name}'. Defaulting to ReLU.")

# Define supported normalization methods
supported_normalization_methods = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'MaxAbsScaler': MaxAbsScaler(),
    'Normalizer': Normalizer(),
}

# Normalization methods
position_scaler = supported_normalization_methods.get(position_norm_method, None)
if position_norm_method not in supported_normalization_methods:
    print(f"Unsupported normalization method '{position_norm_method}' for position. No scaling applied.")

momenta_scaler = supported_normalization_methods.get(momenta_norm_method, None)
if momenta_norm_method not in supported_normalization_methods:
    print(f"Unsupported normalization method '{momenta_norm_method}' for momenta. No scaling applied.")

# Normalize the data based on specified methods
if position_scaler is not None:
    train_position_norm = torch.FloatTensor(position_scaler.fit_transform(train_position.numpy()))
    val_position_norm = torch.FloatTensor(position_scaler.transform(val_position.numpy()))
    test_position_norm = torch.FloatTensor(position_scaler.transform(test_position.numpy()))
else:
    train_position_norm = train_position
    val_position_norm = val_position
    test_position_norm = test_position

if momenta_scaler is not None:
    train_momenta_norm = torch.FloatTensor(momenta_scaler.fit_transform(train_momenta.numpy()))
    val_momenta_norm = torch.FloatTensor(momenta_scaler.transform(val_momenta.numpy()))
    test_momenta_norm = torch.FloatTensor(momenta_scaler.transform(test_momenta.numpy()))
else:
    train_momenta_norm = train_momenta
    val_momenta_norm = val_momenta
    test_momenta_norm = test_momenta

# ---------------------------
# 6. Model Definition
# ---------------------------

# Hidden layers configuration
hidden_layers = [hidden_layer_size * (2 ** i) for i in range(num_hidden_layers)]  # [64, 128]

# Define the CVAE model with the specified hyperparameters
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_layers, activation_function):
        super(CVAE, self).__init__()

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_layers[0]))
        encoder_layers.append(activation_function)
        for i in range(len(hidden_layers) - 1):
            encoder_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            encoder_layers.append(activation_function)
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(hidden_layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[-1], latent_dim)

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim + condition_dim, hidden_layers[-1]))
        decoder_layers.append(activation_function)
        for i in reversed(range(len(hidden_layers) - 1)):
            decoder_layers.append(nn.Linear(hidden_layers[i+1], hidden_layers[i]))
            decoder_layers.append(activation_function)
        decoder_layers.append(nn.Linear(hidden_layers[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

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

# Initialize the model with correct keyword arguments
model = CVAE(
    input_dim=9,               # Corrected from INPUT_DIM to input_dim
    latent_dim=LATENT_DIM,
    condition_dim=9,
    hidden_layers=hidden_layers,
    activation_function=activation_function
).to(device)

# Optionally use DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)

# ---------------------------
# 7. Optimizer and Loss Function
# ---------------------------

# Define the optimizer with L2 regularization (weight_decay) if enabled
if use_l2:
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)
    print(f"L2 regularization enabled with weight_decay={L2_LAMBDA}")
else:
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    print("L2 regularization not used (weight_decay=0.0)")

# Define the loss function with L1 and Beta regularization if applicable
def loss_fn(recon_x, x, mu, logvar):
    # Reconstruction loss (Mean Squared Error)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')

    # KL Divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_divergence /= x.size(0) * x.size(1)  # Normalize by batch size and feature size

    # Apply Beta parameter to KL divergence
    kl_divergence = BETA * kl_divergence

    # Total loss
    loss = recon_loss + kl_divergence

    # L1 Regularization
    if use_l1:
        l1_loss = 0.0
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        loss += L1_LAMBDA * l1_loss

    return loss

# ---------------------------
# 8. DataLoaders
# ---------------------------

# Determine the number of workers based on the system's CPU count
max_workers = 12  # As per the warning
num_workers = min(multiprocessing.cpu_count() - 1, max_workers)

# Create DataLoaders with the optimized num_workers
train_loader = DataLoader(
    TensorDataset(train_position_norm, train_momenta_norm),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,  # Changed from 16 to a dynamic value <= 12
    pin_memory=True,
    persistent_workers=True
)
val_loader = DataLoader(
    TensorDataset(val_position_norm, val_momenta_norm),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,  # Changed from 16 to a dynamic value <= 12
    pin_memory=True,
    persistent_workers=True
)
test_loader = DataLoader(
    TensorDataset(test_position_norm, test_momenta_norm),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,  # Changed from 16 to a dynamic value <= 12
    pin_memory=True,
    persistent_workers=True
)

# ---------------------------
# 9. Training Loop
# ---------------------------

# Initialize lists to store training and validation losses
train_losses = []
val_losses = []

# Initialize GradScaler for mixed precision with device_type
scaler = GradScaler()  # Removed device_type argument

# Training loop with early stopping and mixed precision
best_val_loss = float('inf')
patience_counter = 0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_x, batch_cond in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_cond = batch_cond.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast():  # Removed device_type argument
            recon_x, mu, logvar = model(batch_x, batch_cond)
            loss = loss_fn(recon_x, batch_x, mu, logvar)

        scaler.scale(loss).backward()
        # Gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_cond in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_cond = batch_cond.to(device, non_blocking=True)
            with autocast():  # Removed device_type argument
                recon_x, mu, logvar = model(batch_x, batch_cond)
                loss = loss_fn(recon_x, batch_x, mu, logvar)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Early stopping check
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Best model saved at epoch {epoch+1}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# ---------------------------
# 10. Saving Scalers
# ---------------------------

if position_scaler is not None:
    joblib.dump(position_scaler, POSITION_SCALER_PATH)
    print(f"Position scaler saved to {POSITION_SCALER_PATH}")
if momenta_scaler is not None:
    joblib.dump(momenta_scaler, MOMENTA_SCALER_PATH)
    print(f"Momenta scaler saved to {MOMENTA_SCALER_PATH}")

# ---------------------------
# 11. Evaluation on Test Set
# ---------------------------

# Load the best model for evaluation
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
print("Best model loaded for evaluation.")

model.eval()
test_predictions = []

# Compute latent stats from training set
mu_list = []
logvar_list = []
with torch.no_grad():
    for batch_x, batch_cond in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_cond = batch_cond.to(device, non_blocking=True)
        mu, logvar = model.encode(batch_x)
        mu_list.append(mu.cpu())
        logvar_list.append(logvar.cpu())
    mu_train = torch.cat(mu_list, dim=0)
    logvar_train = torch.cat(logvar_list, dim=0)
    mu_train_mean = mu_train.mean(dim=0)
    mu_train_std = mu_train.std(dim=0)

    # Save latent stats
    torch.save({'mu_train_mean': mu_train_mean, 'mu_train_std': mu_train_std}, LATENT_STATS_PATH)
    print(f"Latent stats saved to {LATENT_STATS_PATH}")

    # Sample z and decode in batches
    z_sample = torch.randn(len(test_momenta_norm), LATENT_DIM)
    z_sample = z_sample * mu_train_std.unsqueeze(0) + mu_train_mean.unsqueeze(0)

    # Move z_sample to CPU to avoid GPU memory issues
    z_sample = z_sample.cpu()

    # Process test data in batches
    for batch_idx, (batch_x, batch_cond) in enumerate(test_loader):
        batch_size = batch_x.size(0)
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + batch_size
        batch_z = z_sample[start_idx:end_idx].to(device, non_blocking=True)
        batch_cond = batch_cond.to(device, non_blocking=True)
        with autocast():  # Removed device_type argument
            pred = model.decode(batch_z, batch_cond)
        test_predictions.append(pred.cpu())
test_predictions = torch.cat(test_predictions, dim=0)

# Inverse transform
if position_scaler:
    test_predictions_inv = position_scaler.inverse_transform(test_predictions.numpy())
    test_position_inv = position_scaler.inverse_transform(test_position.numpy())
else:
    test_predictions_inv = test_predictions.numpy()
    test_position_inv = test_position.numpy()

# Calculate MRE
relative_errors = np.abs(test_predictions_inv - test_position_inv) / (np.abs(test_position_inv) + 1e-8)
mre = np.mean(relative_errors)

# Calculate MSE
mse = np.mean((test_predictions_inv - test_position_inv) ** 2)

print(f"Test MRE: {mre}")
print(f"Test MSE: {mse}")

# Save both metrics to a JSON file
results = {
    'mre': float(mre),  # Convert numpy.float32 to Python float
    'mse': float(mse)   # Convert numpy.float32 to Python float
}

with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {RESULTS_PATH}")

# Calculate and print component-wise MSE
component_names = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
component_mse = np.mean((test_predictions_inv - test_position_inv) ** 2, axis=0)

print("\nComponent-wise MSE:")
for name, mse_value in zip(component_names, component_mse):
    print(f"{name}: {mse_value:.6f}")

# Save detailed results including component-wise MSE
detailed_results = {
    'overall_mre': float(mre),
    'overall_mse': float(mse),
    'component_mse': {name: float(mse_value) for name, mse_value in zip(component_names, component_mse)}
}

with open(DETAILED_RESULTS_PATH, 'w') as f:
    json.dump(detailed_results, f, indent=4)
print(f"Detailed results saved to {DETAILED_RESULTS_PATH}")

# ---------------------------
# 12. Plotting Learning Curves
# ---------------------------

# Ensure that matplotlib does not try to open any window
plt.switch_backend('Agg')

# Plot first 10 epochs
plt.figure(figsize=(10, 6))
epochs_first = range(1, min(11, len(train_losses)+1))
plt.plot(epochs_first, train_losses[:10], label='Train Loss')
plt.plot(epochs_first, val_losses[:10], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss - First 10 Epochs')
plt.legend()
plt.grid(True)
plt.savefig(FIRST_EPOCH_PLOT_PATH)
plt.close()

# Plot remaining epochs
if len(train_losses) > 10:
    plt.figure(figsize=(10, 6))
    epochs_rest = range(11, len(train_losses)+1)
    plt.plot(epochs_rest, train_losses[10:], label='Train Loss')
    plt.plot(epochs_rest, val_losses[10:], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss - Remaining Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(REST_EPOCH_PLOT_PATH)
    plt.close()

# ---------------------------
# 13. Model Summaries
# ---------------------------

print("\n--- Model Summary using torchsummary ---")
try:
    torchsummary.summary(model, input_size=[(9,), (9,)])
except Exception as e:
    print(f"torchsummary encountered an error: {e}")

print("\n--- Model Summary using torchinfo ---")
try:
    x_dummy = torch.randn(1, 9).to(device)
    cond_dummy = torch.randn(1, 9).to(device)
    torchinfo.summary(model, input_data=(x_dummy, cond_dummy), device=device)
except Exception as e:
    print(f"torchinfo encountered an error: {e}")
