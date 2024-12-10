import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import json
import torchsummary
import torchinfo

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
FILEPATH = 'cei_traning_orient_1.csv'
data = pd.read_csv(FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Split data into train, validation, and test sets (70%, 15%, 15%)
train_position, temp_position, train_momenta, temp_momenta = train_test_split(
    position, momenta, test_size=0.3, random_state=42, shuffle=True
)

val_position, test_position, val_momenta, test_momenta = train_test_split(
    temp_position, temp_momenta, test_size=0.5, random_state=42, shuffle=True
)

# Convert to PyTorch tensors
train_position = torch.FloatTensor(train_position).to(device)
val_position = torch.FloatTensor(val_position).to(device)
test_position = torch.FloatTensor(test_position).to(device)
train_momenta = torch.FloatTensor(train_momenta).to(device)
val_momenta = torch.FloatTensor(val_momenta).to(device)
test_momenta = torch.FloatTensor(test_momenta).to(device)

INPUT_DIM = position.shape[1]  # 9
OUTPUT_DIM = momenta.shape[1]  # 9

# Set hyperparameters
LATENT_DIM = 2048
EPOCHS = 300
BATCH_SIZE = 1024
LEARNING_RATE = 0.0001
PATIENCE = 100
MIN_DELTA = 1e-5
activation_name = 'Tanh'
position_norm_method = 'StandardScaler'
momenta_norm_method = 'StandardScaler'
use_l1 = True
L1_LAMBDA = 0.01
use_l2 = True
L2_LAMBDA = 0.01
num_hidden_layers = 4
hidden_layer_size = 64
N_SAMPLES_INFERENCE = 5  # Number of samples to average during inference

activation_function = getattr(nn, activation_name)()

# Normalization methods
if position_norm_method == 'StandardScaler':
    position_scaler = StandardScaler()

if momenta_norm_method == 'StandardScaler':
    momenta_scaler = StandardScaler()
elif momenta_norm_method == 'None':
    momenta_scaler = None

# Normalize the data
if position_scaler is not None:
    train_position_norm = torch.FloatTensor(position_scaler.fit_transform(train_position.cpu())).to(device)
    val_position_norm = torch.FloatTensor(position_scaler.transform(val_position.cpu())).to(device)
    test_position_norm = torch.FloatTensor(position_scaler.transform(test_position.cpu())).to(device)
else:
    train_position_norm = train_position
    val_position_norm = val_position
    test_position_norm = test_position

if momenta_scaler is not None:
    train_momenta_norm = torch.FloatTensor(momenta_scaler.fit_transform(train_momenta.cpu())).to(device)
    val_momenta_norm = torch.FloatTensor(momenta_scaler.transform(val_momenta.cpu())).to(device)
    test_momenta_norm = torch.FloatTensor(momenta_scaler.transform(test_momenta.cpu())).to(device)
else:
    train_momenta_norm = train_momenta
    val_momenta_norm = val_momenta
    test_momenta_norm = test_momenta

# Hidden layers configuration
hidden_layers = [hidden_layer_size * (2 ** i) for i in range(num_hidden_layers)]

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

# Initialize model
model = CVAE(INPUT_DIM, LATENT_DIM, OUTPUT_DIM, hidden_layers, activation_function).to(device)

# Optimizer with L2 regularization
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA if use_l2 else 0.0)

# Loss function
def loss_fn(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_divergence /= x.size(0) * x.size(1)
    
    loss = recon_loss + kl_divergence
    
    if use_l1:
        l1_loss = 0.0
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        loss += L1_LAMBDA * l1_loss
    
    return loss

# Create DataLoaders
train_loader = DataLoader(TensorDataset(train_position_norm, train_momenta_norm), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_position_norm, val_momenta_norm), batch_size=BATCH_SIZE, shuffle=False)

# Training
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    for batch_x, batch_cond in train_loader:
        optimizer.zero_grad()
        recon_x, mu, logvar = model(batch_x, batch_cond)
        loss = loss_fn(recon_x, batch_x, mu, logvar)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_cond in val_loader:
            recon_x, mu, logvar = model(batch_x, batch_cond)
            loss = loss_fn(recon_x, batch_x, mu, logvar)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    # Early stopping
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(torch.load('best_model.pth', map_location=device))

# Compute MRE on test set with proper inference
model.eval()
test_predictions = []

with torch.no_grad():
    # First compute latent distribution parameters from training data
    mu_list = []
    logvar_list = []
    for batch_x, _ in train_loader:
        mu, logvar = model.encode(batch_x)
        mu_list.append(mu)
        logvar_list.append(logvar)
    
    # Calculate mean and std of the latent space from training data
    mu_train = torch.cat(mu_list, dim=0)
    logvar_train = torch.cat(logvar_list, dim=0)
    mu_mean = mu_train.mean(dim=0)
    mu_std = mu_train.std(dim=0)
    
    # For each test sample, sample from the training latent distribution
    for test_cond in test_momenta_norm:
        test_cond = test_cond.unsqueeze(0)
        
        # Sample multiple times and average predictions
        samples = []
        for _ in range(N_SAMPLES_INFERENCE):
            # Sample z from the training distribution
            z = torch.randn(1, LATENT_DIM, device=device) * mu_std + mu_mean
            
            # Decode with the condition
            pred = model.decode(z, test_cond)
            samples.append(pred)
        
        # Average the predictions
        avg_pred = torch.mean(torch.stack(samples, dim=0), dim=0)
        test_predictions.append(avg_pred)

test_predictions = torch.cat(test_predictions, dim=0)

# Inverse transform predictions and compute MRE
if position_scaler is not None:
    test_predictions_inv = position_scaler.inverse_transform(test_predictions.cpu().numpy())
    test_position_inv = position_scaler.inverse_transform(test_position.cpu().numpy())
else:
    test_predictions_inv = test_predictions.cpu().numpy()
    test_position_inv = test_position.cpu().numpy()

relative_errors = np.abs(test_predictions_inv - test_position_inv) / (np.abs(test_position_inv) + 1e-8)
mre = np.mean(relative_errors)

print(f"Test MRE: {mre}")

# Save results
results = {
    'mre': float(mre)
}

with open('RAWresults.json', 'w') as f:
    json.dump(results, f, indent=4)

# Print random test examples
def print_random_test_examples(test_positions, test_momenta, test_predictions, num_examples=5):
    np.random.seed(42)
    total_examples = test_positions.shape[0]
    random_indices = np.random.choice(total_examples, size=num_examples, replace=False)
    
    for idx in random_indices:
        formatted_position = ' '.join([f"{val:.2f}" for val in test_positions[idx]])
        formatted_momenta = ' '.join([f"{val:.2f}" for val in test_momenta[idx]])
        formatted_reconstructed = ' '.join([f"{val:.2f}" for val in test_predictions[idx]])
        
        print(f"Original Position: {formatted_position}")
        print(f"Momenta: {formatted_momenta}")
        print(f"Reconstructed Position: {formatted_reconstructed}\n")

print_random_test_examples(test_position_inv, test_momenta.cpu().numpy(), test_predictions_inv)

# Plot learning curves
plt.switch_backend('Agg')

# First 10 epochs
plt.figure(figsize=(10, 6))
epochs_first = range(1, min(11, len(train_losses)+1))
plt.plot(epochs_first, train_losses[:10], label='Train Loss')
plt.plot(epochs_first, val_losses[:10], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss - First 10 Epochs')
plt.legend()
plt.grid(True)
plt.savefig('RAW-first.png')
plt.close()

# Remaining epochs
if len(train_losses) > 10:
    plt.figure(figsize=(30, 18))
    epochs_rest = range(11, len(train_losses)+1)
    plt.plot(epochs_rest, train_losses[10:], label='Train Loss')
    plt.plot(epochs_rest, val_losses[10:], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss - Remaining Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('RAW-rest.png')
    plt.close()