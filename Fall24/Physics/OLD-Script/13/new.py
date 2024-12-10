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
import json
import torchsummary

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
FILEPATH = 'cei_traning_orient_1.csv'
data = pd.read_csv(FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Split data first
position_train, position_temp, momenta_train, momenta_temp = train_test_split(
    position, momenta, test_size=0.3, random_state=42, shuffle=True
)

position_val, position_test, momenta_val, momenta_test = train_test_split(
    position_temp, momenta_temp, test_size=0.5, random_state=42, shuffle=True
)

# Initialize scalers
position_scaler = StandardScaler()
momenta_scaler = StandardScaler()

# Fit scalers only on training data
position_train_norm = position_scaler.fit_transform(position_train)
momenta_train_norm = momenta_scaler.fit_transform(momenta_train)

# Transform validation and test sets using training set statistics
position_val_norm = position_scaler.transform(position_val)
position_test_norm = position_scaler.transform(position_test)
momenta_val_norm = momenta_scaler.transform(momenta_val)
momenta_test_norm = momenta_scaler.transform(momenta_test)

# Convert to PyTorch tensors
train_position = torch.FloatTensor(position_train_norm).to(device)
val_position = torch.FloatTensor(position_val_norm).to(device)
test_position = torch.FloatTensor(position_test_norm).to(device)
train_momenta = torch.FloatTensor(momenta_train_norm).to(device)
val_momenta = torch.FloatTensor(momenta_val_norm).to(device)
test_momenta = torch.FloatTensor(momenta_test_norm).to(device)

# Model configuration
INPUT_DIM = position.shape[1]
OUTPUT_DIM = momenta.shape[1]
LATENT_DIM = 1024
EPOCHS = 200
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
BETA = 0.01  # KL divergence weight
PATIENCE = 20
MIN_DELTA = 1e-6

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim):
        super(CVAE, self).__init__()
        
        # Encoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024)
        )
        
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        
        # Decoder architecture
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z, condition):
        combined = torch.cat((z, condition), dim=1)
        return self.decoder(combined)

    def forward(self, x, condition):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, condition), mu, logvar

# Initialize model and optimizer
model = CVAE(INPUT_DIM, LATENT_DIM, OUTPUT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def loss_function(recon_x, x, mu, logvar, batch_size):
    # Reconstruction loss (MSE)
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum') / batch_size
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
    return MSE + BETA * KLD, MSE, KLD

# Create DataLoaders
train_dataset = TensorDataset(train_position, train_momenta)
val_dataset = TensorDataset(val_position, val_momenta)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training loop with proper loss calculation
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    train_mse = 0.0
    train_kld = 0.0
    num_batches = 0
    
    for batch_x, batch_cond in train_loader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch_x, batch_cond)
        loss, mse, kld = loss_function(recon_batch, batch_x, mu, logvar, batch_x.size(0))
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_mse += mse.item()
        train_kld += kld.item()
        num_batches += 1
    
    avg_train_loss = train_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_mse = 0.0
    val_kld = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_cond in val_loader:
            recon_batch, mu, logvar = model(batch_x, batch_cond)
            loss, mse, kld = loss_function(recon_batch, batch_x, mu, logvar, batch_x.size(0))
            
            val_loss += loss.item()
            val_mse += mse.item()
            val_kld += kld.item()
            num_val_batches += 1
    
    avg_val_loss = val_loss / num_val_batches
    val_losses.append(avg_val_loss)
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}]')
        print(f'Train Loss: {avg_train_loss:.6f} (MSE: {train_mse/num_batches:.6f}, KLD: {train_kld/num_batches:.6f})')
        print(f'Val Loss: {avg_val_loss:.6f} (MSE: {val_mse/num_val_batches:.6f}, KLD: {val_kld/num_val_batches:.6f})')
    
    # Early stopping
    if avg_val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# Testing and MRE calculation
model.eval()
test_predictions = []
with torch.no_grad():
    for i in range(len(test_momenta)):
        mu_test, logvar_test = model.encode(test_position[i:i+1])
        z_test = model.reparameterize(mu_test, logvar_test)
        pred = model.decode(z_test, test_momenta[i:i+1])
        test_predictions.append(pred)

test_predictions = torch.cat(test_predictions, dim=0)

# Inverse transform predictions and actual values
test_predictions_inv = position_scaler.inverse_transform(test_predictions.cpu().numpy())
test_position_inv = position_scaler.inverse_transform(test_position.cpu().numpy())

# Calculate MRE
relative_errors = np.abs(test_predictions_inv - test_position_inv) / (np.abs(test_position_inv) + 1e-8)
mre = np.mean(relative_errors)

print(f"Test MRE: {mre}")

# Save results
results = {'mre': float(mre)}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('learning_curves.png')
plt.close()

# Model summary
print("\nModel Summary:")
torchsummary.summary(model, input_size=[(9,), (9,)])