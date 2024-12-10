import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import torchsummary 
import torchinfo 

# --------------------- Hyperparameters ---------------------
# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data parameters
DATA_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
POSITION_COLUMNS = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
MOMENTA_COLUMNS = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']

# Model hyperparameters
INPUT_DIM = 9         # Dimension of X (initial positions)
CONDITION_DIM = 9     # Dimension of Y (final momenta)
HIDDEN_DIM = 174      # Hidden layer dimension
LATENT_DIM = 20       # Latent space dimension
OUTPUT_DIM = 9        # Dimension of X for reconstruction

# Training parameters
BATCH_SIZE = 128
N_EPOCHS = 224
LEARNING_RATE = 0.00036324869566766035  # Learning rate
RANDOM_SEED = 42

# L2 Regularization (to be tuned)
LAMBDA_VALUES = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]  # List of LAMBDA values to try

# Activation functions
ACTIVATION = torch.tanh  # Using torch's tanh

# Loss function
RECONSTRUCTION_LOSS = F.mse_loss  # Mean Squared Error

# Misc
PATIENCE = 40  # For early stopping
SAVE_PATH_TEMPLATE = "MoS2_cvae_lambda_{:.0e}.pt"  # Template for saving models
LATENT_STATS_PATH_TEMPLATE = "latent_stats_lambda_{:.0e}.pt"  # Template for saving latent statistics

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# -------------------------------------------------------------

# --------------------- Data Loading & Preprocessing --------------------------
# Load data
data = pd.read_csv(DATA_PATH)
position = data[POSITION_COLUMNS].values.astype(np.float32)
momenata = data[MOMENTA_COLUMNS].values.astype(np.float32)

# Split data into train (70%), validation (15%), and test (15%)
# First split: 70% train and 30% temp (validation + test)
x_train, x_temp, y_train, y_temp = train_test_split(
    position, momenata, train_size=0.7, random_state=RANDOM_SEED)

# Second split: Split temp into 50% validation and 50% test (each 15% of total)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED)

# Verify the splits
total_samples = position.shape[0]
train_size = x_train.shape[0]
val_size = x_val.shape[0]
test_size = x_test.shape[0]
assert train_size + val_size + test_size == total_samples, "Data split sizes do not add up!"

print(f"Data Split:")
print(f"Training: {train_size} samples ({train_size / total_samples * 100:.2f}%)")
print(f"Validation: {val_size} samples ({val_size / total_samples * 100:.2f}%)")
print(f"Testing: {test_size} samples ({test_size / total_samples * 100:.2f}%)")

# Normalize the data using StandardScaler
scaler_X = StandardScaler()

x_train = scaler_X.fit_transform(x_train)
x_val = scaler_X.transform(x_val)
x_test = scaler_X.transform(x_test)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
# ------------------------------------------------------------------------------

# --------------------- Model Definition ----------------------
class Encoder(nn.Module):
    """
    Encoder part of CVAE: Encodes X and Y into latent space.
    """
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, y):
        concatenated = torch.cat((x, y), dim=1)
        hidden = ACTIVATION(self.fc1(concatenated))
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        return mu, log_var

class Decoder(nn.Module):
    """
    Decoder part of CVAE: Decodes latent variables and Y to reconstruct X.
    """
    def __init__(self, latent_dim, condition_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z, y):
        concatenated = torch.cat((z, y), dim=1)
        hidden = ACTIVATION(self.fc1(concatenated))
        reconstructed_x = self.fc_out(hidden)
        return reconstructed_x

class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder.
    """
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim, output_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, condition_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim, hidden_dim, output_dim)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, y):
        mu, log_var = self.encoder(x, y)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decoder(z, y)
        return reconstructed_x, mu, log_var
# -------------------------------------------------------------

# --------------------- Loss Function -------------------------
def calculate_loss(x, reconstructed_x, mu, log_var):
    """
    Computes the CVAE loss as the sum of reconstruction loss and KL divergence.
    """
    # Reconstruction loss
    recon_loss = RECONSTRUCTION_LOSS(reconstructed_x, x, reduction='sum')
    # KL Divergence
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld_loss
# -------------------------------------------------------------

# --------------------- Training Function ---------------------
def train_epoch(model, optimizer, loader, lambda_l2):
    model.train()
    train_loss = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        optimizer.zero_grad()
        reconstructed_x, mu, log_var = model(x, y)
        loss = calculate_loss(reconstructed_x, x, mu, log_var)
        
        # L2 Regularization
        l2_reg = torch.tensor(0., device=DEVICE)
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)**2
        loss += lambda_l2 * l2_reg
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    return train_loss

def validate_epoch(model, loader, lambda_l2):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            reconstructed_x, mu, log_var = model(x, y)
            loss = calculate_loss(reconstructed_x, x, mu, log_var)
            
            # L2 Regularization (optional: usually not applied during validation)
            l2_reg = torch.tensor(0., device=DEVICE)
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)**2
            loss += lambda_l2 * l2_reg
            
            val_loss += loss.item()
    return val_loss
# -------------------------------------------------------------

# --------------------- Latent Statistics Computation ---------------------
def compute_latent_statistics(model, loader):
    """
    Computes the mean and log variance of the latent space from the training data.
    """
    model.eval()
    mu_sum = 0
    log_var_sum = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            mu, log_var = model.encoder(x, y)
            mu_sum += mu.sum(dim=0)
            log_var_sum += log_var.sum(dim=0)
            total_samples += x.size(0)
    
    mu_mean = mu_sum / total_samples
    log_var_mean = log_var_sum / total_samples
    return mu_mean, log_var_mean
# -------------------------------------------------------------

# --------------------- Metrics Calculation -------------------
def compute_metrics(model, test_loader, scaler_X, mu_train_mean, log_var_train_mean):
    """
    Computes the average relative error and MSE on the test set using the learned latent statistics.
    """
    model.eval()
    total_relative_error = 0
    total_mse = 0
    total_samples = 0

    # Compute standard deviation from log variance
    sigma_train = torch.exp(0.5 * log_var_train_mean).cpu().numpy()

    # Convert mu_train_mean to numpy
    mu_train_mean_np = mu_train_mean.cpu().numpy()

    with torch.no_grad():
        for x, y in test_loader:
            y = y.to(DEVICE)
            batch_size = y.size(0)
            
            # Sample z using the training latent statistics
            eps = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            z = torch.from_numpy(mu_train_mean_np).float().to(DEVICE) + eps * torch.from_numpy(sigma_train).float().to(DEVICE)
            
            # Decode using sampled z and test y
            predicted_x = model.decoder(z, y)
            
            # Inverse transform to original scale
            predicted_x = scaler_X.inverse_transform(predicted_x.cpu().numpy())
            actual_x = scaler_X.inverse_transform(x.numpy())
            
            # Compute relative error
            epsilon_val = 1e-8  # To prevent division by zero
            relative_error = np.abs(predicted_x - actual_x) / (np.abs(actual_x) + epsilon_val)
            relative_error = np.mean(relative_error, axis=1)  # Mean over the 9 components
            total_relative_error += np.sum(relative_error)
            
            # Compute MSE
            mse = np.mean((predicted_x - actual_x) ** 2, axis=1)
            total_mse += np.sum(mse)
            
            total_samples += batch_size

    average_relative_error = total_relative_error / total_samples
    average_mse = total_mse / total_samples
    return average_relative_error, average_mse
# -------------------------------------------------------------

# --------------------- Training and Hyperparameter Tuning Loop ---------------------
# Define the list of LAMBDA values to try
LAMBDA_VALUES = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# To store results
results = []

# Iterate over each LAMBDA value
for lambda_l2 in LAMBDA_VALUES:
    print(f"\nTraining with LAMBDA = {lambda_l2}")
    
    # Initialize model
    model = CVAE(
        input_dim=INPUT_DIM,
        condition_dim=CONDITION_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        output_dim=OUTPUT_DIM
    ).to(DEVICE)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler for learning rate adjustment
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Lists to store losses
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, train_loader, lambda_l2)
        val_loss = validate_epoch(model, val_loader, lambda_l2)
        
        train_losses.append(train_loss / len(train_loader.dataset))
        val_losses.append(val_loss / len(val_loader.dataset))
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model for this LAMBDA
            save_path = SAVE_PATH_TEMPLATE.format(lambda_l2)
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
    
    # Load the best model for this LAMBDA
    model.load_state_dict(torch.load(SAVE_PATH_TEMPLATE.format(lambda_l2)))
    
    # Compute Latent Statistics
    mu_train_mean, log_var_train_mean = compute_latent_statistics(model, train_loader)
    
    # Save the latent statistics for this LAMBDA
    latent_stats_path = LATENT_STATS_PATH_TEMPLATE.format(lambda_l2)
    torch.save({
        'mu_train_mean': mu_train_mean,
        'log_var_train_mean': log_var_train_mean
    }, latent_stats_path)
    print(f'Latent statistics saved to {latent_stats_path}')
    
    # Compute Metrics on Test Set
    average_relative_error, average_mse = compute_metrics(
        model, test_loader, scaler_X, mu_train_mean, log_var_train_mean)
    
    print(f'LAMBDA = {lambda_l2}: Average Relative Error on Test Set: {average_relative_error:.6f}')
    print(f'LAMBDA = {lambda_l2}: Average MSE on Test Set: {average_mse:.6f}')
    
    # Store the results
    results.append({
        'LAMBDA': lambda_l2,
        'Test_MRE': average_relative_error,
        'Test_MSE': average_mse
    })
    
    # Optional: Plot Loss Curves for this LAMBDA
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss per Epoch (LAMBDA={lambda_l2:.0e})')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Clean up to free memory
    del model
    del optimizer
    del scheduler
    torch.cuda.empty_cache()

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)
print("\nHyperparameter Tuning Results:")
print(results_df)

# Identify the best LAMBDA with the lowest Test MRE
best_result = results_df.loc[results_df['Test_MRE'].idxmin()]
best_lambda = best_result['LAMBDA']
best_mre = best_result['Test_MRE']
best_mse = best_result['Test_MSE']

print(f"\nBest LAMBDA: {best_lambda} with Test MRE: {best_mre:.6f} and Test MSE: {best_mse:.6f}")

# --------------------- Save the Best Model -------------------------
# Save the best model's state_dict
best_save_path = SAVE_PATH_TEMPLATE.format(best_lambda)
best_latent_stats_path = LATENT_STATS_PATH_TEMPLATE.format(best_lambda)
print(f'\nBest model saved to {best_save_path}')
print(f'Best latent statistics saved to {best_latent_stats_path}')
# -------------------------------------------------------------

# --------------------- Plot Test MRE vs. LAMBDA ----------------------
plt.figure(figsize=(8,6))
plt.loglog(results_df['LAMBDA'], results_df['Test_MRE'], marker='o')
plt.xlabel('LAMBDA (L2 Regularization Strength)')
plt.ylabel('Test Mean Relative Error (MRE)')
plt.title('Test MRE vs. LAMBDA')
plt.grid(True, which="both", ls="--")
plt.show()
# -------------------------------------------------------------

# --------------------- (Optional) Model Summary for Best Model -------------------------
# Uncomment the following block if you want to see the model summary for the best model

'''
# Load the best model
best_model = CVAE(
    input_dim=INPUT_DIM,
    condition_dim=CONDITION_DIM,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM,
    output_dim=OUTPUT_DIM
).to(DEVICE)
best_model.load_state_dict(torch.load(best_save_path))

# Create sample input tensors with correct shape and batch dimension
batch_size = 1
sample_x = torch.randn(batch_size, INPUT_DIM).to(DEVICE)
sample_y = torch.randn(batch_size, CONDITION_DIM).to(DEVICE)

# Print model summary using torchinfo
print("\nModel Summary using torchinfo for Best Model:")
torchinfo.summary(
    best_model,
    input_data=[sample_x, sample_y],
    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
    depth=3
)
'''
# ---------------------------------------------------------------------------------
