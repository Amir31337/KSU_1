'''
===== Cross-Validation Results =====
Fold 1: MSE = 0.805270, MRE = 1.829406
Fold 2: MSE = 0.821108, MRE = 1.195287
Fold 3: MSE = 0.811052, MRE = 2.161192
Fold 4: MSE = 0.793801, MRE = 1.113001
Fold 5: MSE = 0.818063, MRE = 1.330502

Average MSE across all folds: 0.809859 ± 0.009738
Average MRE across all folds: 1.525878 ± 0.403406
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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
HIDDEN_DIM = 174      # Increased hidden layer dimension
LATENT_DIM = 20       # Increased latent space dimension
OUTPUT_DIM = 9        # Dimension of X for reconstruction

# Training parameters
BATCH_SIZE = 128
N_EPOCHS = 224
LEARNING_RATE = 0.00036324869566766035
RANDOM_SEED = 42
N_FOLDS = 5

# Activation functions
ACTIVATION = F.tanh

# Loss function
RECONSTRUCTION_LOSS = F.mse_loss  # Mean Squared Error

# Misc
PATIENCE = 40  # For early stopping
SAVE_DIR = "cvae_kfold_models"  # Directory to save models and latent stats
os.makedirs(SAVE_DIR, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# -------------------------------------------------------------

# --------------------- Data Loading & Preprocessing --------------------------
# Load data
data = pd.read_csv(DATA_PATH)
X = data[POSITION_COLUMNS].values.astype(np.float32)
Y = data[MOMENTA_COLUMNS].values.astype(np.float32)
# -----------------------------------------------------------------------------

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
def train_epoch(model, optimizer, loader):
    model.train()
    train_loss = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        optimizer.zero_grad()
        reconstructed_x, mu, log_var = model(x, y)
        loss = calculate_loss(x, reconstructed_x, mu, log_var)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    return train_loss

def validate_epoch(model, loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            reconstructed_x, mu, log_var = model(x, y)
            loss = calculate_loss(x, reconstructed_x, mu, log_var)
            val_loss += loss.item()
    return val_loss
# -------------------------------------------------------------

# --------------------- Latent Statistics Computation ---------------------
def compute_latent_statistics(model, loader):
    """
    Computes the mean and log variance of the latent space from the training data.
    Also collects z_train by reparameterizing mu and log_var.
    """
    model.eval()
    mu_sum = 0
    log_var_sum = 0
    z_list = []
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            mu, log_var = model.encoder(x, y)
            mu_sum += mu.sum(dim=0)
            log_var_sum += log_var.sum(dim=0)
            z = model.reparameterize(mu, log_var)
            z_list.append(z.cpu())
            total_samples += x.size(0)
    
    mu_mean = mu_sum / total_samples
    log_var_mean = log_var_sum / total_samples
    z_train = torch.cat(z_list, dim=0)
    return mu_mean, log_var_mean, z_train
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
            z = torch.from_numpy(mu_train_mean_np).float().to(DEVICE) + \
                eps * torch.from_numpy(sigma_train).float().to(DEVICE)
            
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

# --------------------- Cross-Validation Setup ---------------------
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# Lists to store metrics for each fold
fold_mse = []
fold_mre = []

for fold, (train_val_idx, test_idx) in enumerate(kf.split(X), 1):
    print(f'\n===== Fold {fold} / {N_FOLDS} =====')
    
    # Split data
    X_train_val, X_test = X[train_val_idx], X[test_idx]
    Y_train_val, Y_test = Y[train_val_idx], Y[test_idx]
    
    # Further split train_val into train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=0.15, random_state=RANDOM_SEED)
    
    # Normalize the data using StandardScaler fitted on training data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    Y_train = scaler_Y.fit_transform(Y_train)
    Y_val = scaler_Y.transform(Y_val)
    Y_test_scaled = scaler_Y.transform(Y_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    # --------------------- Model Initialization ---------------------
    model = CVAE(
        input_dim=INPUT_DIM,
        condition_dim=CONDITION_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        output_dim=OUTPUT_DIM
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # -------------------------------------------------------------
    
    # --------------------- Training Loop -------------------------
    # Lists to store losses
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, train_loader)
        val_loss = validate_epoch(model, val_loader)
        
        train_losses.append(train_loss / len(train_loader.dataset))
        val_losses.append(val_loss / len(val_loader.dataset))
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model for this fold
            model_path = os.path.join(SAVE_DIR, f"MoS2_cvae_fold{fold}.pt")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
    
    # Load the best model for this fold
    model.load_state_dict(torch.load(model_path))
    # -------------------------------------------------------------
    
    # --------------------- Compute Latent Statistics ----------------------
    mu_train_mean, log_var_train_mean, z_train = compute_latent_statistics(model, train_loader)
    
    # Compute standard deviation
    sigma_train_mean = torch.exp(0.5 * log_var_train_mean)
    
    # Save latent statistics and z_train
    latent_stats = {
        'mu_train_mean': mu_train_mean.cpu(),
        'sigma_train_mean': sigma_train_mean.cpu(),
        'z_train': z_train.cpu()
    }
    latent_stats_path = os.path.join(SAVE_DIR, f"latent_stats_fold{fold}.pt")
    torch.save(latent_stats, latent_stats_path)
    print(f'Latent statistics saved to {latent_stats_path}')
    # ----------------------------------------------------------------------
    
    # --------------------- Testing Phase --------------------------
    average_relative_error, average_mse = compute_metrics(
        model, test_loader, scaler_X, mu_train_mean, log_var_train_mean)
    
    print(f'Fold {fold} - Average Relative Error on Test Set: {average_relative_error:.6f}')
    print(f'Fold {fold} - Average MSE on Test Set: {average_mse:.6f}')
    
    fold_mse.append(average_mse)
    fold_mre.append(average_relative_error)
    # -------------------------------------------------------------
    
    # --------------------- Plot Loss Curves ----------------------
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss per Epoch - Fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(SAVE_DIR, f"loss_curve_fold{fold}.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f'Loss curves saved to {loss_plot_path}')
    # -------------------------------------------------------------

# --------------------- Aggregate Metrics -------------------------
mean_mse = np.mean(fold_mse)
std_mse = np.std(fold_mse)
mean_mre = np.mean(fold_mre)
std_mre = np.std(fold_mre)

print('\n===== Cross-Validation Results =====')
for fold in range(1, N_FOLDS + 1):
    print(f'Fold {fold}: MSE = {fold_mse[fold-1]:.6f}, MRE = {fold_mre[fold-1]:.6f}')
print(f'\nAverage MSE across all folds: {mean_mse:.6f} ± {std_mse:.6f}')
print(f'Average MRE across all folds: {mean_mre:.6f} ± {std_mre:.6f}')
# -------------------------------------------------------------
