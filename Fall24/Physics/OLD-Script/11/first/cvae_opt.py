'''
Value (MRE): 0.6723
INFO:__main__:  Params:
INFO:__main__:    hidden_dim: 1152
INFO:__main__:    latent_dim: 1920
INFO:__main__:    batch_size: 128
INFO:__main__:    n_epochs: 182
INFO:__main__:    learning_rate: 5.772971044368426e-05
INFO:__main__:    activation: ReLU
INFO:__main__:    patience: 7
INFO:__main__:    position_norm: standard
INFO:__main__:    momenta_norm: none

INFO:__main__:Final Test MSE: 0.7288
INFO:__main__:Final Test MRE: 0.6848
'''
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
import warnings
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set up logging for Optuna
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clear GPU memory
torch.cuda.empty_cache()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CVAE model
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_dims, activation_fn):
        super(CVAE, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(activation_fn)  # Removed ()
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim + condition_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(activation_fn)  # Removed ()
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
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

# Define the loss function for CVAE
def cvae_loss_fn(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_divergence

# Define normalization methods mapping
NORMALIZATION_METHODS = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler,
    'none': None
}

# Define activation functions mapping
ACTIVATION_FUNCTIONS = {  # Corrected dictionary name
    'ELU': nn.ELU(),
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU()
}

# Filepath to the dataset
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'

# Function to load and preprocess data
def load_and_preprocess_data(position_norm, momenta_norm):
    data = pd.read_csv(FILEPATH)
    position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
    momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values
    
    # Apply normalization
    if position_norm != 'none':
        position_scaler = NORMALIZATION_METHODS[position_norm]()
        position_normalized = position_scaler.fit_transform(position)
    else:
        position_normalized = position.copy()
        position_scaler = None
    
    if momenta_norm != 'none':
        momenta_scaler = NORMALIZATION_METHODS[momenta_norm]()
        momenta_normalized = momenta_scaler.fit_transform(momenta)
    else:
        momenta_normalized = momenta.copy()
        momenta_scaler = None
    
    return position_normalized, momenta_normalized, position_scaler, momenta_scaler

# Define the objective function for Optuna
def objective(trial):
    # ============================
    # Hyperparameter Sampling
    # ============================
    hidden_dim = trial.suggest_int('hidden_dim', 256, 2048, step=128)
    latent_dim = trial.suggest_int('latent_dim', 128, 4096, step=128)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    n_epochs = trial.suggest_int('n_epochs', 50, 200)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    activation_name = trial.suggest_categorical('activation', ['ELU', 'ReLU', 'LeakyReLU'])
    patience = trial.suggest_int('patience', 5, 20)
    position_norm = trial.suggest_categorical('position_norm', ['standard', 'minmax', 'robust', 'none'])
    momenta_norm = trial.suggest_categorical('momenta_norm', ['standard', 'minmax', 'robust', 'none'])
    
    # Log the sampled hyperparameters
    logger.info(f"Trial {trial.number}:")
    logger.info(f"  hidden_dim: {hidden_dim}")
    logger.info(f"  latent_dim: {latent_dim}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  n_epochs: {n_epochs}")
    logger.info(f"  learning_rate: {learning_rate}")
    logger.info(f"  activation: {activation_name}")
    logger.info(f"  patience: {patience}")
    logger.info(f"  position_norm: {position_norm}")
    logger.info(f"  momenta_norm: {momenta_norm}")
    
    # ============================
    # Data Loading and Preprocessing
    # ============================
    position_normalized, momenta_normalized, position_scaler, momenta_scaler = load_and_preprocess_data(position_norm, momenta_norm)
    
    # ============================
    # Data Splitting
    # ============================
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        position_normalized, momenta_normalized, test_size=0.15, random_state=42
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=0.1765, random_state=42
    )
    # This ensures 70% train, 15% val, 15% test
    
    # ============================
    # Convert to PyTorch tensors
    # ============================
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    Y_train_tensor = torch.FloatTensor(Y_train).to(device)
    Y_val_tensor = torch.FloatTensor(Y_val).to(device)
    Y_test_tensor = torch.FloatTensor(Y_test).to(device)
    
    # ============================
    # Create DataLoaders
    # ============================
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # ============================
    # Initialize Model
    # ============================
    activation_fn = ACTIVATION_FUNCTIONS[activation_name]  # Removed ()
    hidden_dims = [hidden_dim, hidden_dim // 2]
    model = CVAE(
        input_dim=9,
        latent_dim=latent_dim,
        condition_dim=9,
        hidden_dims=hidden_dims,
        activation_fn=activation_fn
    ).to(device)
    
    # ============================
    # Optimizer
    # ============================
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # ============================
    # Training Loop
    # ============================
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch_idx, (position, momenta) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_x, mu, logvar = model(position, momenta)
            loss = cvae_loss_fn(recon_x, position, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (position, momenta) in enumerate(val_loader):
                recon_x, mu, logvar = model(position, momenta)
                loss = cvae_loss_fn(recon_x, position, mu, logvar)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        # Early Stopping
        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_cvae_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    model.load_state_dict(torch.load('best_cvae_model.pth'))
    model.eval()
    
    # ============================
    # Save Latent Distribution Parameters
    # ============================
    with torch.no_grad():
        mu_train, logvar_train = model.encode(X_train_tensor)
        z_train = model.reparameterize(mu_train, logvar_train)
        z_train_np = z_train.cpu().numpy()
        latent_z_mean = np.mean(z_train_np, axis=0)
        latent_z_std = np.std(z_train_np, axis=0)
        np.save('latent_z_mean.npy', latent_z_mean)
        np.save('latent_z_std.npy', latent_z_std)
    
    # ============================
    # Testing Phase
    # ============================
    latent_z_mean = np.load('latent_z_mean.npy')
    latent_z_std = np.load('latent_z_std.npy')
    
    latent_z_mean_tensor = torch.tensor(latent_z_mean).to(device)
    latent_z_std_tensor = torch.tensor(latent_z_std).to(device)
    
    with torch.no_grad():
        # Sample z from the latent distribution
        z_sample = torch.randn(len(Y_test_tensor), latent_dim).to(device) * latent_z_std_tensor + latent_z_mean_tensor
        all_predicted_positions = model.decode(z_sample, Y_test_tensor)
    
    # Inverse transform the predicted and actual positions
    if position_norm != 'none' and position_scaler is not None:
        all_predicted_positions_inverse = position_scaler.inverse_transform(all_predicted_positions.cpu().numpy())
        test_position_inverse = position_scaler.inverse_transform(X_test_tensor.cpu().numpy())
    else:
        all_predicted_positions_inverse = all_predicted_positions.cpu().numpy()
        test_position_inverse = X_test_tensor.cpu().numpy()
    
    # Calculate evaluation metrics
    mse = np.mean((all_predicted_positions_inverse - test_position_inverse) ** 2)
    mae = np.mean(np.abs(all_predicted_positions_inverse - test_position_inverse))
    
    # Calculate Mean Relative Error (MRE)
    epsilon = 1e-8  # To avoid division by zero
    relative_errors = np.abs(all_predicted_positions_inverse - test_position_inverse) / (np.abs(test_position_inverse) + epsilon)
    mean_relative_error = np.mean(relative_errors)
    
    logger.info(f"Trial {trial.number} - Test MSE: {mse:.4f}, Test MRE: {mean_relative_error:.4f}")
    
    # ============================
    # Report to Optuna
    # ============================
    return mean_relative_error

# Main function to execute the optimization
def main():
    # Define the study
    study = optuna.create_study(direction='minimize', study_name='CVAE Hyperparameter Optimization')
    
    # Optimize the objective function
    study.optimize(objective, n_trials=50, timeout=None)  # Adjust n_trials as needed
    
    # Retrieve the best trial
    best_trial = study.best_trial
    
    logger.info("===== Best Trial =====")
    logger.info(f"  Value (MRE): {best_trial.value:.4f}")
    logger.info("  Params:")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # ============================
    # Retrain the model with the best hyperparameters to save loss plots
    # ============================
    logger.info("Retraining the best model to save loss plots and final metrics...")
    
    # Extract best hyperparameters
    hidden_dim = best_trial.params['hidden_dim']
    latent_dim = best_trial.params['latent_dim']
    batch_size = best_trial.params['batch_size']
    n_epochs = best_trial.params['n_epochs']
    learning_rate = best_trial.params['learning_rate']
    activation_name = best_trial.params['activation']
    patience = best_trial.params['patience']
    position_norm = best_trial.params['position_norm']
    momenta_norm = best_trial.params['momenta_norm']
    
    # Load and preprocess data
    position_normalized, momenta_normalized, position_scaler, momenta_scaler = load_and_preprocess_data(position_norm, momenta_norm)
    
    # Split data
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        position_normalized, momenta_normalized, test_size=0.15, random_state=42
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=0.1765, random_state=42
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    Y_train_tensor = torch.FloatTensor(Y_train).to(device)
    Y_val_tensor = torch.FloatTensor(Y_val).to(device)
    Y_test_tensor = torch.FloatTensor(Y_test).to(device)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    activation_fn = ACTIVATION_FUNCTIONS[activation_name]
    hidden_dims = [hidden_dim, hidden_dim // 2]
    model = CVAE(
        input_dim=9,
        latent_dim=latent_dim,
        condition_dim=9,
        hidden_dims=hidden_dims,
        activation_fn=activation_fn
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch_idx, (position, momenta) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_x, mu, logvar = model(position, momenta)
            loss = cvae_loss_fn(recon_x, position, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (position, momenta) in enumerate(val_loader):
                recon_x, mu, logvar = model(position, momenta)
                loss = cvae_loss_fn(recon_x, position, mu, logvar)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        # Early Stopping
        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_cvae_model_final.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    model.load_state_dict(torch.load('best_cvae_model_final.pth'))
    model.eval()
    
    # Save Latent Distribution Parameters
    with torch.no_grad():
        mu_train, logvar_train = model.encode(X_train_tensor)
        z_train = model.reparameterize(mu_train, logvar_train)
        z_train_np = z_train.cpu().numpy()
        latent_z_mean = np.mean(z_train_np, axis=0)
        latent_z_std = np.std(z_train_np, axis=0)
        np.save('latent_z_mean_final.npy', latent_z_mean)
        np.save('latent_z_std_final.npy', latent_z_std)
    
    # Testing Phase
    latent_z_mean = np.load('latent_z_mean_final.npy')
    latent_z_std = np.load('latent_z_std_final.npy')
    
    latent_z_mean_tensor = torch.tensor(latent_z_mean).to(device)
    latent_z_std_tensor = torch.tensor(latent_z_std).to(device)
    
    with torch.no_grad():
        # Sample z from the latent distribution
        z_sample = torch.randn(len(Y_test_tensor), latent_dim).to(device) * latent_z_std_tensor + latent_z_mean_tensor
        all_predicted_positions = model.decode(z_sample, Y_test_tensor)
    
    # Inverse transform the predicted and actual positions
    if position_norm != 'none' and position_scaler is not None:
        all_predicted_positions_inverse = position_scaler.inverse_transform(all_predicted_positions.cpu().numpy())
        test_position_inverse = position_scaler.inverse_transform(X_test_tensor.cpu().numpy())
    else:
        all_predicted_positions_inverse = all_predicted_positions.cpu().numpy()
        test_position_inverse = X_test_tensor.cpu().numpy()
    
    # Calculate evaluation metrics
    mse = np.mean((all_predicted_positions_inverse - test_position_inverse) ** 2)
    mae = np.mean(np.abs(all_predicted_positions_inverse - test_position_inverse))
    
    # Calculate Mean Relative Error (MRE)
    epsilon = 1e-8  # To avoid division by zero
    relative_errors = np.abs(all_predicted_positions_inverse - test_position_inverse) / (np.abs(test_position_inverse) + epsilon)
    mean_relative_error = np.mean(relative_errors)
    
    logger.info(f"Final Test MSE: {mse:.4f}")
    logger.info(f"Final Test MRE: {mean_relative_error:.4f}")
    
    # ============================
    # Plot Loss Curves
    # ============================
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.title('CVAE Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('best_cvae_loss_curves.png')
    plt.close()
    
    # ============================
    # Save Final Results
    # ============================
    results = {
        'test_mse': mse,
        'test_mae': mae,
        'mean_relative_error': mean_relative_error,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    torch.save(results, 'best_cvae_results.pt')
    logger.info("Final results and loss curves saved.")

if __name__ == "__main__":
    main()
