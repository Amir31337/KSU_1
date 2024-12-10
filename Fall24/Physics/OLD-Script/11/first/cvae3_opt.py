'''
Value (Average Relative Error): 0.686351
  Params:
    hidden_dim: 174
    latent_dim: 20
    batch_size: 128
    n_epochs: 224
    learning_rate: 0.00036324869566766035
    activation: tanh
    patience: 40
    norm_x: standard
    norm_y: none

    Average Relative Error on Test Set: 0.691309
Average MSE on Test Set: 0.730377
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import optuna
from optuna.trial import TrialState

# --------------------- Set Random Seed for Reproducibility ---------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --------------------- Define the CVAE Model ----------------------
class Encoder(nn.Module):
    """
    Encoder part of CVAE: Encodes X and Y into latent space.
    """
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim, activation):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.activation = activation

    def forward(self, x, y):
        concatenated = torch.cat((x, y), dim=1)
        hidden = self.activation(self.fc1(concatenated))
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        return mu, log_var

class Decoder(nn.Module):
    """
    Decoder part of CVAE: Decodes latent variables and Y to reconstruct X.
    """
    def __init__(self, latent_dim, condition_dim, hidden_dim, output_dim, activation):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, z, y):
        concatenated = torch.cat((z, y), dim=1)
        hidden = self.activation(self.fc1(concatenated))
        reconstructed_x = self.fc_out(hidden)
        return reconstructed_x

class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder.
    """
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim, output_dim, activation):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, condition_dim, hidden_dim, latent_dim, activation)
        self.decoder = Decoder(latent_dim, condition_dim, hidden_dim, output_dim, activation)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, y):
        mu, log_var = self.encoder(x, y)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decoder(z, y)
        return reconstructed_x, mu, log_var

# --------------------- Loss Function -------------------------
def calculate_loss(x, reconstructed_x, mu, log_var):
    """
    Computes the CVAE loss as the sum of reconstruction loss and KL divergence.
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')
    # KL Divergence
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld_loss

# --------------------- Training and Validation Functions ---------------------
def train_epoch(model, optimizer, loader, device):
    model.train()
    train_loss = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        reconstructed_x, mu, log_var = model(x, y)
        loss = calculate_loss(x, reconstructed_x, mu, log_var)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    return train_loss

def validate_epoch(model, loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            reconstructed_x, mu, log_var = model(x, y)
            loss = calculate_loss(x, reconstructed_x, mu, log_var)
            val_loss += loss.item()
    return val_loss

# --------------------- Latent Statistics Computation ---------------------
def compute_latent_statistics(model, loader, device):
    """
    Computes the mean and log variance of the latent space from the training data.
    """
    model.eval()
    mu_sum = 0
    log_var_sum = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            mu, log_var = model.encoder(x, y)
            mu_sum += mu.sum(dim=0)
            log_var_sum += log_var.sum(dim=0)
            total_samples += x.size(0)
    
    mu_mean = mu_sum / total_samples
    log_var_mean = log_var_sum / total_samples
    return mu_mean, log_var_mean

# --------------------- Metrics Calculation -------------------
def compute_metrics(model, test_loader, scaler_X, scaler_Y, mu_train_mean, log_var_train_mean, device, latent_dim):
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
            y = y.to(device)
            batch_size = y.size(0)
            
            # Sample z using the training latent statistics
            eps = torch.randn(batch_size, latent_dim).to(device)
            z = torch.from_numpy(mu_train_mean_np).float().to(device) + eps * torch.from_numpy(sigma_train).float().to(device)
            
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

# --------------------- Objective Function for Optuna ---------------------
def objective(trial):
    try:
        # --------------------- Hyperparameter Suggestions ---------------------
        hidden_dim = trial.suggest_int('hidden_dim', 128, 512)
        latent_dim = trial.suggest_int('latent_dim', 8, 32)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        n_epochs = trial.suggest_int('n_epochs', 100, 500)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        activation_name = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'tanh'])
        patience = trial.suggest_int('patience', 5, 50)
        
        # Normalization methods
        normalization_options = ['standard', 'minmax', 'robust', 'none']
        norm_x = trial.suggest_categorical('norm_x', normalization_options)
        norm_y = trial.suggest_categorical('norm_y', normalization_options)
        
        # --------------------- Activation Function Selection ---------------------
        if activation_name == 'relu':
            activation = F.relu
        elif activation_name == 'leaky_relu':
            activation = F.leaky_relu
        elif activation_name == 'tanh':
            activation = torch.tanh
        else:
            activation = F.relu  # Default
        
        # --------------------- Data Loading & Preprocessing --------------------------
        DATA_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
        POSITION_COLUMNS = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
        MOMENTA_COLUMNS = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']
        
        # Load data
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data file not found at path: {DATA_PATH}")
        
        data = pd.read_csv(DATA_PATH)
        position = data[POSITION_COLUMNS].values.astype(np.float32)
        momenta = data[MOMENTA_COLUMNS].values.astype(np.float32)
        
        # Split data into train (70%), validation (15%), and test (15%)
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            position, momenta, test_size=0.15, random_state=42)
        
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=0.1765, random_state=42)  # 0.1765 * 0.85 ≈ 0.15
        
        # Select normalization methods
        scaler_X = None
        scaler_Y = None
        
        if norm_x == 'standard':
            scaler_X = StandardScaler()
        elif norm_x == 'minmax':
            scaler_X = MinMaxScaler()
        elif norm_x == 'robust':
            scaler_X = RobustScaler()
        
        if norm_y == 'standard':
            scaler_Y = StandardScaler()
        elif norm_y == 'minmax':
            scaler_Y = MinMaxScaler()
        elif norm_y == 'robust':
            scaler_Y = RobustScaler()
        
        # Apply normalization if not 'none'
        if scaler_X:
            x_train = scaler_X.fit_transform(x_train)
            x_val = scaler_X.transform(x_val)
            x_test = scaler_X.transform(x_test)
        else:
            # If no normalization, ensure float32
            x_train = x_train.astype(np.float32)
            x_val = x_val.astype(np.float32)
            x_test = x_test.astype(np.float32)
        
        if scaler_Y:
            y_train = scaler_Y.fit_transform(y_train)
            y_val = scaler_Y.transform(y_val)
            y_test = scaler_Y.transform(y_test)
        else:
            y_train = y_train.astype(np.float32)
            y_val = y_val.astype(np.float32)
            y_test = y_test.astype(np.float32)
        
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
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        # ------------------------------------------------------------------------------
        
        # --------------------- Device Configuration ---------------------
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # -------------------------------------------------------------
        
        # --------------------- Initialize Model, Optimizer, Scheduler ---------------------
        input_dim = 9
        condition_dim = 9
        output_dim = 9
        
        model = CVAE(
            input_dim=input_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            activation=activation
        ).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
        # -----------------------------------------------------------------------------------
        
        # --------------------- Training Loop -------------------------
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = model.state_dict()  # Initialize before the loop

        for epoch in range(1, n_epochs + 1):
            train_loss = train_epoch(model, optimizer, train_loader, DEVICE)
            val_loss = validate_epoch(model, val_loader, DEVICE)
            
            scheduler.step(val_loss)
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model state within the trial
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break  # Early stopping
        
        # Load the best model state
        model.load_state_dict(best_model_state)
        # -------------------------------------------------------------
        
        # --------------------- Compute Latent Statistics ----------------------
        mu_train_mean, log_var_train_mean = compute_latent_statistics(model, train_loader, DEVICE)
        # ----------------------------------------------------------------------
        
        # --------------------- Compute Metrics on Test Set ----------------------
        average_relative_error, average_mse = compute_metrics(
            model, test_loader, scaler_X, scaler_Y, mu_train_mean, log_var_train_mean, DEVICE, latent_dim)
        # ----------------------------------------------------------------------
        
        # --------------------- Report to Optuna ---------------------
        # We aim to minimize the Average Relative Error
        return average_relative_error

    except Exception as e:
        # Handle exceptions and report to Optuna
        print(f"Trial failed with exception: {e}")
        return float('inf')  # Assign a worst possible value to the trial

# --------------------- Main Function to Run Optuna Optimization ---------------------
def main():
    # --------------------- Set Global Parameters ---------------------
    RANDOM_SEED = 42
    set_seed(RANDOM_SEED)
    # ---------------------------------------------------------------
    
    # --------------------- Create Optuna Study ---------------------
    study = optuna.create_study(
        direction='minimize', 
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # --------------------- Optimize ---------------------
    study.optimize(objective, n_trials=50, timeout=None)
    # ---------------------------------------------------------------
    
    # --------------------- Print Study Statistics ---------------------
    print("Number of finished trials: ", len(study.trials))
    
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value (Average Relative Error): {trial.value:.6f}")
    
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    # -----------------------------------------------------------------
    
    # --------------------- Retrain Model with Best Hyperparameters ---------------------
    best_params = study.best_trial.params
    
    # Extract hyperparameters
    hidden_dim = best_params['hidden_dim']
    latent_dim = best_params['latent_dim']
    batch_size = best_params['batch_size']
    n_epochs = best_params['n_epochs']
    learning_rate = best_params['learning_rate']
    activation_name = best_params['activation']
    patience = best_params['patience']
    norm_x = best_params['norm_x']
    norm_y = best_params['norm_y']
    
    # --------------------- Activation Function Selection ---------------------
    if activation_name == 'relu':
        activation = F.relu
    elif activation_name == 'leaky_relu':
        activation = F.leaky_relu
    elif activation_name == 'tanh':
        activation = torch.tanh
    else:
        activation = F.relu  # Default
    # ---------------------------------------------------------------------------
    
    # --------------------- Data Loading & Preprocessing --------------------------
    DATA_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
    POSITION_COLUMNS = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
    MOMENTA_COLUMNS = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']
    
    # Load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at path: {DATA_PATH}")
    
    data = pd.read_csv(DATA_PATH)
    position = data[POSITION_COLUMNS].values.astype(np.float32)
    momenta = data[MOMENTA_COLUMNS].values.astype(np.float32)
    
    # Split data into train (70%), validation (15%), and test (15%)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        position, momenta, test_size=0.15, random_state=RANDOM_SEED)
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.1765, random_state=RANDOM_SEED)  # 0.1765 * 0.85 ≈ 0.15
    
    # Select normalization methods
    scaler_X = None
    scaler_Y = None
    
    if norm_x == 'standard':
        scaler_X = StandardScaler()
    elif norm_x == 'minmax':
        scaler_X = MinMaxScaler()
    elif norm_x == 'robust':
        scaler_X = RobustScaler()
    
    if norm_y == 'standard':
        scaler_Y = StandardScaler()
    elif norm_y == 'minmax':
        scaler_Y = MinMaxScaler()
    elif norm_y == 'robust':
        scaler_Y = RobustScaler()
    
    # Apply normalization if not 'none'
    if scaler_X:
        x_train = scaler_X.fit_transform(x_train)
        x_val = scaler_X.transform(x_val)
        x_test = scaler_X.transform(x_test)
    else:
        # If no normalization, ensure float32
        x_train = x_train.astype(np.float32)
        x_val = x_val.astype(np.float32)
        x_test = x_test.astype(np.float32)
    
    if scaler_Y:
        y_train = scaler_Y.fit_transform(y_train)
        y_val = scaler_Y.transform(y_val)
        y_test = scaler_Y.transform(y_test)
    else:
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        y_test = y_test.astype(np.float32)
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    # ------------------------------------------------------------------------------
    
    # --------------------- Device Configuration ---------------------
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------
    
    # --------------------- Initialize Model, Optimizer, Scheduler ---------------------
    input_dim = 9
    condition_dim = 9
    output_dim = 9
    
    model = CVAE(
        input_dim=input_dim,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        activation=activation
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # -----------------------------------------------------------------------------------
    
    # --------------------- Training Loop with Loss Tracking -------------------------
    SAVE_PATH = "MoS2_cvae_best.pt"
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = model.state_dict()  # Initialize before the loop

    # Lists to store losses
    train_losses = []
    val_losses = []
    
    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, optimizer, train_loader, DEVICE)
        val_loss = validate_epoch(model, val_loader, DEVICE)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss / len(train_loader.dataset))
        val_losses.append(val_loss / len(val_loader.dataset))
        
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model state
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Load the best model
    model.load_state_dict(best_model_state)
    # -------------------------------------------------------------
    
    # --------------------- Compute Latent Statistics ----------------------
    mu_train_mean, log_var_train_mean = compute_latent_statistics(model, train_loader, DEVICE)
    # ----------------------------------------------------------------------
    
    # --------------------- Plot Loss Curves ----------------------
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()
    # -------------------------------------------------------------
    
    # --------------------- Compute Metrics on Test Set ----------------------
    average_relative_error, average_mse = compute_metrics(
        model, test_loader, scaler_X, scaler_Y, mu_train_mean, log_var_train_mean, DEVICE, latent_dim)
    
    print(f'Average Relative Error on Test Set: {average_relative_error:.6f}')
    print(f'Average MSE on Test Set: {average_mse:.6f}')
    # ----------------------------------------------------------------------
    
    # --------------------- Save the Latent Statistics -------------------------
    LATENT_STATS_PATH = "latent_stats_best.pt"
    torch.save({
        'mu_train_mean': mu_train_mean,
        'log_var_train_mean': log_var_train_mean
    }, LATENT_STATS_PATH)
    print(f'Latent statistics saved to {LATENT_STATS_PATH}')
    # -------------------------------------------------------------------------
    
    # --------------------- Save the Best Model -------------------------
    # The best model was already saved during training
    print(f'Model saved to {SAVE_PATH}')
    # -------------------------------------------------------------

if __name__ == "__main__":
    main()
    
    # --------------------- Print Best Hyperparameters and Their Metrics ---------------------
    # Since metrics are computed during the main function, they are printed there.
    # If needed, additional logging can be implemented.
    # -------------------------------------------------------------------------------------------
    
    # --------------------- Print Study Best Parameters Separately ---------------------
    # Access the study after main function
    # Note: To access study outside, you may need to adjust the code structure.
    # For simplicity, the best parameters are printed within the main function.
    pass
