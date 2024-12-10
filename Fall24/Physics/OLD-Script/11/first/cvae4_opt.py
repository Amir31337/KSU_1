'''
Study statistics: 
  Number of finished trials: 200
  Number of pruned trials: 82
  Number of complete trials: 118
Best trial:
  Value (MRE): 0.703240
  Hyperparameters:
    latent_dim: 363
    epochs: 2412
    batch_size: 2048
    learning_rate: 6.28567514806851e-05
    patience: 103
    min_delta: 0.09970680168667148
    encoder_num_layers: 10
    encoder_layer_1_size: 1485
    encoder_layer_2_size: 1846
    encoder_layer_3_size: 1639
    encoder_layer_4_size: 761
    encoder_layer_5_size: 1023
    encoder_layer_6_size: 1627
    encoder_layer_7_size: 675
    encoder_layer_8_size: 1398
    encoder_layer_9_size: 1156
    encoder_layer_10_size: 722
    decoder_num_layers: 5
    decoder_layer_1_size: 162
    decoder_layer_2_size: 909
    decoder_layer_3_size: 1926
    decoder_layer_4_size: 281
    decoder_layer_5_size: 1089
    activation: softplus
    norm_same: True
    norm_method: maxabs
Optuna study results have been saved to 'optuna_study_results.csv'.
'''
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (MinMaxScaler, StandardScaler, RobustScaler,
                                   MaxAbsScaler, QuantileTransformer, PowerTransformer)
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState

# Set a seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CVAE model components
class Encoder(nn.Module):
    def __init__(self, position_dim, momenta_dim, latent_dim, hidden_layers, activation):
        super(Encoder, self).__init__()
        layers = []
        input_dim = position_dim + momenta_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            input_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, position, momenta):
        x = torch.cat([position, momenta], dim=1)
        x = self.encoder(x)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_logvar(x)
        return z_mean, z_log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, momenta_dim, output_dim, hidden_layers, activation):
        super(Decoder, self).__init__()
        layers = []
        input_dim = latent_dim + momenta_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z, momenta):
        x = torch.cat([z, momenta], dim=1)
        x = self.decoder(x)
        return x

class CVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, position, momenta):
        mu, logvar = self.encoder(position, momenta)
        z = self.reparameterize(mu, logvar)
        recon_position = self.decoder(z, momenta)
        return recon_position, mu, logvar

# Define the objective function for Optuna
def objective(trial):
    # =========================
    # Hyperparameter Suggestions
    # =========================
    LATENT_DIM = trial.suggest_int('latent_dim', 8, 512)
    EPOCHS = trial.suggest_int('epochs', 100, 3000)
    BATCH_SIZE = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024, 2048, 4096, 8192])
    LEARNING_RATE = trial.suggest_loguniform('learning_rate', 1e-7, 1e-1)
    PATIENCE = trial.suggest_int('patience', 5, 200)
    MIN_DELTA = trial.suggest_loguniform('min_delta', 1e-7, 1e-1)
    
    # Encoder Layers: number of layers and size
    encoder_num_layers = trial.suggest_int('encoder_num_layers', 1, 10)
    encoder_layers = []
    for i in range(encoder_num_layers):
        layer_size = trial.suggest_int(f'encoder_layer_{i+1}_size', 16, 2048)
        encoder_layers.append(layer_size)
    
    # Decoder Layers: number of layers and size
    decoder_num_layers = trial.suggest_int('decoder_num_layers', 1, 10)
    decoder_layers = []
    for i in range(decoder_num_layers):
        layer_size = trial.suggest_int(f'decoder_layer_{i+1}_size', 16, 2048)
        decoder_layers.append(layer_size)
    
    # Activation function
    activation_name = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'tanh', 'selu', 'gelu', 'softplus', 'relu6'])
    activation_funcs = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'tanh': nn.Tanh(),
        'selu': nn.SELU(),
        'gelu': nn.GELU(),
        'softplus': nn.Softplus(),
        'relu6': nn.ReLU6()
    }
    activation = activation_funcs[activation_name]
    
    # Normalization methods
    normalization_options = ['minmax', 'standard', 'robust', 'maxabs', 'quantile', 'power', 'none']
    
    # Decide if normalization methods for position and momenta are same or different
    norm_same = trial.suggest_categorical('norm_same', [True, False])
    
    if norm_same:
        norm_method_pos = trial.suggest_categorical('norm_method', normalization_options[:-1])  # Exclude 'none' if same
        norm_method_mom = norm_method_pos
    else:
        norm_method_pos = trial.suggest_categorical('norm_method_pos', normalization_options)
        norm_method_mom = trial.suggest_categorical('norm_method_mom', normalization_options)
    
    # =========================
    # Data Loading and Preprocessing
    # =========================
    FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
    
    try:
        data = pd.read_csv(FILEPATH)
    except Exception as e:
        raise RuntimeError(f"Error reading the CSV file: {e}")
    
    position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
    momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values
    
    # Check for NaNs or infinite values
    if np.isnan(position).any() or np.isinf(position).any():
        raise ValueError("Position data contains NaNs or infinite values.")
    if np.isnan(momenta).any() or np.isinf(momenta).any():
        raise ValueError("Momenta data contains NaNs or infinite values.")
    
    # Apply normalization
    def get_scaler(method):
        if method == 'minmax':
            return MinMaxScaler()
        elif method == 'standard':
            return StandardScaler()
        elif method == 'robust':
            return RobustScaler()
        elif method == 'maxabs':
            return MaxAbsScaler()
        elif method == 'quantile':
            return QuantileTransformer(output_distribution='normal', random_state=SEED)
        elif method == 'power':
            return PowerTransformer()
        elif method == 'none':
            return None
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    scaler_pos = get_scaler(norm_method_pos) if norm_method_pos != 'none' else None
    scaler_mom = get_scaler(norm_method_mom) if norm_method_mom != 'none' else None
    
    if scaler_pos:
        try:
            position_normalized = scaler_pos.fit_transform(position)
        except Exception as e:
            raise RuntimeError(f"Error during position normalization: {e}")
    else:
        position_normalized = position.copy()
    
    if scaler_mom:
        try:
            momenta_normalized = scaler_mom.fit_transform(momenta)
        except Exception as e:
            raise RuntimeError(f"Error during momenta normalization: {e}")
    else:
        momenta_normalized = momenta.copy()
    
    # Split data into train (70%), temp (30%)
    try:
        train_position, temp_position, train_momenta, temp_momenta = train_test_split(
            position_normalized, momenta_normalized, test_size=0.3, random_state=SEED
        )
    except Exception as e:
        raise RuntimeError(f"Error during data splitting (train/temp): {e}")
    
    # Split temp into validation (15%) and test (15%)
    try:
        val_position, test_position, val_momenta, test_momenta = train_test_split(
            temp_position, temp_momenta, test_size=0.5, random_state=SEED
        )
    except Exception as e:
        raise RuntimeError(f"Error during data splitting (validation/test): {e}")
    
    # Convert to PyTorch tensors
    train_position = torch.FloatTensor(train_position)
    train_momenta = torch.FloatTensor(train_momenta)
    val_position = torch.FloatTensor(val_position)
    val_momenta = torch.FloatTensor(val_momenta)
    test_position = torch.FloatTensor(test_position)
    test_momenta = torch.FloatTensor(test_momenta)
    
    # =========================
    # Instantiate Model
    # =========================
    POSITION_DIM = train_position.shape[1]
    MOMENTA_DIM = train_momenta.shape[1]
    
    encoder = Encoder(
        position_dim=POSITION_DIM,
        momenta_dim=MOMENTA_DIM,
        latent_dim=LATENT_DIM,
        hidden_layers=encoder_layers,
        activation=activation
    ).to(device)
    
    decoder = Decoder(
        latent_dim=LATENT_DIM,
        momenta_dim=MOMENTA_DIM,
        output_dim=POSITION_DIM,
        hidden_layers=decoder_layers,
        activation=activation
    ).to(device)
    
    cvae = CVAE(encoder, decoder).to(device)
    
    # =========================
    # Loss Function
    # =========================
    def loss_function(recon_position, position, mu, logvar):
        MSE = nn.functional.mse_loss(recon_position, position, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD
    
    # =========================
    # Optimizer
    # =========================
    optimizer = optim.Adam(cvae.parameters(), lr=LEARNING_RATE)
    
    # =========================
    # DataLoaders
    # =========================
    train_dataset = TensorDataset(train_position, train_momenta)
    val_dataset = TensorDataset(val_position, val_momenta)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # =========================
    # Training Loop with Gradient Clipping and NaN Handling
    # =========================
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    CLIP_VALUE = 1.0  # Gradient clipping value
    
    for epoch in range(EPOCHS):
        # Training
        cvae.train()
        train_loss = 0
        for batch_idx, (position, momenta) in enumerate(train_loader):
            position = position.to(device)
            momenta = momenta.to(device)
            optimizer.zero_grad()
            recon_position, mu, logvar = cvae(position, momenta)
            loss = loss_function(recon_position, position, mu, logvar)
            
            if torch.isnan(loss) or torch.isinf(loss):
                # Prune trial if loss is NaN or Inf
                raise optuna.exceptions.TrialPruned()
            
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(cvae.parameters(), CLIP_VALUE)
            train_loss += loss.item()
            optimizer.step()
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        cvae.eval()
        val_loss = 0
        with torch.no_grad():
            for position, momenta in val_loader:
                position = position.to(device)
                momenta = momenta.to(device)
                recon_position, mu, logvar = cvae(position, momenta)
                loss = loss_function(recon_position, position, mu, logvar)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    raise optuna.exceptions.TrialPruned()
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Logging
        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Print progress
        print(f'Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early Stopping
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            # Store the best model's state_dict in-memory
            best_model_state = cvae.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch + 1}')
                break
    
    # =========================
    # Load Best Model
    # =========================
    cvae.load_state_dict(best_model_state)
    
    # =========================
    # Evaluation on Test Set
    # =========================
    cvae.eval()
    with torch.no_grad():
        # Sample from the training distribution's statistics
        mu_train, logvar_train = cvae.encoder(train_position.to(device), train_momenta.to(device))
        mu_train_mean = mu_train.mean(dim=0).cpu().numpy()
        std_train_mean = torch.exp(0.5 * logvar_train).mean(dim=0).cpu().numpy()
        
        # Sample z from normal distribution
        z = np.random.normal(loc=mu_train_mean, scale=std_train_mean, size=(test_momenta.shape[0], LATENT_DIM))
        z = torch.FloatTensor(z).to(device)
        test_momenta = test_momenta.to(device)
        
        # Reconstruct positions
        predicted_position = cvae.decoder(z, test_momenta)
    
    # Denormalize the predictions and true positions
    if norm_same and norm_method_pos != 'none':
        predicted_position_denorm = scaler_pos.inverse_transform(predicted_position.cpu().numpy())
        true_position_denorm = scaler_pos.inverse_transform(test_position.cpu().numpy())
    else:
        # Handle different normalization methods
        if scaler_pos:
            predicted_position_denorm = scaler_pos.inverse_transform(predicted_position.cpu().numpy())
            true_position_denorm = scaler_pos.inverse_transform(test_position.cpu().numpy())
        else:
            predicted_position_denorm = predicted_position.cpu().numpy()
            true_position_denorm = test_position.cpu().numpy()
    
    # Calculate evaluation metrics
    mse = np.mean((true_position_denorm - predicted_position_denorm) ** 2)
    mae = np.mean(np.abs(true_position_denorm - predicted_position_denorm))
    
    epsilon = 1e-8
    relative_errors = np.abs(true_position_denorm - predicted_position_denorm) / (np.abs(true_position_denorm) + epsilon)
    mre = np.mean(relative_errors)
    
    # Save predicted positions
    np.savetxt('predicted_positions.csv', predicted_position_denorm, delimiter=',')
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_curves.png')
    plt.close()
    
    # Print metrics
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test MRE: {mre:.6f}")
    
    # Return MRE for Optuna to minimize
    return mre

# =========================
# Optuna Study Setup
# =========================
def main():
    # Define the Optuna study
    study = optuna.create_study(
        direction='minimize', 
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=50)
    )
    
    # Optimize the study
    try:
        study.optimize(objective, n_trials=200, timeout=7200)  # Adjust n_trials and timeout as needed
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    
    # Gather study statistics
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    
    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")
    
    # Display the best trial
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value (MRE): {trial.value:.6f}")
    
    print("  Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save the study results
    df = study.trials_dataframe()
    df.to_csv('optuna_study_results.csv', index=False)
    print("Optuna study results have been saved to 'optuna_study_results.csv'.")

if __name__ == "__main__":
    main()
