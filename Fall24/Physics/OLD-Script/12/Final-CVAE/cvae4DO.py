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
import random

# =========================
# Reproducibility
# =========================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =========================
# Device Configuration
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Hyperparameters
# =========================
LATENT_DIM = 1024
EPOCHS = 65
BATCH_SIZE = 512
LEARNING_RATE = 2.1210335031751337e-05
PATIENCE = 11
MIN_DELTA = 1.5894218493676975e-05
HIDDEN_LAYER_SIZE = 512
NUM_HIDDEN_LAYERS = 1
ACTIVATION_NAME = 'Tanh'
POSITION_NORM_METHOD = 'MinMaxScaler'
MOMENTA_NORM_METHOD = None

# Dropout rates to explore (Grid Search)
DROPOUT_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]

# File paths
DATA_FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
MODEL_SAVE_DIR = 'models'
RESULTS_DIR = 'results'

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# Data Loading and Preprocessing
# =========================
# Load data
data = pd.read_csv(DATA_FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Normalize the data
if POSITION_NORM_METHOD == 'StandardScaler':
    position_scaler = StandardScaler()
elif POSITION_NORM_METHOD == 'MinMaxScaler':
    position_scaler = MinMaxScaler()
else:
    position_scaler = None

if MOMENTA_NORM_METHOD == 'StandardScaler':
    momenta_scaler = StandardScaler()
elif MOMENTA_NORM_METHOD == 'MinMaxScaler':
    momenta_scaler = MinMaxScaler()
else:
    momenta_scaler = None

if position_scaler is not None:
    position_normalized = position_scaler.fit_transform(position)
else:
    position_normalized = position

if momenta_scaler is not None:
    momenta_normalized = momenta_scaler.fit_transform(momenta)
else:
    momenta_normalized = momenta

# Split data into train, validation, and test sets (70%, 15%, 15%)
train_position, temp_position, train_momenta, temp_momenta = train_test_split(
    position_normalized, momenta_normalized, test_size=0.3, random_state=42
)

val_position, test_position, val_momenta, test_momenta = train_test_split(
    temp_position, temp_momenta, test_size=0.5, random_state=42
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

# Create DataLoaders
train_dataset = TensorDataset(train_position, train_momenta)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(val_position, val_momenta)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = TensorDataset(test_position, test_momenta)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# Define the CVAE Model with Dropout
# =========================
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_layers, activation_function, dropout_p=0.5):
        super(CVAE, self).__init__()

        self.dropout = nn.Dropout(p=dropout_p)  # Initialize Dropout layer

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_layers[0]))
        encoder_layers.append(activation_function)
        encoder_layers.append(self.dropout)  # Apply Dropout after activation

        for i in range(len(hidden_layers) - 1):
            encoder_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            encoder_layers.append(activation_function)
            encoder_layers.append(self.dropout)  # Apply Dropout after activation

        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(hidden_layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[-1], latent_dim)

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim + condition_dim, hidden_layers[-1]))
        decoder_layers.append(activation_function)
        decoder_layers.append(self.dropout)  # Apply Dropout after activation

        for i in reversed(range(len(hidden_layers) - 1)):
            decoder_layers.append(nn.Linear(hidden_layers[i+1], hidden_layers[i]))
            decoder_layers.append(activation_function)
            decoder_layers.append(self.dropout)  # Apply Dropout after activation

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

# =========================
# Define the Loss Function
# =========================
def cvae_loss_fn(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_divergence /= x.size(0) * x.size(1)  # Normalize by batch size and input dimensions
    total_loss = recon_loss + kl_divergence
    if torch.isnan(total_loss):
        print("NaN detected in loss computation.")
    return total_loss

# =========================
# Define Training and Evaluation Function
# =========================
def train_and_evaluate(dropout_p, dropout_p_idx):
    print(f"\n=== Training with Dropout Probability: {dropout_p} ===")
    
    # Initialize activation function
    if ACTIVATION_NAME == 'LeakyReLU':
        activation_function = nn.LeakyReLU()
    else:
        activation_function = getattr(nn, ACTIVATION_NAME)()
    
    # Initialize the model
    hidden_layers = [HIDDEN_LAYER_SIZE // (2 ** i) for i in range(NUM_HIDDEN_LAYERS)]
    cvae = CVAE(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        condition_dim=OUTPUT_DIM,
        hidden_layers=hidden_layers,
        activation_function=activation_function,
        dropout_p=dropout_p
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(cvae.parameters(), lr=LEARNING_RATE)
    
    # Training loop with early stopping
    cvae_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    model_saved = False  # Flag to check if model was saved
    
    for epoch in range(EPOCHS):
        cvae.train()
        train_loss_epoch = 0

        for batch_idx, (position_batch, momenta_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_position, mu, logvar = cvae(position_batch, momenta_batch)
            loss = cvae_loss_fn(recon_position, position_batch, mu, logvar)

            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx+1}. Skipping this batch.")
                continue  # Skip this batch

            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_epoch += loss.item()

        # If no batches were processed, break the loop
        if train_loss_epoch == 0:
            print(f"No valid batches in epoch {epoch+1}. Stopping training.")
            break

        train_loss_epoch /= len(train_loader)
        cvae_losses.append(train_loss_epoch)

        # Validation
        cvae.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for position_batch, momenta_batch in val_loader:
                recon_position, mu, logvar = cvae(position_batch, momenta_batch)
                loss = cvae_loss_fn(recon_position, position_batch, mu, logvar)
                if torch.isnan(loss):
                    print(f"NaN loss detected during validation at epoch {epoch+1}. Skipping this batch.")
                    continue  # Skip this batch
                val_loss_epoch += loss.item()

        val_loss_epoch /= len(val_loader)
        val_losses.append(val_loss_epoch)

        print(f'Epoch {epoch+1}/{EPOCHS}, Training Loss: {train_loss_epoch:.6f}, Validation Loss: {val_loss_epoch:.6f}')

        # Early stopping and model saving
        if not np.isnan(val_loss_epoch) and val_loss_epoch < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss_epoch
            patience_counter = 0
            # Save the model
            model_path = os.path.join(MODEL_SAVE_DIR, f'best_cvae_p{dropout_p_idx}.pth')
            torch.save(cvae.state_dict(), model_path)
            model_saved = True
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # If a model was saved, evaluate on the test set
    if model_saved:
        print(f"Loading best model for Dropout p={dropout_p}")
        # Load the best model
        cvae.load_state_dict(torch.load(model_path))

        # Compute and save Mean and Std of latent variables on training set
        cvae.eval()
        with torch.no_grad():
            mu_list = []
            logvar_list = []
            z_train_list = []
            for position_batch, momenta_batch in train_loader:
                mu, logvar = cvae.encode(position_batch)
                z = cvae.reparameterize(mu, logvar)
                mu_list.append(mu)
                logvar_list.append(logvar)
                z_train_list.append(z)

            mu_train = torch.cat(mu_list, dim=0)
            logvar_train = torch.cat(logvar_list, dim=0)
            z_train = torch.cat(z_train_list, dim=0)

            # Compute mean and std of latent variables
            mu_train_mean = mu_train.mean(dim=0)
            mu_train_std = mu_train.std(dim=0)

            # Convert to standard Python floats
            mu_train_mean = mu_train_mean.cpu().numpy().astype(float).tolist()
            mu_train_std = mu_train_std.cpu().numpy().astype(float).tolist()

            # Save mu_train_mean and mu_train_std
            latent_stats_path = os.path.join(RESULTS_DIR, f'latent_stats_p{dropout_p_idx}.pt')
            torch.save({'mu_train_mean': mu_train_mean, 'mu_train_std': mu_train_std}, latent_stats_path)

        # For test set, sample z from training distribution and decode
        test_predictions = []
        cvae.eval()
        with torch.no_grad():
            # Load latent stats
            latent_stats = torch.load(latent_stats_path)
            mu_train_mean = torch.tensor(latent_stats['mu_train_mean']).to(device)
            mu_train_std = torch.tensor(latent_stats['mu_train_std']).to(device)

            # Sample z from training distribution
            z_sample = torch.randn(len(test_momenta), LATENT_DIM).to(device)
            z_sample = z_sample * mu_train_std.unsqueeze(0) + mu_train_mean.unsqueeze(0)
            for i in range(len(test_momenta)):
                momenta_sample = test_momenta[i].unsqueeze(0)
                predicted_position = cvae.decode(z_sample[i].unsqueeze(0), momenta_sample)
                test_predictions.append(predicted_position)

        test_predictions = torch.cat(test_predictions, dim=0)

        # Inverse transform the predicted and actual positions
        if position_scaler is not None:
            test_predictions_inverse = position_scaler.inverse_transform(test_predictions.cpu().numpy().astype(float))
            test_position_inverse = position_scaler.inverse_transform(test_position.cpu().numpy().astype(float))
        else:
            test_predictions_inverse = test_predictions.cpu().numpy().astype(float)
            test_position_inverse = test_position.cpu().numpy().astype(float)

        # Calculate MSE and MRE on test set using original values
        mse = float(np.mean((test_predictions_inverse - test_position_inverse) ** 2))
        relative_errors = np.abs(test_predictions_inverse - test_position_inverse) / (np.abs(test_position_inverse) + 1e-8)
        mre = float(np.mean(relative_errors))

        print(f"Test MSE for Dropout p={dropout_p}: {mse}")
        print(f"Test MRE for Dropout p={dropout_p}: {mre}")

        # Save the results
        results = {
            'dropout_p': float(dropout_p),
            'mse': mse,
            'mre': mre,
            'hyperparameters': {
                'LATENT_DIM': LATENT_DIM,
                'EPOCHS': EPOCHS,
                'BATCH_SIZE': BATCH_SIZE,
                'LEARNING_RATE': LEARNING_RATE,
                'PATIENCE': PATIENCE,
                'MIN_DELTA': MIN_DELTA,
                'hidden_layer_size': HIDDEN_LAYER_SIZE,
                'num_hidden_layers': NUM_HIDDEN_LAYERS,
                'activation': ACTIVATION_NAME,
                'dropout_p': float(dropout_p),
                'position_norm_method': POSITION_NORM_METHOD,
                'momenta_norm_method': MOMENTA_NORM_METHOD
            }
        }
        results_path = os.path.join(RESULTS_DIR, f'results_p{dropout_p_idx}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        # Plot training and validation loss curves
        # First 10 epochs
        plt.figure()
        plt.plot(range(1, min(len(cvae_losses), 10) + 1), cvae_losses[:10], label='Training Loss')
        plt.plot(range(1, min(len(val_losses), 10) + 1), val_losses[:10], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Loss Curves (First 10 Epochs) - Dropout p={dropout_p}')
        plt.savefig(os.path.join(RESULTS_DIR, f'loss_curves_first_10_epochs_p{dropout_p_idx}.png'))
        plt.close()

        # Remaining epochs
        if len(cvae_losses) > 10:
            plt.figure()
            plt.plot(range(11, len(cvae_losses) + 1), cvae_losses[10:], label='Training Loss')
            plt.plot(range(11, len(val_losses) + 1), val_losses[10:], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'Loss Curves (Epochs 11 onwards) - Dropout p={dropout_p}')
            plt.savefig(os.path.join(RESULTS_DIR, f'loss_curves_rest_epochs_p{dropout_p_idx}.png'))
            plt.close()
        else:
            print("Less than 10 epochs completed, skipping the second loss plot.")

        return mre
    else:
        print(f"No valid model was saved for Dropout p={dropout_p}.")
        return None

# =========================
# Grid Search Over Dropout Rates
# =========================
def perform_grid_search(dropout_rates):
    dropout_mre = {}
    for idx, p in enumerate(dropout_rates):
        mre = train_and_evaluate(dropout_p=p, dropout_p_idx=idx)
        if mre is not None:
            # Ensure keys are strings to avoid JSON serialization issues
            dropout_mre[str(p)] = mre
    return dropout_mre

# Perform Grid Search
dropout_mre_results = perform_grid_search(DROPOUT_RATES)

# =========================
# Identify and Report the Best Dropout Rate
# =========================
if dropout_mre_results:
    # Convert keys back to float for comparison if necessary
    # Find the key with the minimum MRE
    best_p_str = min(dropout_mre_results, key=dropout_mre_results.get)
    best_p = float(best_p_str)
    best_mre = dropout_mre_results[best_p_str]
    print(f"\n=== Grid Search Completed ===")
    print(f"Best Dropout Probability: {best_p} with Test MRE: {best_mre}")

    # Save the overall grid search results
    grid_search_results = {
        'dropout_mre': {float(k): float(v) for k, v in dropout_mre_results.items()},
        'best_dropout_p': float(best_p),
        'best_test_mre': float(best_mre),
        'hyperparameters': {
            'LATENT_DIM': LATENT_DIM,
            'EPOCHS': EPOCHS,
            'BATCH_SIZE': BATCH_SIZE,
            'LEARNING_RATE': LEARNING_RATE,
            'PATIENCE': PATIENCE,
            'MIN_DELTA': MIN_DELTA,
            'hidden_layer_size': HIDDEN_LAYER_SIZE,
            'num_hidden_layers': NUM_HIDDEN_LAYERS,
            'activation': ACTIVATION_NAME,
            'position_norm_method': POSITION_NORM_METHOD,
            'momenta_norm_method': MOMENTA_NORM_METHOD,
            'dropout_rates_explored': DROPOUT_RATES
        }
    }
    grid_search_results_path = os.path.join(RESULTS_DIR, 'grid_search_results.json')
    with open(grid_search_results_path, 'w') as f:
        json.dump(grid_search_results, f, indent=4)

    # Optionally, plot the Grid Search results
    try:
        dropout_p_values = list(grid_search_results['dropout_mre'].keys())
        mre_values = list(grid_search_results['dropout_mre'].values())

        plt.figure()
        plt.plot(dropout_p_values, mre_values, marker='o')
        plt.xlabel('Dropout Probability (p)')
        plt.ylabel('Test MRE')
        plt.title('Grid Search: Dropout Probability vs. Test MRE')
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, 'grid_search_dropout_mre.png'))
        plt.close()
        print(f"Grid search plot saved to {os.path.join(RESULTS_DIR, 'grid_search_dropout_mre.png')}")
    except Exception as e:
        print(f"Failed to plot grid search results: {e}")
else:
    print("No valid models were trained during Grid Search.")
