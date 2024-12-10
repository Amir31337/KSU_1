import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sys  # Added for logging
import warnings  # For handling warnings
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import autocast, GradScaler  # Corrected import for mixed precision
from scipy import stats
import concurrent.futures
import multiprocessing
import random

# =======================
# Logger Class for Logging
# =======================

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")  # Use "w" to overwrite existing file
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# =======================
# Set Random Seeds for Reproducibility
# =======================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =======================
# Define Hyperparameter Sets
# =======================

hyperparameter_sets = [
    {
        "latent_dim": 512,
        "epochs": 200,
        "batch_size": 1024,
        "learning_rate": 0.0097564082801063459,
        "patience": 50,
        "min_delta": 0.001,
        "activation_name": "ReLU",
        "position_norm_method": "MinMaxScaler",
        "momenta_norm_method": "MinMaxScaler",
        "use_l1": False,
        "use_l2": False,
        "num_hidden_layers": 3,
        "hidden_layer_size": 256
    },
    {
        "latent_dim": 256,
        "epochs": 200,
        "batch_size": 256,
        "learning_rate": 0.0026468425008510338,
        "patience": 100,
        "min_delta": 0.01,
        "activation_name": "Tanh",
        "position_norm_method": "MinMaxScaler",
        "momenta_norm_method": "StandardScaler",
        "use_l1": True,
        "l1_lambda": 0.2,
        "use_l2": False,
        "num_hidden_layers": 1,
        "hidden_layer_size": 512
    },
    {
        "latent_dim": 128,
        "epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.00018963648105805656,
        "patience": 100,
        "min_delta": 0.001,
        "activation_name": "ReLU",
        "position_norm_method": "MinMaxScaler",
        "momenta_norm_method": "MinMaxScaler",
        "use_l1": False,
        "use_l2": False,
        "num_hidden_layers": 1,
        "hidden_layer_size": 512
    },
    {
        "latent_dim": 256,
        "epochs": 300,
        "batch_size": 128,
        "learning_rate": 2.220822537058567e-05,
        "patience": 100,
        "min_delta": 0.01,
        "activation_name": "ELU",
        "position_norm_method": "MinMaxScaler",
        "momenta_norm_method": "StandardScaler",
        "use_l1": False,
        "use_l2": False,
        "num_hidden_layers": 2,
        "hidden_layer_size": 1024
    },
    {
        "latent_dim": 256,
        "epochs": 100,
        "batch_size": 512,
        "learning_rate": 1.013831099621457e-05,
        "patience": 100,
        "min_delta": 0.01,
        "activation_name": "ELU",
        "position_norm_method": "MinMaxScaler",
        "momenta_norm_method": "MinMaxScaler",
        "use_l1": True,
        "l1_lambda": 0.0001,
        "use_l2": True,
        "l2_lambda": 0.0001,
        "num_hidden_layers": 4,
        "hidden_layer_size": 512
    }
]

# =======================
# Function Definitions
# =======================

def create_output_directory(run_index, hyperparams):
    """
    Creates a unique directory for each run based on run index and hyperparameters.
    """
    # Create a descriptive directory name
    dir_name = f"run_{run_index+1}_latent{hyperparams['latent_dim']}_lr{hyperparams['learning_rate']}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def save_json(data, filepath):
    """
    Saves a dictionary to a JSON file.
    """
    with open(filepath, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def plot_learning_curves(train_losses, val_losses, output_dir):
    """
    Plots and saves the learning curves.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'))  # Save before showing
    plt.close()

def plot_split_learning_curves(train_losses, val_losses, output_dir, split_epoch=2):
    """
    Plots and saves split learning curves.
    """
    plt.figure(figsize=(15, 5))

    # First subplot - First split_epoch epochs
    plt.subplot(1, 2, 1)
    plt.plot(range(1, split_epoch + 1), train_losses[:split_epoch], label='Training Loss')
    plt.plot(range(1, split_epoch + 1), val_losses[:split_epoch], label='Validation Loss')
    plt.title(f'Learning Curve (First {split_epoch} Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Second subplot - Remaining epochs
    remaining_epochs = range(split_epoch + 1, len(train_losses) + 1)
    plt.subplot(1, 2, 2)
    plt.plot(remaining_epochs, train_losses[split_epoch:], label='Training Loss')
    plt.plot(remaining_epochs, val_losses[split_epoch:], label='Validation Loss')
    plt.title(f'Learning Curve (Epochs {split_epoch+1} onwards)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curve_split.png'))  # Save before showing
    plt.close()

def plot_distribution(var, label, output_dir):
    """
    Plots and saves the distribution of a variable with fitted normal distribution.
    Also performs the Anderson-Darling test.
    """
    # Ignore runtime warnings during statistical tests
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # Check if variance is zero or data contains NaN/infinite values
        if np.var(var) == 0 or not np.isfinite(var).all():
            print(f"Variable {label} has zero variance or contains non-finite values. Cannot fit normal distribution.")
            ad_results = {
                "label": label,
                "statistic": None,
                "significance_levels": None,
                "critical_values": None
            }
            mean = np.nanmean(var)
            std = 0
            # Plot histogram without fitted normal distribution
            plt.figure(figsize=(10, 6))
            plt.hist(var, bins=50, density=True, alpha=0.7, label='Data')
            plt.legend()
            plt.title(f'Distribution Plot - {label}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'distribution_plot_{label}.png'))
            plt.close()
            return ad_results, mean, std

        try:
            # Anderson-Darling test
            result = stats.anderson(var, dist='norm')

            # Histogram with fitted normal
            mean, std = stats.norm.fit(var)
            x = np.linspace(min(var), max(var), 100)
            pdf = stats.norm.pdf(x, mean, std)

            plt.figure(figsize=(10, 6))
            plt.hist(var, bins=50, density=True, alpha=0.7, label='Data')
            plt.plot(x, pdf, 'r-', label=f'Normal Dist.\n(μ={mean:.2f}, σ={std:.2f})')
            plt.legend()
            plt.title(f'Distribution Plot - {label}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'distribution_plot_{label}.png'))
            plt.close()

            # Prepare Anderson-Darling test results
            ad_results = {
                "label": label,
                "statistic": float(result.statistic),
                "significance_levels": result.significance_level.tolist(),
                "critical_values": result.critical_values.tolist()
            }
            return ad_results, mean, std
        except Exception as e:
            print(f"An error occurred while processing variable {label}: {e}")
            ad_results = {
                "label": label,
                "statistic": None,
                "significance_levels": None,
                "critical_values": None
            }
            mean = np.nanmean(var)
            std = np.nanstd(var)
            # Plot histogram without fitted normal distribution
            plt.figure(figsize=(10, 6))
            plt.hist(var, bins=50, density=True, alpha=0.7, label='Data')
            plt.legend()
            plt.title(f'Distribution Plot - {label}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'distribution_plot_{label}.png'))
            plt.close()
            return ad_results, mean, std

def run_experiment(hyperparams, run_index, gpu_id):
    """
    Runs a single CVAE experiment with the given hyperparameters on the specified GPU.
    """
    # Create output directory
    output_dir = create_output_directory(run_index, hyperparams)

    # Set up logging to file
    log_file_path = os.path.join(output_dir, 'output_log.txt')
    sys.stdout = Logger(log_file_path)
    sys.stderr = Logger(log_file_path)

    # Set device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Run {run_index+1}: Using device {device}")

    # Load data
    FILEPATH = 'sim_million_orient.csv'  # Replace with your actual file path
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

    INPUT_DIM = position.shape[1]  # 9
    OUTPUT_DIM = momenta.shape[1]  # 9

    # Set activation function
    activation_name = hyperparams.get("activation_name", "ReLU")
    try:
        activation_function = getattr(nn, activation_name)()
    except AttributeError:
        print(f"Activation function {activation_name} not found. Using ReLU instead.")
        activation_function = nn.ReLU()

    # Normalization methods
    position_norm_method = hyperparams.get("position_norm_method", "MinMaxScaler")
    momenta_norm_method = hyperparams.get("momenta_norm_method", "MinMaxScaler")

    if position_norm_method == 'StandardScaler':
        position_scaler = StandardScaler()
    elif position_norm_method == 'MinMaxScaler':
        position_scaler = MinMaxScaler()
    elif position_norm_method == 'None':
        position_scaler = None
    else:
        print(f"Unknown position_norm_method '{position_norm_method}'. Using MinMaxScaler.")
        position_scaler = MinMaxScaler()

    if momenta_norm_method == 'StandardScaler':
        momenta_scaler = StandardScaler()
    elif momenta_norm_method == 'MinMaxScaler':
        momenta_scaler = MinMaxScaler()
    elif momenta_norm_method == 'None':
        momenta_scaler = None
    else:
        print(f"Unknown momenta_norm_method '{momenta_norm_method}'. Using MinMaxScaler.")
        momenta_scaler = MinMaxScaler()

    # Normalize the data
    if position_scaler is not None:
        train_position_norm = torch.FloatTensor(position_scaler.fit_transform(train_position))
        val_position_norm = torch.FloatTensor(position_scaler.transform(val_position))
        test_position_norm = torch.FloatTensor(position_scaler.transform(test_position))
    else:
        train_position_norm = torch.FloatTensor(train_position)
        val_position_norm = torch.FloatTensor(val_position)
        test_position_norm = torch.FloatTensor(test_position)

    if momenta_scaler is not None:
        train_momenta_norm = torch.FloatTensor(momenta_scaler.fit_transform(train_momenta))
        val_momenta_norm = torch.FloatTensor(momenta_scaler.transform(val_momenta))
        test_momenta_norm = torch.FloatTensor(momenta_scaler.transform(test_momenta))
    else:
        train_momenta_norm = torch.FloatTensor(momenta)
        val_momenta_norm = torch.FloatTensor(val_momenta)
        test_momenta_norm = torch.FloatTensor(test_momenta)

    # Hidden layers configuration
    hidden_layers = [hyperparams.get("hidden_layer_size", 256)] * hyperparams.get("num_hidden_layers", 3)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(train_position_norm, train_momenta_norm)
    train_loader = DataLoader(train_dataset, batch_size=hyperparams.get("batch_size", 256), shuffle=True, pin_memory=True)

    val_dataset = TensorDataset(val_position_norm, val_momenta_norm)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams.get("batch_size", 256), shuffle=False, pin_memory=True)

    test_dataset = TensorDataset(test_position_norm, test_momenta_norm)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams.get("batch_size", 256), shuffle=False, pin_memory=True)

    # Define the CVAE model
    class CVAE(nn.Module):
        def __init__(self, input_dim, output_dim, latent_dim, hidden_layers, activation_function):
            super(CVAE, self).__init__()
            # Encoder
            encoder_layers = []
            prev_dim = input_dim + output_dim  # positions + momenta
            for h_dim in hidden_layers:
                encoder_layers.append(nn.Linear(prev_dim, h_dim))
                encoder_layers.append(activation_function)
                prev_dim = h_dim
            self.encoder = nn.Sequential(*encoder_layers)
            # Latent space
            self.fc_mu = nn.Linear(prev_dim, latent_dim)
            self.fc_logvar = nn.Linear(prev_dim, latent_dim)
            # Decoder
            decoder_layers = []
            prev_dim = latent_dim + output_dim  # latent vector + momenta
            for h_dim in reversed(hidden_layers):
                decoder_layers.append(nn.Linear(prev_dim, h_dim))
                decoder_layers.append(activation_function)
                prev_dim = h_dim
            decoder_layers.append(nn.Linear(prev_dim, input_dim))  # output_dim == positions
            self.decoder = nn.Sequential(*decoder_layers)

        def encode(self, x, c):
            # x: positions, c: conditions (momenta)
            inputs = torch.cat([x, c], dim=1)
            h = self.encoder(inputs)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std  # z ~ N(mu, std)

        def decode(self, z, c):
            inputs = torch.cat([z, c], dim=1)
            return self.decoder(inputs)

        def forward(self, x, c):
            mu, logvar = self.encode(x, c)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z, c)
            return x_recon, mu, logvar

    # Instantiate the model
    model = CVAE(INPUT_DIM, OUTPUT_DIM, hyperparams.get("latent_dim", 512), hidden_layers, activation_function).to(device)

    # Define the loss function with L1 and L2 regularization and beta parameter
    def loss_function(x_recon, x, mu, logvar, model, use_l1, use_l2, L1_LAMBDA, L2_LAMBDA, beta):
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
        # KL divergence
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # L1 and L2 regularization
        l1_reg = torch.tensor(0., device=device)
        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            if use_l1:
                l1_reg = l1_reg + torch.norm(param, 1)
            if use_l2:
                l2_reg = l2_reg + torch.norm(param, 2)
        total_loss = recon_loss + beta * kl_div
        if use_l1:
            total_loss += L1_LAMBDA * l1_reg
        if use_l2:
            total_loss += L2_LAMBDA * l2_reg
        return total_loss, recon_loss, kl_div

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=hyperparams.get("learning_rate", 0.001))

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Training loop with early stopping
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Variables to compute mean and std of mu
    mu_sum = torch.zeros(hyperparams.get("latent_dim", 512)).to(device)
    mu_squared_sum = torch.zeros(hyperparams.get("latent_dim", 512)).to(device)
    total_samples = 0

    for epoch in range(hyperparams.get("epochs", 100)):
        model.train()
        train_loss = 0
        total_recon_loss = 0
        total_kl_div = 0
        for batch_idx, (data_x, data_c) in enumerate(train_loader):
            data_x = data_x.to(device)
            data_c = data_c.to(device)
            optimizer.zero_grad()

            with autocast():
                x_recon, mu, logvar = model(data_x, data_c)
                loss, recon_loss, kl_div = loss_function(
                    x_recon, data_x, mu, logvar, model,
                    hyperparams.get("use_l1", False),
                    hyperparams.get("use_l2", False),
                    hyperparams.get("l1_lambda", 0),
                    hyperparams.get("l2_lambda", 0),
                    hyperparams.get("beta", 1.0)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * data_x.size(0)  # Multiply by batch size
            total_recon_loss += recon_loss.item() * data_x.size(0)
            total_kl_div += kl_div.item() * data_x.size(0)

            # Update running sums for mu
            batch_size = data_x.size(0)
            mu_sum += mu.sum(dim=0)
            mu_squared_sum += (mu ** 2).sum(dim=0)
            total_samples += batch_size

        train_losses.append(train_loss / len(train_loader.dataset))

        # Validation
        model.eval()
        val_loss = 0
        total_val_recon_loss = 0
        total_val_kl_div = 0
        with torch.no_grad():
            for batch_idx, (data_x, data_c) in enumerate(val_loader):
                data_x = data_x.to(device)
                data_c = data_c.to(device)
                with autocast():
                    x_recon, mu, logvar = model(data_x, data_c)
                    loss, recon_loss, kl_div = loss_function(
                        x_recon, data_x, mu, logvar, model,
                        hyperparams.get("use_l1", False),
                        hyperparams.get("use_l2", False),
                        hyperparams.get("l1_lambda", 0),
                        hyperparams.get("l2_lambda", 0),
                        hyperparams.get("beta", 1.0)
                    )
                    val_loss += loss.item() * data_x.size(0)  # Multiply by batch size
                    total_val_recon_loss += recon_loss.item() * data_x.size(0)
                    total_val_kl_div += kl_div.item() * data_x.size(0)
        val_losses.append(val_loss / len(val_loader.dataset))

        # Early stopping
        if val_losses[-1] + hyperparams.get("min_delta", 0.001) < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_no_improve = 0
            # Save the best model
            best_model_path = os.path.join(output_dir, 'best_cvae_model.pth')
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve == hyperparams.get("patience", 10):
            print(f'Run {run_index+1}: Early stopping at epoch {epoch+1}')
            break

        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kl_div = total_kl_div / len(train_loader.dataset)
        avg_val_recon_loss = total_val_recon_loss / len(val_loader.dataset)
        avg_val_kl_div = total_val_kl_div / len(val_loader.dataset)
        print(f'Run {run_index+1} Epoch {epoch+1}, Train Loss: {train_losses[-1]:.6f}, Recon Loss: {avg_recon_loss:.6f}, KL Div: {avg_kl_div:.6f}')
        print(f'               Val Loss: {val_losses[-1]:.6f}, Val Recon Loss: {avg_val_recon_loss:.6f}, Val KL Div: {avg_val_kl_div:.6f}')

        # Clear GPU cache to help manage memory
        torch.cuda.empty_cache()

    # Compute latent distribution parameters from training data
    if total_samples == 0:
        print(f"Run {run_index+1}: No samples processed during training. Cannot compute latent_mu_mean and latent_mu_std.")
        latent_mu_mean = torch.zeros_like(mu_sum)
        latent_mu_std = torch.zeros_like(mu_sum)
    else:
        latent_mu_mean = mu_sum / total_samples
        latent_mu_variance = mu_squared_sum / total_samples - latent_mu_mean ** 2
        latent_mu_variance = torch.clamp(latent_mu_variance, min=0)
        latent_mu_std = torch.sqrt(latent_mu_variance)

    print(f"Run {run_index+1}: Latent mean std dev: {latent_mu_std.mean().item():.6f}")

    # Evaluation on training data
    model.eval()
    train_mse_total = 0
    train_mre_total = 0
    with torch.no_grad():
        for batch_idx, (data_x, data_c) in enumerate(train_loader):
            data_x = data_x.to(device)
            data_c = data_c.to(device)
            with autocast():
                x_recon, _, _ = model(data_x, data_c)
                mse = nn.functional.mse_loss(x_recon, data_x, reduction='sum').item()
                # Add epsilon to avoid division by zero
                epsilon = 1e-8
                mre = (torch.abs(x_recon - data_x) / (torch.abs(data_x) + epsilon)).sum().item()
                train_mse_total += mse
                train_mre_total += mre
    train_mse = train_mse_total / len(train_loader.dataset)
    train_mre = train_mre_total / (len(train_loader.dataset) * INPUT_DIM)

    # Evaluation on validation data
    val_mse_total = 0
    val_mre_total = 0
    with torch.no_grad():
        for batch_idx, (data_x, data_c) in enumerate(val_loader):
            data_x = data_x.to(device)
            data_c = data_c.to(device)
            with autocast():
                x_recon, _, _ = model(data_x, data_c)
                mse = nn.functional.mse_loss(x_recon, data_x, reduction='sum').item()
                epsilon = 1e-8
                mre = (torch.abs(x_recon - data_x) / (torch.abs(data_x) + epsilon)).sum().item()
                val_mse_total += mse
                val_mre_total += mre
    val_mse = val_mse_total / len(val_loader.dataset)
    val_mre = val_mre_total / (len(val_loader.dataset) * INPUT_DIM)

    # Evaluation on test data without data leakage
    test_mse_total = 0
    test_mre_total = 0
    test_recon_positions = []
    with torch.no_grad():
        for batch_idx, (data_x, data_c) in enumerate(test_loader):
            data_c = data_c.to(device)
            data_x = data_x.to(device)
            batch_size = data_c.size(0)
            # Sample z from standard normal distribution N(0, I)
            z = torch.randn(batch_size, hyperparams.get("latent_dim", 512)).to(device)
            with autocast():
                x_recon = model.decode(z, data_c)
            mse = nn.functional.mse_loss(x_recon, data_x, reduction='sum').item()
            epsilon = 1e-8
            mre = (torch.abs(x_recon - data_x) / (torch.abs(data_x) + epsilon)).sum().item()
            test_mse_total += mse
            test_mre_total += mre
            test_recon_positions.append(x_recon.cpu())

    test_mse = test_mse_total / len(test_loader.dataset)
    test_mre = test_mre_total / (len(test_loader.dataset) * INPUT_DIM)

    # Concatenate all reconstructed positions
    test_recon_positions = torch.cat(test_recon_positions, dim=0)

    # Print the errors
    print(f'\nRun {run_index+1} Metrics:')
    print(f'Train MSE: {train_mse:.6f}, Train MRE: {train_mre:.2%}')
    print(f'Validation MSE: {val_mse:.6f}, Validation MRE: {val_mre:.2%}')
    print(f'Test MSE: {test_mse:.6f}, Test MRE: {test_mre:.2%}')

    # Plot learning curves
    plot_learning_curves(train_losses, val_losses, output_dir)

    # Plot split learning curves
    plot_split_learning_curves(train_losses, val_losses, output_dir)

    # Print 5 random samples from the test set
    print("\nSampled Test Data:")
    num_samples = 5
    indices = np.random.choice(len(test_position_norm), num_samples, replace=False)

    for idx in indices:
        pos_actual = test_position_norm[idx].numpy()
        mom = test_momenta_norm[idx].numpy()
        pos_recon = test_recon_positions[idx].numpy()
        print(f"Sample {idx}:")
        print(f"Position (Actual): {pos_actual}")
        print(f"Momenta: {mom}")
        print(f"Position (Reconstructed): {pos_recon}\n")

    # Invert normalization to get original scale for positions and reconstructed positions
    if position_scaler is not None:
        test_position_actual = position_scaler.inverse_transform(test_position_norm.cpu().numpy())
        test_recon_position_original = position_scaler.inverse_transform(test_recon_positions.numpy())
    else:
        test_position_actual = test_position_norm.cpu().numpy()
        test_recon_position_original = test_recon_positions.numpy()

    # Invert normalization for momenta
    if momenta_scaler is not None:
        test_momenta_actual = momenta_scaler.inverse_transform(test_momenta_norm.cpu().numpy())
    else:
        test_momenta_actual = test_momenta_norm.cpu().numpy()

    # Create a DataFrame
    # Define the number of dimensions
    INPUT_DIM = position.shape[1]  # Should be 9
    OUTPUT_DIM = momenta.shape[1]  # Should be 9

    # Create column names for actual positions, momenta, and reconstructed positions
    position_cols = [f'Position_Actual_{i+1}' for i in range(INPUT_DIM)]
    momenta_cols = [f'Momenta_{i+1}' for i in range(OUTPUT_DIM)]
    recon_cols = [f'Position_Reconstructed_{i+1}' for i in range(INPUT_DIM)]

    # Initialize a dictionary to hold the data
    data_dict = {}

    # Populate the dictionary with actual positions
    for i, col in enumerate(position_cols):
        data_dict[col] = test_position_actual[:, i]

    # Populate the dictionary with momenta
    for i, col in enumerate(momenta_cols):
        data_dict[col] = test_momenta_actual[:, i]

    # Populate the dictionary with reconstructed positions
    for i, col in enumerate(recon_cols):
        data_dict[col] = test_recon_position_original[:, i]

    # Create the DataFrame
    test_results_df = pd.DataFrame(data_dict)

    # Specify the file path where you want to save the CSV within the run directory
    OUTPUT_CSV_PATH = os.path.join(output_dir, 'test_result.csv')  # Save within run directory

    # Save the DataFrame to a CSV file
    test_results_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Run {run_index+1}: Test results saved to {OUTPUT_CSV_PATH}")

    # =======================
    # Post-Training Analysis
    # =======================

    # Read the reconstructed positions from the saved CSV
    recon_df = pd.read_csv(OUTPUT_CSV_PATH)
    cx = recon_df['Position_Reconstructed_1']
    cy = recon_df['Position_Reconstructed_2']
    cz = recon_df['Position_Reconstructed_3']
    ox = recon_df['Position_Reconstructed_4']
    oy = recon_df['Position_Reconstructed_5']
    oz = recon_df['Position_Reconstructed_6']
    sx = recon_df['Position_Reconstructed_7']
    sy = recon_df['Position_Reconstructed_8']
    sz = recon_df['Position_Reconstructed_9']

    pos_vars = [cx, cy, cz, ox, oy, oz, sx, sy, sz]
    labels = ['Carbon_X', 'Carbon_Y', 'Carbon_Z',
              'Oxygen_X', 'Oxygen_Y', 'Oxygen_Z',
              'Sulfur_X', 'Sulfur_Y', 'Sulfur_Z']

    ad_test_results = []
    distribution_stats = []

    for i in range(9):
        var = pos_vars[i].values
        label = labels[i]

        # Anderson-Darling test and plot
        ad_result, mean, std = plot_distribution(var, label, output_dir)
        ad_test_results.append(ad_result)
        distribution_stats.append({"label": label, "mean": mean, "std": std})

    # Save Anderson-Darling test results and distribution stats to JSON
    analysis_results = {
        "anderson_darling_tests": ad_test_results,
        "distribution_statistics": distribution_stats
    }

    analysis_json_path = os.path.join(output_dir, 'post_training_analysis.json')
    save_json(analysis_results, analysis_json_path)
    print(f"Run {run_index+1}: Post-training analysis saved to {analysis_json_path}")

    # =======================
    # Save Metrics and Hyperparameters to JSON
    # =======================

    # Prepare metrics with MREs as percentages
    results = {
        "metrics_test_mre": f"{test_mre * 100:.2f}%",
        "metrics_test_mse": test_mse,
        "metrics_train_mre": f"{train_mre * 100:.2f}%",
        "metrics_train_mse": train_mse,
        "metrics_val_mre": f"{val_mre * 100:.2f}%",
        "metrics_val_mse": val_mse,
        "params_BATCH_SIZE": hyperparams.get("batch_size", 256),
        "params_EPOCHS": hyperparams.get("epochs", 100),
        "params_L1_LAMBDA": hyperparams.get("l1_lambda", 0),
        "params_L2_LAMBDA": hyperparams.get("l2_lambda", 0),
        "params_LATENT_DIM": hyperparams.get("latent_dim", 512),
        "params_LEARNING_RATE": hyperparams.get("learning_rate", 0.001),
        "params_MIN_DELTA": hyperparams.get("min_delta", 0.001),
        "params_PATIENCE": hyperparams.get("patience", 10),
        "params_activation_name": activation_name,
        "params_hidden_layer_size": hyperparams.get("hidden_layer_size", 256),
        "params_momenta_norm_method": momenta_norm_method,
        "params_num_hidden_layers": hyperparams.get("num_hidden_layers", 3),
        "params_position_norm_method": position_norm_method,
        "params_use_l1": hyperparams.get("use_l1", False),
        "params_use_l2": hyperparams.get("use_l2", False)
    }

    # Include additional hyperparameters if present
    if hyperparams.get("use_l1", False) and "l1_lambda" in hyperparams:
        results["params_l1_lambda"] = hyperparams["l1_lambda"]
    if hyperparams.get("use_l2", False) and "l2_lambda" in hyperparams:
        results["params_l2_lambda"] = hyperparams["l2_lambda"]

    # Specify the JSON file path within the run directory
    JSON_OUTPUT_PATH = os.path.join(output_dir, 'training_results.json')

    # Save the dictionary to a JSON file
    save_json(results, JSON_OUTPUT_PATH)

    print(f"Run {run_index+1}: Training metrics and hyperparameters saved to {JSON_OUTPUT_PATH}")

    print(f"Run {run_index+1} completed.\n{'-'*50}")

# =======================
# Main Execution with Multiprocessing
# =======================

def main():
    # Determine the number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Only {num_gpus} GPU(s) detected. This script is optimized for 2 GPUs.")
    else:
        print(f"{num_gpus} GPUs detected. Utilizing GPUs 0 and 1.")

    # Assign GPUs in a round-robin fashion
    gpu_ids = [0, 1] if num_gpus >= 2 else [0] * len(hyperparameter_sets)

    # Function to pair hyperparameter sets with GPU IDs
    def get_gpu_id(run_idx):
        if num_gpus >= 2:
            return gpu_ids[run_idx % 2]
        elif num_gpus == 1:
            return 0
        else:
            raise RuntimeError("No GPUs detected. Please ensure at least one GPU is available.")

    # Prepare arguments for each run
    runs = []
    for run_idx, hyperparams in enumerate(hyperparameter_sets):
        gpu_id = get_gpu_id(run_idx)
        runs.append((hyperparams, run_idx, gpu_id))

    # Define the maximum number of parallel workers (2 for 2 GPUs)
    max_workers = min(2, len(runs)) if num_gpus >= 2 else min(1, len(runs))

    # Use ProcessPoolExecutor to run experiments in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all runs to the executor
        futures = [executor.submit(run_experiment, *run) for run in runs]

        # Optionally, track progress
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred during a run: {e}")

    print("\nAll runs completed.")

if __name__ == '__main__':
    # Set the start method to 'spawn' to prevent issues on some platforms
    multiprocessing.set_start_method('spawn')
    main()
