import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
    Normalizer,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars
import time
import seaborn as sns
import sys

# ===============================
# 1. Configuration and Hyperparameters
# ===============================

# Set random seeds for reproducibility
def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(0)

# Hyperparameters and configuration
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/Data/random_million_orient.csv'  # Replace with your actual file path
RESULT_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/CVAE/Energy/Best/Rand'  # Replace with your desired result path

save_dir = RESULT_PATH
csv_path = os.path.join(save_dir, "test_predictions.csv")

TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 512  # Adjust batch size as needed
EPOCHS = 50
LEARNING_RATE = 1e-5  # Adjust learning rate as needed

# Loss Weights
MSE_WEIGHT = 0.11767020152970051
KLD_WEIGHT = 10.220479086664307
MRE2_WEIGHT = 0.00025690809650127846
ENERGY_DIFF_WEIGHT = 0.0005698338123047986

# Regularization Parameters
use_l1 = False       # Flag to enable/disable L1 regularization
use_l2 = True        # Flag to enable/disable L2 regularization
l1_lambda = 0.0      # L1 regularization strength
l2_lambda = 0.001    # L2 regularization strength

# Early Stopping Parameters
PATIENCE = 5
MIN_DELTA = 1e-2
SAMPLES_TO_PRINT = 5  # Number of random samples to print after training

# Model hyperparameters
HIDDEN_DIM_SIZE = 16  # Base hidden dimension size
NUM_HIDDEN_LAYERS = 4  # Number of hidden layers
HIDDEN_DIMS = [HIDDEN_DIM_SIZE * (2 ** i) for i in range(NUM_HIDDEN_LAYERS)]  # [32, 64, 128, 256]
LATENT_DIM = 32  # Latent dimension size

# Activation function options
ACTIVATION_FUNCTION = 'tanh'  # Options: 'relu', 'tanh', 'sigmoid', etc.

# Normalization method options for positions and momenta
# Available options: 'minmax', 'standard', 'maxabs', 'robust', 'quantile_uniform', 'quantile_normal', 'power', 'normalize', None
POSITION_NORMALIZATION_METHOD = 'standard'  # Set to None if no normalization is desired for positions
MOMENTA_NORMALIZATION_METHOD = 'standard'     # Set to None if no normalization is desired for momenta

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable CUDA benchmark for performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Mixed Precision Training
use_mixed_precision = False  # Set to True to enable mixed-precision training

# Masses
mC = 21894.71361
mO = 29164.39289
mS = 58441.80487
mass = {'C': mC, 'O': mO, 'S': mS}

# Atom indices
atom_indices = {'C': [0, 1, 2], 'O': [3, 4, 5], 'S': [6, 7, 8]}

# Small epsilon to prevent division by zero
epsilon = 1e-10

# Create result directory if it doesn't exist
os.makedirs(RESULT_PATH, exist_ok=True)

# ===============================
# 2. Data Loading and Preprocessing
# ===============================

# Load data
print("Loading data...")
data = pd.read_csv(FILEPATH)
positions = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values.astype(np.float32)
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values.astype(np.float32)

# Split data into train, validation, test sets
print("Splitting data...")
positions_train, positions_temp, momenta_train, momenta_temp = train_test_split(
    positions, momenta, test_size=(TEST_SIZE + VAL_SIZE), random_state=42
)
val_size_adjusted = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
positions_val, positions_test, momenta_val, momenta_test = train_test_split(
    positions_temp, momenta_temp, test_size=val_size_adjusted, random_state=42
)

# Normalization (ensure that the normalization is invertible)
print("Applying normalization...")

def get_scaler(method):
    if method == 'minmax':
        return MinMaxScaler()
    elif method == 'standard':
        return StandardScaler()
    elif method == 'maxabs':
        return MaxAbsScaler()
    elif method == 'robust':
        return RobustScaler()
    elif method == 'quantile_uniform':
        return QuantileTransformer(output_distribution='uniform')
    elif method == 'quantile_normal':
        return QuantileTransformer(output_distribution='normal')
    elif method == 'power':
        return PowerTransformer()
    elif method == 'normalize':
        return Normalizer()
    elif method is None:
        return None
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

# Define normalization for positions
scaler_pos = get_scaler(POSITION_NORMALIZATION_METHOD)
# Define normalization for momenta
scaler_mom = get_scaler(MOMENTA_NORMALIZATION_METHOD)

# Apply normalization to positions
if scaler_pos is not None:
    positions_train = scaler_pos.fit_transform(positions_train)
    positions_val = scaler_pos.transform(positions_val)
    positions_test = scaler_pos.transform(positions_test)
    # Store mean and scale for inverse transformation
    if hasattr(scaler_pos, 'mean_'):
        mean_pos = scaler_pos.mean_
    elif hasattr(scaler_pos, 'center_'):
        mean_pos = scaler_pos.center_
    else:
        mean_pos = np.zeros(positions_train.shape[1], dtype=np.float32)

    if hasattr(scaler_pos, 'scale_'):
        scale_pos = scaler_pos.scale_
    else:
        scale_pos = np.ones(positions_train.shape[1], dtype=np.float32)
else:
    mean_pos = np.zeros(positions_train.shape[1], dtype=np.float32)
    scale_pos = np.ones(positions_train.shape[1], dtype=np.float32)

# Apply normalization to momenta
if scaler_mom is not None:
    momenta_train = scaler_mom.fit_transform(momenta_train)
    momenta_val = scaler_mom.transform(momenta_val)
    momenta_test = scaler_mom.transform(momenta_test)
    # Store mean and scale for inverse transformation
    if hasattr(scaler_mom, 'mean_'):
        mean_mom = scaler_mom.mean_
    elif hasattr(scaler_mom, 'center_'):
        mean_mom = scaler_mom.center_
    else:
        mean_mom = np.zeros(momenta_train.shape[1], dtype=np.float32)

    if hasattr(scaler_mom, 'scale_'):
        scale_mom = scaler_mom.scale_
    else:
        scale_mom = np.ones(momenta_train.shape[1], dtype=np.float32)
else:
    mean_mom = np.zeros(momenta_train.shape[1], dtype=np.float32)
    scale_mom = np.ones(momenta_train.shape[1], dtype=np.float32)

# Verify data integrity after normalization
print("Verifying data integrity...")
def check_data_integrity(array, name):
    if np.isnan(array).any():
        raise ValueError(f"NaN values found in {name}")
    if np.isinf(array).any():
        raise ValueError(f"Inf values found in {name}")

check_data_integrity(positions_train, "positions_train")
check_data_integrity(positions_val, "positions_val")
check_data_integrity(positions_test, "positions_test")
check_data_integrity(momenta_train, "momenta_train")
check_data_integrity(momenta_val, "momenta_val")
check_data_integrity(momenta_test, "momenta_test")

# Convert to torch tensors
positions_train_tensor = torch.from_numpy(positions_train)
momenta_train_tensor = torch.from_numpy(momenta_train)

positions_val_tensor = torch.from_numpy(positions_val)
momenta_val_tensor = torch.from_numpy(momenta_val)

positions_test_tensor = torch.from_numpy(positions_test)
momenta_test_tensor = torch.from_numpy(momenta_test)

# Create TensorDatasets
train_dataset = TensorDataset(positions_train_tensor, momenta_train_tensor)
val_dataset = TensorDataset(positions_val_tensor, momenta_val_tensor)
test_dataset = TensorDataset(positions_test_tensor, momenta_test_tensor)

# Create DataLoaders with optimized settings
print("Creating DataLoaders...")
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=8, pin_memory=True, prefetch_factor=4
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=8, pin_memory=True, prefetch_factor=4
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=8, pin_memory=True, prefetch_factor=4
)

# Convert mean and scale to torch tensors
mean_pos = torch.tensor(mean_pos, device=device, dtype=torch.float32)
scale_pos = torch.tensor(scale_pos, device=device, dtype=torch.float32)
mean_mom = torch.tensor(mean_mom, device=device, dtype=torch.float32)
scale_mom = torch.tensor(scale_mom, device=device, dtype=torch.float32)

# ===============================
# 3. Model Definition
# ===============================

class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dims, activation_function):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Select activation function
        activation_dict = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),  # Swish is also known as SiLU in PyTorch
            'prelu': nn.PReLU(),
        }
        if activation_function not in activation_dict:
            raise ValueError("Unsupported activation function")
        self.activation = activation_dict[activation_function]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim + condition_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(self.activation)
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim + condition_dim
        reversed_hidden_dims = hidden_dims[::-1]
        for h_dim in reversed_hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(self.activation)
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x, y):
        # x: positions, y: momenta
        xy = torch.cat([x, y], dim=1)
        h = self.encoder(xy)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        x_recon = self.decoder(zy)
        return x_recon

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        # Clamp logvar to prevent extreme values
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar

# Initialize the model
input_dim = positions.shape[1]  # Number of position features
condition_dim = momenta.shape[1]  # Number of momentum features
model = CVAE(input_dim, condition_dim, LATENT_DIM, HIDDEN_DIMS, ACTIVATION_FUNCTION).to(device)

# ===============================
# 4. Optimizer and Loss Function
# ===============================

# Configure optimizer with conditional L1 and L2 regularization
l1_lambda_effective = l1_lambda if use_l1 else 0.0
l2_lambda_effective = l2_lambda if use_l2 else 0.0

optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=l2_lambda_effective)

if use_l1:
    print(f"L1 Regularization Enabled with lambda={l1_lambda_effective}")
else:
    print("L1 Regularization Disabled")

if use_l2:
    print(f"L2 Regularization Enabled with lambda={l2_lambda_effective}")
else:
    print("L2 Regularization Disabled")

# Loss scaler for mixed precision
if use_mixed_precision:
    scaler_fp16 = torch.cuda.amp.GradScaler()
    print("Mixed Precision Training Enabled")
else:
    scaler_fp16 = None
    print("Mixed Precision Training Disabled")

# Early stopping
class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            print(f"Validation loss improved to {self.best_loss:.4f}")
        else:
            self.counter +=1
            print(f"No improvement in validation loss for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

# ===============================
# 5. Training Loop
# ===============================

train_losses = []
val_losses = []

# Initialize lists for additional metrics
train_mre = []
val_mre = []
train_energy_diff = []
val_energy_diff = []
train_recon_loss = []
val_recon_loss = []
train_kld = []
val_kld = []

# For computing mean and variance of z over the training set
mu_sum = torch.zeros(LATENT_DIM, device=device)
mu_square_sum = torch.zeros(LATENT_DIM, device=device)
total_samples = 0

print("Starting training...")
start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_unreg_loss = 0  # Loss without regularization
    total_recon_loss = 0
    total_kld_loss = 0
    total_mre2_loss = 0
    total_l1_loss = 0  # To track L1 loss if used
    total_energy_loss = 0  # To track Energy Loss

    # Initialize accumulators for additional metrics
    epoch_train_mre = 0.0
    epoch_train_energy_diff = 0.0
    epoch_train_recon_loss = 0.0
    epoch_train_kld = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training")
    for x, y in progress_bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                x_recon, mu, logvar = model(x, y)
                # Monitor for NaNs in intermediate outputs
                if torch.isnan(mu).any() or torch.isnan(logvar).any() or torch.isnan(x_recon).any():
                    print("NaN detected in mu, logvar, or x_recon. Stopping training.")
                    exit(1)  # Exit or handle appropriately

                # Reconstruction loss
                recon_loss = F.mse_loss(x_recon, x, reduction='mean')

                # KL divergence
                kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                # MRE^2 Loss
                mre2_loss = torch.mean(((x - x_recon) / (torch.abs(x) + epsilon)) ** 2)

                # Inverse transform x_recon and y to original scale
                x_recon_original = x_recon * scale_pos + mean_pos
                y_original = y * scale_mom + mean_mom

                # Compute Energy Loss
                # KE_total from y_original (momenta)
                momenta_C = y_original[:, 0:3]
                momenta_O = y_original[:, 3:6]
                momenta_S = y_original[:, 6:9]

                KE_C = torch.sum(momenta_C ** 2, dim=1) / (2 * mC)
                KE_O = torch.sum(momenta_O ** 2, dim=1) / (2 * mO)
                KE_S = torch.sum(momenta_S ** 2, dim=1) / (2 * mS)

                KE_total = KE_C + KE_O + KE_S  # Shape [batch_size]

                # PE_pred from x_recon_original (predicted positions)
                positions_C = x_recon_original[:, 0:3]
                positions_O = x_recon_original[:, 3:6]
                positions_S = x_recon_original[:, 6:9]

                rCO = torch.norm(positions_C - positions_O, dim=1)
                rCS = torch.norm(positions_C - positions_S, dim=1)
                rOS = torch.norm(positions_O - positions_S, dim=1)

                # Prevent division by zero
                rCO = torch.clamp(rCO, min=epsilon)
                rCS = torch.clamp(rCS, min=epsilon)
                rOS = torch.clamp(rOS, min=epsilon)

                PE_pred = (4 / rCO) + (4 / rCS) + (4 / rOS)  # Shape [batch_size]

                # Compute Energy Difference
                EnergyDiff = (KE_total - PE_pred) / (torch.abs(KE_total) + epsilon)
                EnergyLoss = torch.mean(EnergyDiff ** 2)

                # Unregularized total loss
                unreg_loss = (MSE_WEIGHT * recon_loss +
                              KLD_WEIGHT * kld_loss +
                              MRE2_WEIGHT * mre2_loss +
                              ENERGY_DIFF_WEIGHT * EnergyLoss)

                # Total loss with L1 regularization if enabled
                if use_l1:
                    l1_loss = sum(p.abs().sum() for p in model.parameters())
                    loss = unreg_loss + l1_lambda_effective * l1_loss
                    total_l1_loss += l1_loss.item()
                else:
                    loss = unreg_loss

            scaler_fp16.scale(loss).backward()
            scaler_fp16.step(optimizer)
            scaler_fp16.update()
        else:
            x_recon, mu, logvar = model(x, y)
            # Monitor for NaNs in intermediate outputs
            if torch.isnan(mu).any() or torch.isnan(logvar).any() or torch.isnan(x_recon).any():
                print("NaN detected in mu, logvar, or x_recon. Stopping training.")
                exit(1)  # Exit or handle appropriately

            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x, reduction='mean')

            # KL divergence
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # MRE^2 Loss
            mre2_loss = torch.mean(((x - x_recon) / (torch.abs(x) + epsilon)) ** 2)

            # Inverse transform x_recon and y to original scale
            x_recon_original = x_recon * scale_pos + mean_pos
            y_original = y * scale_mom + mean_mom

            # Compute Energy Loss
            # KE_total from y_original (momenta)
            momenta_C = y_original[:, 0:3]
            momenta_O = y_original[:, 3:6]
            momenta_S = y_original[:, 6:9]

            KE_C = torch.sum(momenta_C ** 2, dim=1) / (2 * mC)
            KE_O = torch.sum(momenta_O ** 2, dim=1) / (2 * mO)
            KE_S = torch.sum(momenta_S ** 2, dim=1) / (2 * mS)

            KE_total = KE_C + KE_O + KE_S  # Shape [batch_size]

            # PE_pred from x_recon_original (predicted positions)
            positions_C = x_recon_original[:, 0:3]
            positions_O = x_recon_original[:, 3:6]
            positions_S = x_recon_original[:, 6:9]

            rCO = torch.norm(positions_C - positions_O, dim=1)
            rCS = torch.norm(positions_C - positions_S, dim=1)
            rOS = torch.norm(positions_O - positions_S, dim=1)

            # Prevent division by zero
            rCO = torch.clamp(rCO, min=epsilon)
            rCS = torch.clamp(rCS, min=epsilon)
            rOS = torch.clamp(rOS, min=epsilon)

            PE_pred = (4 / rCO) + (4 / rCS) + (4 / rOS)  # Shape [batch_size]

            # Compute Energy Difference
            EnergyDiff = (KE_total - PE_pred) / (torch.abs(KE_total) + epsilon)
            EnergyLoss = torch.mean(EnergyDiff ** 2)

            # Unregularized total loss
            unreg_loss = (MSE_WEIGHT * recon_loss +
                          KLD_WEIGHT * kld_loss +
                          MRE2_WEIGHT * mre2_loss +
                          ENERGY_DIFF_WEIGHT * EnergyLoss)

            # Total loss with L1 regularization if enabled
            if use_l1:
                l1_loss = sum(p.abs().sum() for p in model.parameters())
                loss = unreg_loss + l1_lambda_effective * l1_loss
                total_l1_loss += l1_loss.item()
            else:
                loss = unreg_loss

            loss.backward()
            optimizer.step()

        # Accumulate losses
        total_loss += loss.item() * x.size(0)
        total_unreg_loss += unreg_loss.item() * x.size(0)
        total_recon_loss += recon_loss.item() * x.size(0)
        total_kld_loss += kld_loss.item() * x.size(0)
        total_mre2_loss += mre2_loss.item() * x.size(0)
        total_energy_loss += EnergyLoss.item() * x.size(0)

        # Accumulate mu for computing mean and variance
        mu_sum += mu.sum(dim=0)
        mu_square_sum += (mu ** 2).sum(dim=0)
        total_samples += x.size(0)

        # Accumulate metrics for this epoch
        epoch_train_recon_loss += recon_loss.item() * x.size(0)
        epoch_train_kld += kld_loss.item() * x.size(0)
        epoch_train_mre += mre2_loss.item() * x.size(0)
        epoch_train_energy_diff += EnergyLoss.item() * x.size(0)

        # Update progress bar
        if use_l1:
            progress_bar.set_postfix({
                'Total Loss': f"{(total_unreg_loss / ((progress_bar.n + 1) * BATCH_SIZE)):.4f}",
                'Recon Loss': f"{(total_recon_loss / ((progress_bar.n + 1) * BATCH_SIZE)):.4f}",
                'KLD Loss': f"{(total_kld_loss / ((progress_bar.n + 1) * BATCH_SIZE)):.4f}",
                'MRE2 Loss': f"{(total_mre2_loss / ((progress_bar.n + 1) * BATCH_SIZE)):.4f}",
                'Energy Loss': f"{(total_energy_loss / ((progress_bar.n + 1) * BATCH_SIZE)):.4f}",
                'L1 Loss': f"{(total_l1_loss / ((progress_bar.n + 1) * BATCH_SIZE)):.4f}"
            })
        else:
            progress_bar.set_postfix({
                'Total Loss': f"{(total_unreg_loss / ((progress_bar.n + 1) * BATCH_SIZE)):.4f}",
                'Recon Loss': f"{(total_recon_loss / ((progress_bar.n + 1) * BATCH_SIZE)):.4f}",
                'KLD Loss': f"{(total_kld_loss / ((progress_bar.n + 1) * BATCH_SIZE)):.4f}",
                'MRE2 Loss': f"{(total_mre2_loss / ((progress_bar.n + 1) * BATCH_SIZE)):.4f}",
                'Energy Loss': f"{(total_energy_loss / ((progress_bar.n + 1) * BATCH_SIZE)):.4f}",
            })

    # Compute average metrics for this epoch
    avg_train_loss = total_unreg_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    avg_train_recon_loss = epoch_train_recon_loss / len(train_loader.dataset)
    train_recon_loss.append(avg_train_recon_loss)

    avg_train_kld = epoch_train_kld / len(train_loader.dataset)
    train_kld.append(avg_train_kld)

    avg_train_mre = epoch_train_mre / len(train_loader.dataset)
    train_mre.append(avg_train_mre)

    avg_train_energy_diff = epoch_train_energy_diff / len(train_loader.dataset)
    train_energy_diff.append(avg_train_energy_diff)

    # Validation
    model.eval()
    total_val_loss = 0
    total_val_energy_loss = 0

    # Initialize accumulators for validation metrics
    epoch_val_recon_loss = 0.0
    epoch_val_kld = 0.0
    epoch_val_mre = 0.0
    epoch_val_energy_diff = 0.0

    with torch.no_grad():
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation")
        for x_val, y_val in progress_bar_val:
            x_val = x_val.to(device, non_blocking=True)
            y_val = y_val.to(device, non_blocking=True)

            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    x_recon, mu, logvar = model(x_val, y_val)
                    # Monitor for NaNs in validation
                    if torch.isnan(mu).any() or torch.isnan(logvar).any() or torch.isnan(x_recon).any():
                        print("NaN detected in validation mu, logvar, or x_recon. Stopping training.")
                        exit(1)  # Exit or handle appropriately

                    # Reconstruction loss
                    recon_loss = F.mse_loss(x_recon, x_val, reduction='mean')

                    # KL divergence
                    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                    # MRE^2 Loss
                    mre2_loss = torch.mean(((x_val - x_recon) / (torch.abs(x_val) + epsilon)) ** 2)

                    # Inverse transform x_recon and y_val to original scale
                    x_recon_original = x_recon * scale_pos + mean_pos
                    y_original = y_val * scale_mom + mean_mom

                    # Compute Energy Loss
                    # KE_total from y_original (momenta)
                    momenta_C = y_original[:, 0:3]
                    momenta_O = y_original[:, 3:6]
                    momenta_S = y_original[:, 6:9]

                    KE_C = torch.sum(momenta_C ** 2, dim=1) / (2 * mC)
                    KE_O = torch.sum(momenta_O ** 2, dim=1) / (2 * mO)
                    KE_S = torch.sum(momenta_S ** 2, dim=1) / (2 * mS)

                    KE_total = KE_C + KE_O + KE_S  # Shape [batch_size]

                    # PE_pred from x_recon_original (predicted positions)
                    positions_C = x_recon_original[:, 0:3]
                    positions_O = x_recon_original[:, 3:6]
                    positions_S = x_recon_original[:, 6:9]

                    rCO = torch.norm(positions_C - positions_O, dim=1)
                    rCS = torch.norm(positions_C - positions_S, dim=1)
                    rOS = torch.norm(positions_O - positions_S, dim=1)

                    # Prevent division by zero
                    rCO = torch.clamp(rCO, min=epsilon)
                    rCS = torch.clamp(rCS, min=epsilon)
                    rOS = torch.clamp(rOS, min=epsilon)

                    PE_pred = (4 / rCO) + (4 / rCS) + (4 / rOS)  # Shape [batch_size]

                    # Compute Energy Difference
                    EnergyDiff = (KE_total - PE_pred) / (torch.abs(KE_total) + epsilon)
                    EnergyLoss = torch.mean(EnergyDiff ** 2)

                    # Unregularized total loss
                    unreg_loss = (MSE_WEIGHT * recon_loss +
                                  KLD_WEIGHT * kld_loss +
                                  MRE2_WEIGHT * mre2_loss +
                                  ENERGY_DIFF_WEIGHT * EnergyLoss)
            else:
                x_recon, mu, logvar = model(x_val, y_val)
                # Monitor for NaNs in validation
                if torch.isnan(mu).any() or torch.isnan(logvar).any() or torch.isnan(x_recon).any():
                    print("NaN detected in validation mu, logvar, or x_recon. Stopping training.")
                    exit(1)  # Exit or handle appropriately

                # Reconstruction loss
                recon_loss = F.mse_loss(x_recon, x_val, reduction='mean')

                # KL divergence
                kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                # MRE^2 Loss
                mre2_loss = torch.mean(((x_val - x_recon) / (torch.abs(x_val) + epsilon)) ** 2)

                # Inverse transform x_recon and y_val to original scale
                x_recon_original = x_recon * scale_pos + mean_pos
                y_original = y_val * scale_mom + mean_mom

                # Compute Energy Loss
                # KE_total from y_original (momenta)
                momenta_C = y_original[:, 0:3]
                momenta_O = y_original[:, 3:6]
                momenta_S = y_original[:, 6:9]

                KE_C = torch.sum(momenta_C ** 2, dim=1) / (2 * mC)
                KE_O = torch.sum(momenta_O ** 2, dim=1) / (2 * mO)
                KE_S = torch.sum(momenta_S ** 2, dim=1) / (2 * mS)

                KE_total = KE_C + KE_O + KE_S  # Shape [batch_size]

                # PE_pred from x_recon_original (predicted positions)
                positions_C = x_recon_original[:, 0:3]
                positions_O = x_recon_original[:, 3:6]
                positions_S = x_recon_original[:, 6:9]

                rCO = torch.norm(positions_C - positions_O, dim=1)
                rCS = torch.norm(positions_C - positions_S, dim=1)
                rOS = torch.norm(positions_O - positions_S, dim=1)

                # Prevent division by zero
                rCO = torch.clamp(rCO, min=epsilon)
                rCS = torch.clamp(rCS, min=epsilon)
                rOS = torch.clamp(rOS, min=epsilon)

                PE_pred = (4 / rCO) + (4 / rCS) + (4 / rOS)  # Shape [batch_size]

                # Compute Energy Difference
                EnergyDiff = (KE_total - PE_pred) / (torch.abs(KE_total) + epsilon)
                EnergyLoss = torch.mean(EnergyDiff ** 2)

                # Unregularized total loss
                unreg_loss = (MSE_WEIGHT * recon_loss +
                              KLD_WEIGHT * kld_loss +
                              MRE2_WEIGHT * mre2_loss +
                              ENERGY_DIFF_WEIGHT * EnergyLoss)

            total_val_loss += unreg_loss.item() * x_val.size(0)
            total_val_energy_loss += EnergyLoss.item() * x_val.size(0)

            # Accumulate validation metrics for this epoch
            epoch_val_recon_loss += recon_loss.item() * x_val.size(0)
            epoch_val_kld += kld_loss.item() * x_val.size(0)
            epoch_val_mre += mre2_loss.item() * x_val.size(0)
            epoch_val_energy_diff += EnergyLoss.item() * x_val.size(0)

    # Compute average validation metrics for this epoch
    avg_val_loss = total_val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    avg_val_recon_loss = epoch_val_recon_loss / len(val_loader.dataset)
    val_recon_loss.append(avg_val_recon_loss)

    avg_val_kld = epoch_val_kld / len(val_loader.dataset)
    val_kld.append(avg_val_kld)

    avg_val_mre = epoch_val_mre / len(val_loader.dataset)
    val_mre.append(avg_val_mre)

    avg_val_energy_diff = epoch_val_energy_diff / len(val_loader.dataset)
    val_energy_diff.append(avg_val_energy_diff)

    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Early stopping
    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(RESULT_PATH, f'cvae_model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

# Compute mean and variance of latent variables over the training set
mu_train_mean = mu_sum / total_samples
var_train = (mu_square_sum / total_samples) - mu_train_mean ** 2
std_train = torch.sqrt(var_train + epsilon)

# Save mu_train_mean, std_train, and normalization parameters
torch.save({
    'mu_train_mean': mu_train_mean.cpu(),
    'std_train': std_train.cpu(),
    'mean_pos': mean_pos.cpu(),
    'scale_pos': scale_pos.cpu(),
    'mean_mom': mean_mom.cpu(),
    'scale_mom': scale_mom.cpu()
}, os.path.join(RESULT_PATH, 'latent_stats.pth'))

# Save the final trained model
torch.save(model.state_dict(), os.path.join(RESULT_PATH, 'cvae_model_final.pth'))
print("Training complete and model saved.")

# ===============================
# 6. Plot Learning Curves and Metrics
# ===============================

print("Plotting learning curves and error metrics...")
epochs_range = range(1, len(train_losses)+1)

# Plot for the first 10 epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs_range[:10], train_losses[:10], label='Train Loss')
plt.plot(epochs_range[:10], val_losses[:10], label='Validation Loss')
plt.title('Learning Curve (First 10 Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, 'learning_curve_first_10_epochs.png'))
plt.close()

# Plot for the remaining epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs_range[10:], train_losses[10:], label='Train Loss')
plt.plot(epochs_range[10:], val_losses[10:], label='Validation Loss')
plt.title('Learning Curve (Epochs 11 onwards)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, 'learning_curve_remaining_epochs.png'))
plt.close()

# ===============================
# 7. Plot Additional Error Metrics
# ===============================

# Function to plot and save a metric
def plot_metric(metric_train, metric_val, metric_name, ylabel, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, metric_train, label='Train')
    plt.plot(epochs_range, metric_val, label='Validation')
    plt.title(f'{metric_name} Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, filename))
    plt.close()

# Plot Mean Relative Error (MRE)
plot_metric(
    train_mre,
    val_mre,
    metric_name='Mean Relative Error (MRE)',
    ylabel='MRE (squared)',
    filename='mre_over_epochs.png'
)

# Plot Energy Differences
plot_metric(
    train_energy_diff,
    val_energy_diff,
    metric_name='Energy Differences',
    ylabel='Energy Difference (squared)',
    filename='energy_diff_over_epochs.png'
)

# Plot Reconstruction Loss (MSE)
plot_metric(
    train_recon_loss,
    val_recon_loss,
    metric_name='Reconstruction Loss (MSE)',
    ylabel='MSE',
    filename='recon_loss_over_epochs.png'
)

# Plot Kullback-Leibler Divergence (KLD)
plot_metric(
    train_kld,
    val_kld,
    metric_name='Kullback-Leibler Divergence (KLD)',
    ylabel='KLD',
    filename='kld_over_epochs.png'
)

print("Additional error metrics plots saved.")

# ===============================
# 8. Evaluation on Test Set
# ===============================

print("Evaluating on test set...")
model.eval()
with torch.no_grad():
    # Load mu_train_mean, std_train, and normalization parameters
    latent_stats = torch.load(os.path.join(RESULT_PATH, 'latent_stats.pth'), map_location=device)
    mu_train_mean = latent_stats['mu_train_mean'].to(device)
    std_train = latent_stats['std_train'].to(device)
    mean_pos = latent_stats['mean_pos'].to(device)
    scale_pos = latent_stats['scale_pos'].to(device)
    mean_mom = latent_stats['mean_mom'].to(device)
    scale_mom = latent_stats['scale_mom'].to(device)

    # Move test tensors to device
    positions_test_tensor = positions_test_tensor.to(device)
    momenta_test_tensor = momenta_test_tensor.to(device)

    # Sample z from training latent distribution
    z = torch.randn(len(test_dataset), LATENT_DIM, device=device) * std_train + mu_train_mean
    y_test_tensor = momenta_test_tensor  # momenta_test_tensor is already on device

    # Inverse transform y_test_tensor to original scale
    y_test_original = y_test_tensor * scale_mom + mean_mom

    if use_mixed_precision:
        with torch.cuda.amp.autocast():
            x_pred = model.decode(z, y_test_tensor)
    else:
        x_pred = model.decode(z, y_test_tensor)

    # Inverse transform x_pred to original scale
    x_pred = x_pred * scale_pos + mean_pos

    x_pred = x_pred.cpu().numpy()

    # Inverse transform positions_test_tensor to original scale
    x_test_np = (positions_test_tensor * scale_pos + mean_pos).cpu().numpy()

    # Compute MRE and MSE
    MRE = np.mean(np.abs(x_test_np - x_pred) / (np.abs(x_test_np) + epsilon)) * 100
    MSE = np.mean((x_test_np - x_pred) ** 2)
    print(f'Average MRE: {MRE:.2f}%')
    print(f'Average MSE: {MSE:.6f}')

    # Energy calculations
    # Inverse transform momenta_test_tensor to original scale
    momenta_np = (momenta_test_tensor * scale_mom + mean_mom).cpu().numpy()

    # Split momenta into C, O, S
    momenta_C = momenta_np[:, 0:3]
    momenta_O = momenta_np[:, 3:6]
    momenta_S = momenta_np[:, 6:9]

    # Compute Kinetic Energy (KE)
    KE_C = np.sum(momenta_C ** 2, axis=1) / (2 * mC)
    KE_O = np.sum(momenta_O ** 2, axis=1) / (2 * mO)
    KE_S = np.sum(momenta_S ** 2, axis=1) / (2 * mS)
    KE_total = KE_C + KE_O + KE_S  # Shape [batch_size]

    # Compute Potential Energy (PE) for predicted positions
    positions_C_pred = x_pred[:, 0:3]
    positions_O_pred = x_pred[:, 3:6]
    positions_S_pred = x_pred[:, 6:9]

    rCO_pred = np.linalg.norm(positions_C_pred - positions_O_pred, axis=1)
    rCS_pred = np.linalg.norm(positions_C_pred - positions_S_pred, axis=1)
    rOS_pred = np.linalg.norm(positions_O_pred - positions_S_pred, axis=1)

    # To prevent division by zero, add epsilon
    rCO_pred = np.maximum(rCO_pred, epsilon)
    rCS_pred = np.maximum(rCS_pred, epsilon)
    rOS_pred = np.maximum(rOS_pred, epsilon)

    PE_pred = (4 / rCO_pred + 4 / rCS_pred + 4 / rOS_pred)  # Shape [batch_size]

    # Compute Energy Difference squared
    EnergyDiff = ((KE_total - PE_pred) / (np.abs(KE_total) + epsilon)) ** 2
    EnergyDiff_mean = np.mean(EnergyDiff)
    print(f'Average Energy Difference: {EnergyDiff_mean:.2e}')

    # Save metrics to a text file
    with open(os.path.join(RESULT_PATH, 'metrics.txt'), 'w') as f:
        f.write(f'Average MRE: {MRE:.2f}%\n')
        f.write(f'Average MSE: {MSE:.6f}\n')
        f.write(f'Average Energy Difference: {EnergyDiff_mean:.2e}\n')

    # Print S random samples from test set
    S = SAMPLES_TO_PRINT
    indices = np.random.choice(len(x_test_np), S, replace=False)
    with open(os.path.join(RESULT_PATH, 'sample_predictions.txt'), 'w') as f:
        for idx in indices:
            f.write(f'Sample {idx}:\n')
            f.write('Real Positions:\n')
            f.write(f'Carbon (C): ({x_test_np[idx,0]:.4f}, {x_test_np[idx,1]:.4f}, {x_test_np[idx,2]:.4f})\n')
            f.write(f'Oxygen (O): ({x_test_np[idx,3]:.4f}, {x_test_np[idx,4]:.4f}, {x_test_np[idx,5]:.4f})\n')
            f.write(f'Sulfur (S): ({x_test_np[idx,6]:.4f}, {x_test_np[idx,7]:.4f}, {x_test_np[idx,8]:.4f})\n')
            f.write('Predicted Positions:\n')
            f.write(f'Carbon (C): ({x_pred[idx,0]:.4f}, {x_pred[idx,1]:.4f}, {x_pred[idx,2]:.4f})\n')
            f.write(f'Oxygen (O): ({x_pred[idx,3]:.4f}, {x_pred[idx,4]:.4f}, {x_pred[idx,5]:.4f})\n')
            f.write(f'Sulfur (S): ({x_pred[idx,6]:.4f}, {x_pred[idx,7]:.4f}, {x_pred[idx,8]:.4f})\n')
            f.write('---\n')

    # Optionally, print the samples
    for idx in indices:
        print(f'Sample {idx}:')
        print('Real Positions:')
        print(f'Carbon (C): ({x_test_np[idx,0]:.4f}, {x_test_np[idx,1]:.4f}, {x_test_np[idx,2]:.4f})')
        print(f'Oxygen (O): ({x_test_np[idx,3]:.4f}, {x_test_np[idx,4]:.4f}, {x_test_np[idx,5]:.4f})')
        print(f'Sulfur (S): ({x_test_np[idx,6]:.4f}, {x_test_np[idx,7]:.4f}, {x_test_np[idx,8]:.4f})')
        print('Predicted Positions:')
        print(f'Carbon (C): ({x_pred[idx,0]:.4f}, {x_pred[idx,1]:.4f}, {x_pred[idx,2]:.4f})')
        print(f'Oxygen (O): ({x_pred[idx,3]:.4f}, {x_pred[idx,4]:.4f}, {x_pred[idx,5]:.4f})')
        print(f'Sulfur (S): ({x_pred[idx,6]:.4f}, {x_pred[idx,7]:.4f}, {x_pred[idx,8]:.4f})')
        print('---')

    # ===============================
    # 8.1. Save Test Predictions to CSV
    # ===============================

    print("Saving test predictions to CSV...")
    # Also get the original positions and momenta (un-normalized)
    positions_test_original = (positions_test_tensor * scale_pos + mean_pos).cpu().numpy()
    momenta_test_original = (momenta_test_tensor * scale_mom + mean_mom).cpu().numpy()

    # Prepare the DataFrame with the specified columns
    columns = [
        'cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz',
        'pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz',
        'pred_cx', 'pred_cy', 'pred_cz', 'pred_ox', 'pred_oy', 'pred_oz', 'pred_sx', 'pred_sy', 'pred_sz'
    ]

    data_dict = {}
    for idx, col in enumerate(['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']):
        data_dict[col] = positions_test_original[:, idx]
    for idx, col in enumerate(['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']):
        data_dict[col] = momenta_test_original[:, idx]
    for idx, col in enumerate(['pred_cx', 'pred_cy', 'pred_cz', 'pred_ox', 'pred_oy', 'pred_oz', 'pred_sx', 'pred_sy', 'pred_sz']):
        data_dict[col] = x_pred[:, idx]

    df = pd.DataFrame(data_dict, columns=columns)

    # Save the DataFrame to CSV
    output_csv_path = os.path.join(RESULT_PATH, 'test_predictions.csv')
    df.to_csv(output_csv_path, index=False)

    print(f"Test predictions saved to {output_csv_path}")

# ===============================
# 9. Save Configurations
# ===============================

csv_path = "/home/g/ghanaatian/MYFILES/FALL24/Physics/CVAE/Energy/Best/Sim/test_predictions.csv"
save_dir = "/home/g/ghanaatian/MYFILES/FALL24/Physics/CVAE/Energy/Best/Sim"

print("Saving configurations...")
with open(os.path.join(RESULT_PATH, 'configurations.txt'), 'w') as f:
    f.write('Hyperparameters and Configurations:\n')
    f.write(f'FILEPATH: {FILEPATH}\n')
    f.write(f'RESULT_PATH: {RESULT_PATH}\n')
    f.write(f'TEST_SIZE: {TEST_SIZE}\n')
    f.write(f'VAL_SIZE: {VAL_SIZE}\n')
    f.write(f'BATCH_SIZE: {BATCH_SIZE}\n')
    f.write(f'EPOCHS: {EPOCHS}\n')
    f.write(f'LEARNING_RATE: {LEARNING_RATE}\n')
    f.write(f'MSE_WEIGHT: {MSE_WEIGHT}\n')
    f.write(f'KLD_WEIGHT: {KLD_WEIGHT}\n')
    f.write(f'MRE2_WEIGHT: {MRE2_WEIGHT}\n')
    f.write(f'ENERGY_DIFF_WEIGHT: {ENERGY_DIFF_WEIGHT}\n')
    f.write(f'PATIENCE: {PATIENCE}\n')
    f.write(f'MIN_DELTA: {MIN_DELTA}\n')
    f.write(f'SAMPLES_TO_PRINT: {SAMPLES_TO_PRINT}\n')
    f.write(f'HIDDEN_DIM_SIZE: {HIDDEN_DIM_SIZE}\n')
    f.write(f'NUM_HIDDEN_LAYERS: {NUM_HIDDEN_LAYERS}\n')
    f.write(f'HIDDEN_DIMS: {HIDDEN_DIMS}\n')
    f.write(f'LATENT_DIM: {LATENT_DIM}\n')
    f.write(f'ACTIVATION_FUNCTION: {ACTIVATION_FUNCTION}\n')
    f.write(f'POSITION_NORMALIZATION_METHOD: {POSITION_NORMALIZATION_METHOD}\n')
    f.write(f'MOMENTA_NORMALIZATION_METHOD: {MOMENTA_NORMALIZATION_METHOD}\n')
    f.write(f'use_l1: {use_l1}\n')
    f.write(f'use_l2: {use_l2}\n')
    f.write(f'l1_lambda: {l1_lambda}\n')
    f.write(f'l2_lambda: {l2_lambda}\n')
    f.write(f'use_mixed_precision: {use_mixed_precision}\n')

print("All tasks completed successfully.")

# ===============================
# 9. Save Configurations
# ===============================

print("Plotting distribution plots...")

def read_csv_file(csv_path):
    """
    Reads the CSV file and returns a pandas DataFrame.
    Includes error handling for common issues.
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file {csv_path} does not exist.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print("Error: The CSV file is malformed.")
        sys.exit(1)


def validate_columns(df, columns):
    """
    Checks if all specified columns exist in the DataFrame.
    """
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        print(f"Error: The following columns are missing in the CSV file: {missing_columns}")
        sys.exit(1)


def create_distribution_plots(df, columns, save_path):
    """
    Creates a 3x3 grid of distribution plots (histograms with KDE) for each specified column.
    Saves the combined plot to the designated path.
    """
    # Set the style for better aesthetics
    sns.set(style="whitegrid")

    # Define the figure size; adjust as needed
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle("Prediction Columns Distribution Plots", fontsize=24)

    for idx, column in enumerate(columns):
        row = idx // 3
        col = idx % 3
        ax = axes[row][col]
        sns.histplot(df[column], kde=True, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(column, fontsize=16)
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Frequency")

    # Remove any empty subplots if the number of columns is not a multiple of 9
    total_plots = len(columns)
    if total_plots < 9:
        for idx in range(total_plots, 9):
            row = idx // 3
            col = idx % 3
            fig.delaxes(axes[row][col])

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    try:
        plt.savefig(save_path)
        print(f"Distribution plots successfully saved to {save_path}")
    except Exception as e:
        print(f"Error saving distribution plots: {e}")
    finally:
        plt.close()



# Define save path for the distribution plots
distribution_plot_save_path = os.path.join(save_dir, "predictions_distribution_plots.png")

# Ensure the save directory exists; if not, create it
os.makedirs(save_dir, exist_ok=True)

# Define the columns to plot
columns_to_plot = [
        "pred_cx", "pred_cy", "pred_cz",
        "pred_ox", "pred_oy", "pred_oz",
        "pred_sx", "pred_sy", "pred_sz"
    ]

# Read the CSV file
df = read_csv_file(csv_path)

# Validate that all required columns are present
validate_columns(df, columns_to_plot)

# Create and save the distribution plots
create_distribution_plots(df, columns_to_plot, distribution_plot_save_path)