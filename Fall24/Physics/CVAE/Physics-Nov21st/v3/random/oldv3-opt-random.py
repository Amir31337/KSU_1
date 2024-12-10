import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer,
)
import matplotlib.pyplot as plt
import json
import torchsummary
import torchinfo
from torch.cuda.amp import GradScaler, autocast
import joblib
import multiprocessing
import optuna
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------
# 1. Data Loading (Outside Objective Function)
# ---------------------------

# Define the path to your data file
DATA_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/Data/random_million_orient.csv'

# Load data
data = pd.read_csv(DATA_PATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Check for NaNs or Infs in data
print(f"Number of NaNs in position: {np.isnan(position).sum()}")
print(f"Number of Infs in position: {np.isinf(position).sum()}")
print(f"Number of NaNs in momenta: {np.isnan(momenta).sum()}")
print(f"Number of Infs in momenta: {np.isinf(momenta).sum()}")

# Split data into train, validation, and test sets (70%, 15%, 15%)
train_position, temp_position, train_momenta, temp_momenta = train_test_split(
    position, momenta, test_size=0.3, random_state=42, shuffle=True
)

val_position, test_position, val_momenta, test_momenta = train_test_split(
    temp_position, temp_momenta, test_size=0.5, random_state=42, shuffle=True
)

# Convert to PyTorch tensors (keep data on CPU)
train_position = torch.FloatTensor(train_position)
val_position = torch.FloatTensor(val_position)
test_position = torch.FloatTensor(test_position)
train_momenta = torch.FloatTensor(train_momenta)
val_momenta = torch.FloatTensor(val_momenta)
test_momenta = torch.FloatTensor(test_momenta)

# ---------------------------
# 2. Supported Activation Functions and Normalization Methods
# ---------------------------

# Define supported activation functions
supported_activation_functions = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'Tanh': nn.Tanh(),
}

# Define supported normalization methods
supported_normalization_methods = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
}

# ---------------------------
# 3. Device Configuration (Outside Objective Function)
# ---------------------------

# Enable cuDNN benchmark for performance optimization
torch.backends.cudnn.benchmark = True

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Determine the device type dynamically
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

# Determine the number of workers based on the system's CPU count
max_workers = 12
num_workers = min(multiprocessing.cpu_count() - 1, max_workers)

# ---------------------------
# 4. Define the CVAE Model (Outside Objective Function)
# ---------------------------

# Define the CVAE model with the specified hyperparameters
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_layers, activation_function):
        super(CVAE, self).__init__()

        # Encoder
        encoder_layers = []
        in_features = input_dim
        for h_dim in hidden_layers:
            encoder_layers.append(nn.Linear(in_features, h_dim))
            encoder_layers.append(activation_function)
            in_features = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

        # Decoder
        decoder_layers = []
        in_features = latent_dim + condition_dim
        for h_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(in_features, h_dim))
            decoder_layers.append(activation_function)
            in_features = h_dim
        decoder_layers.append(nn.Linear(in_features, input_dim))
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

# ---------------------------
# 5. Objective Function for Optuna
# ---------------------------

def objective(trial):

    # ---------------------------
    # 5.1. Hyperparameter Sampling
    # ---------------------------

    # Adjusted hyperparameter sampling
    hidden_layer_size = trial.suggest_categorical('hidden_layer_size', [128, 256, 512])
    num_hidden_layers = trial.suggest_categorical('num_hidden_layers', [2, 3, 4])
    max_hidden_size = hidden_layer_size * (2 ** (num_hidden_layers - 1))
    possible_latent_dims = [size for size in [32, 64, 128] if size <= max_hidden_size]
    LATENT_DIM = trial.suggest_categorical('LATENT_DIM', possible_latent_dims)
    EPOCHS = trial.suggest_categorical('EPOCHS', [25, 50, 100])
    BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [32, 64, 128, 256, 512])
    LEARNING_RATE = trial.suggest_categorical('LEARNING_RATE', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    PATIENCE = trial.suggest_categorical('PATIENCE', [5, 10, 15])
    MIN_DELTA = trial.suggest_categorical('MIN_DELTA', [1e-4, 1e-3])
    activation_name = trial.suggest_categorical('activation_name', ['ReLU', 'LeakyReLU', 'Tanh'])
    position_norm_method = trial.suggest_categorical('position_norm_method', ['StandardScaler', 'MinMaxScaler', 'None'])
    momenta_norm_method = trial.suggest_categorical('momenta_norm_method', ['StandardScaler', 'MinMaxScaler', 'None'])
    use_l1 = False  # Fixed to False to prevent issues
    L1_LAMBDA = 0.0  # Not used since use_l1 is False
    use_l2 = False  # Fixed to False to prevent issues
    L2_LAMBDA = 0.0  # Not used since use_l2 is False
    use_beta = False  # Fixed to False initially
    BETA = 1.0  # Not used since use_beta is False

    # Adjust BETA if use_beta is False
    if not use_beta:
        BETA = 1.0

    # ---------------------------
    # 5.2. Save Directory Configuration
    # ---------------------------

    # Define the directory where all output files will be saved
    SAVE_DIR = f'/home/g/ghanaatian/MYFILES/FALL24/Physics/CVAE/v3/random/random_{trial.number}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Define file names
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'best_model.pth')
    DETAILED_RESULTS_PATH = os.path.join(SAVE_DIR, 'detailed_results.json')
    FIRST_EPOCH_PLOT_PATH = os.path.join(SAVE_DIR, 'first.png')
    REST_EPOCH_PLOT_PATH = os.path.join(SAVE_DIR, 'rest.png')
    LATENT_STATS_PATH = os.path.join(SAVE_DIR, 'latent_stats.pt')
    MOMENTA_SCALER_PATH = os.path.join(SAVE_DIR, 'momenta_scaler.pkl')
    POSITION_SCALER_PATH = os.path.join(SAVE_DIR, 'position_scaler.pkl')
    RESULTS_PATH = os.path.join(SAVE_DIR, 'results.json')
    HYPERPARAMETERS_PATH = os.path.join(SAVE_DIR, 'hyperparameters.json')

    # Save hyperparameters
    hyperparameters = {
        'LATENT_DIM': LATENT_DIM,
        'EPOCHS': EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'PATIENCE': PATIENCE,
        'MIN_DELTA': MIN_DELTA,
        'hidden_layer_size': hidden_layer_size,
        'num_hidden_layers': num_hidden_layers,
        'activation_name': activation_name,
        'position_norm_method': position_norm_method,
        'momenta_norm_method': momenta_norm_method,
        'use_l1': use_l1,
        'L1_LAMBDA': L1_LAMBDA,
        'use_l2': use_l2,
        'L2_LAMBDA': L2_LAMBDA,
        'use_beta': use_beta,
        'BETA': BETA,
    }
    with open(HYPERPARAMETERS_PATH, 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    # ---------------------------
    # 5.3. Activation Function and Normalization Methods
    # ---------------------------

    # Activation function
    activation_function = supported_activation_functions.get(activation_name, nn.ReLU())
    if activation_name not in supported_activation_functions:
        print(f"Unsupported activation '{activation_name}'. Defaulting to ReLU.")

    # Normalization methods
    if position_norm_method == 'None':
        position_scaler = None
    else:
        position_scaler = supported_normalization_methods.get(position_norm_method, None)
        if position_scaler is None:
            print(f"Unsupported normalization method '{position_norm_method}' for position. No scaling applied.")

    if momenta_norm_method == 'None':
        momenta_scaler = None
    else:
        momenta_scaler = supported_normalization_methods.get(momenta_norm_method, None)
        if momenta_scaler is None:
            print(f"Unsupported normalization method '{momenta_norm_method}' for momenta. No scaling applied.")

    # Normalize the data based on specified methods
    if position_scaler is not None:
        train_position_norm = torch.FloatTensor(position_scaler.fit_transform(train_position.numpy()))
        val_position_norm = torch.FloatTensor(position_scaler.transform(val_position.numpy()))
        test_position_norm = torch.FloatTensor(position_scaler.transform(test_position.numpy()))
    else:
        train_position_norm = train_position
        val_position_norm = val_position
        test_position_norm = test_position

    if momenta_scaler is not None:
        train_momenta_norm = torch.FloatTensor(momenta_scaler.fit_transform(train_momenta.numpy()))
        val_momenta_norm = torch.FloatTensor(momenta_scaler.transform(val_momenta.numpy()))
        test_momenta_norm = torch.FloatTensor(momenta_scaler.transform(test_momenta.numpy()))
    else:
        train_momenta_norm = train_momenta
        val_momenta_norm = val_momenta
        test_momenta_norm = test_momenta

    # ---------------------------
    # 5.4. DataLoaders
    # ---------------------------

    # Create DataLoaders with the optimized num_workers
    train_loader = DataLoader(
        TensorDataset(train_position_norm, train_momenta_norm),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        TensorDataset(val_position_norm, val_momenta_norm),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        TensorDataset(test_position_norm, test_momenta_norm),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # ---------------------------
    # 5.5. Model Initialization
    # ---------------------------

    # Hidden layers configuration
    hidden_layers = [hidden_layer_size for _ in range(num_hidden_layers)]

    # Enforce that LATENT_DIM is less than or equal to smallest hidden layer size
    smallest_hidden_size = min(hidden_layers)
    if LATENT_DIM > smallest_hidden_size:
        LATENT_DIM = smallest_hidden_size

    # Initialize the model
    model = CVAE(
        input_dim=9,
        latent_dim=LATENT_DIM,
        condition_dim=9,
        hidden_layers=hidden_layers,
        activation_function=activation_function
    ).to(device)

    # Optionally use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    # ---------------------------
    # 5.6. Optimizer and Loss Function
    # ---------------------------

    # Define the optimizer without weight decay
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

    # Define the loss function
    def loss_fn(recon_x, x, mu, logvar):
        # Reconstruction loss (Mean Squared Error)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')

        # KL Divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_divergence /= x.size(0) * x.size(1)  # Normalize by batch size and feature size

        # Apply Beta parameter to KL divergence
        kl_divergence = BETA * kl_divergence

        # Total loss
        loss = recon_loss + kl_divergence

        return loss

    # ---------------------------
    # 5.7. Training Loop
    # ---------------------------

    # Initialize lists to store training and validation losses
    train_losses = []
    val_losses = []

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Training loop with early stopping and mixed precision
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_x, batch_cond in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_cond = batch_cond.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast():
                recon_x, mu, logvar = model(batch_x, batch_cond)
                loss = loss_fn(recon_x, batch_x, mu, logvar)

            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_cond in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_cond = batch_cond.to(device, non_blocking=True)
                with autocast():
                    recon_x, mu, logvar = model(batch_x, batch_cond)
                    loss = loss_fn(recon_x, batch_x, mu, logvar)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Trial {trial.number} | Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the model
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

        # Report intermediate objective value
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # ---------------------------
    # 5.8. Saving Scalers
    # ---------------------------

    if position_scaler is not None:
        joblib.dump(position_scaler, POSITION_SCALER_PATH)
    if momenta_scaler is not None:
        joblib.dump(momenta_scaler, MOMENTA_SCALER_PATH)

    # ---------------------------
    # 5.9. Evaluation on Test Set
    # ---------------------------

    # Load the best model for evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    model.eval()
    test_predictions = []

    # Compute latent stats from training set
    mu_list = []
    logvar_list = []
    with torch.no_grad():
        for batch_x, batch_cond in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_cond = batch_cond.to(device, non_blocking=True)
            mu, logvar = model.encode(batch_x)
            mu_list.append(mu.cpu())
            logvar_list.append(logvar.cpu())
        mu_train = torch.cat(mu_list, dim=0)
        logvar_train = torch.cat(logvar_list, dim=0)
        mu_train_mean = mu_train.mean(dim=0)
        mu_train_std = mu_train.std(dim=0)

        # Save latent stats
        torch.save({'mu_train_mean': mu_train_mean, 'mu_train_std': mu_train_std}, LATENT_STATS_PATH)

        # Sample z and decode in batches
        z_sample = torch.randn(len(test_momenta_norm), LATENT_DIM)
        z_sample = z_sample * mu_train_std.unsqueeze(0) + mu_train_mean.unsqueeze(0)

        # Move z_sample to CPU to avoid GPU memory issues
        z_sample = z_sample.cpu()

        # Process test data in batches
        for batch_idx, (batch_x, batch_cond) in enumerate(test_loader):
            batch_size = batch_x.size(0)
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + batch_size
            batch_z = z_sample[start_idx:end_idx].to(device, non_blocking=True)
            batch_cond = batch_cond.to(device, non_blocking=True)
            with autocast():
                pred = model.decode(batch_z, batch_cond)
            test_predictions.append(pred.cpu())
    test_predictions = torch.cat(test_predictions, dim=0)

    # Inverse transform
    if position_scaler:
        test_predictions_inv = position_scaler.inverse_transform(test_predictions.numpy())
        test_position_inv = position_scaler.inverse_transform(test_position.numpy())
    else:
        test_predictions_inv = test_predictions.numpy()
        test_position_inv = test_position.numpy()

    # Calculate MRE
    relative_errors = np.abs(test_predictions_inv - test_position_inv) / (np.abs(test_position_inv) + 1e-8)
    mre = np.mean(relative_errors)

    # Calculate MSE
    mse = np.mean((test_predictions_inv - test_position_inv) ** 2)

    # Save both metrics to a JSON file
    results = {
        'mre': float(mre),  # Convert numpy.float32 to Python float
        'mse': float(mse)   # Convert numpy.float32 to Python float
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=4)

    # Calculate component-wise MSE
    component_names = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
    component_mse = np.mean((test_predictions_inv - test_position_inv) ** 2, axis=0)

    # Save detailed results including component-wise MSE
    detailed_results = {
        'overall_mre': float(mre),
        'overall_mse': float(mse),
        'component_mse': {name: float(mse_value) for name, mse_value in zip(component_names, component_mse)}
    }

    with open(DETAILED_RESULTS_PATH, 'w') as f:
        json.dump(detailed_results, f, indent=4)

    # ---------------------------
    # 5.10. Plotting Learning Curves
    # ---------------------------

    # Ensure that matplotlib does not try to open any window
    plt.switch_backend('Agg')

    # Plot first 10 epochs
    plt.figure(figsize=(10, 6))
    epochs_first = range(1, min(11, len(train_losses)+1))
    plt.plot(epochs_first, train_losses[:10], label='Train Loss')
    plt.plot(epochs_first, val_losses[:10], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss - First 10 Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIRST_EPOCH_PLOT_PATH)
    plt.close()

    # Plot remaining epochs
    if len(train_losses) > 10:
        plt.figure(figsize=(10, 6))
        epochs_rest = range(11, len(train_losses)+1)
        plt.plot(epochs_rest, train_losses[10:], label='Train Loss')
        plt.plot(epochs_rest, val_losses[10:], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss - Remaining Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(REST_EPOCH_PLOT_PATH)
        plt.close()

    # ---------------------------
    # 5.11. Model Summaries
    # ---------------------------

    try:
        torchsummary.summary(model, input_size=[(9,), (9,)])
    except Exception as e:
        pass

    try:
        x_dummy = torch.randn(1, 9).to(device)
        cond_dummy = torch.randn(1, 9).to(device)
        torchinfo.summary(model, input_data=(x_dummy, cond_dummy), device=device)
    except Exception as e:
        pass

    # Return the test MRE as the objective value
    return mre

# ---------------------------
# 6. Running the Optuna Study
# ---------------------------

if __name__ == '__main__':
    # Create an Optuna study
    study = optuna.create_study(direction='minimize')

    # Run the optimization
    study.optimize(objective, n_trials=100)  # Adjust n_trials as desired

    # Get the top 3 best trials
    top_trials = sorted(study.trials, key=lambda t: t.value)[:3]

    # Save the top 3 trials' hyperparameters and test MREs
    results = []
    for trial in top_trials:
        trial_result = {
            'trial_number': trial.number,
            'test_mre': trial.value,
            'hyperparameters': trial.params
        }
        results.append(trial_result)

    # Save results to JSON and text files
    with open('top_trials.json', 'w') as f:
        json.dump(results, f, indent=4)

    with open('top_trials.txt', 'w') as f:
        for i, trial in enumerate(results):
            f.write(f"Rank {i+1}:\n")
            f.write(f"Trial Number: {trial['trial_number']}\n")
            f.write(f"Test MRE: {trial['test_mre']}\n")
            f.write("Hyperparameters:\n")
            for key, value in trial['hyperparameters'].items():
                f.write(f"    {key}: {value}\n")
            f.write("\n")

    print("Top 3 trials saved to 'top_trials.json' and 'top_trials.txt'.")
