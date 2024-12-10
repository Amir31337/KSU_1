import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings("ignore")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
FILEPATH = 'sim_million_orient.csv'
data = pd.read_csv(FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Print GPU name
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Split data into train, validation, and test sets (70%, 15%, 15%)
train_position, temp_position, train_momenta, temp_momenta = train_test_split(
    position, momenta, test_size=0.3, random_state=42, shuffle=True
)

val_position, test_position, val_momenta, test_momenta = train_test_split(
    temp_position, temp_momenta, test_size=0.5, random_state=42, shuffle=True
)

INPUT_DIM = position.shape[1]  # 9
OUTPUT_DIM = momenta.shape[1]  # 9

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

# Define the loss function with L1 and L2 regularization
def loss_function(x_recon, x, mu, logvar, model, use_l1, use_l2, L1_LAMBDA, L2_LAMBDA):
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # L1 and L2 regularization
    l1_reg = torch.tensor(0., requires_grad=True).to(device)
    l2_reg = torch.tensor(0., requires_grad=True).to(device)
    for param in model.parameters():
        if use_l1:
            l1_reg = l1_reg + torch.norm(param, 1)
        if use_l2:
            l2_reg = l2_reg + torch.norm(param, 2)
    total_loss = recon_loss + kl_div
    if use_l1:
        total_loss += L1_LAMBDA * l1_reg
    if use_l2:
        total_loss += L2_LAMBDA * l2_reg
    return total_loss

# Define the objective function for Optuna
def objective(trial: Trial):
    # Hyperparameter sampling
    LATENT_DIM = trial.suggest_categorical('LATENT_DIM', [128, 256, 512, 1024])
    EPOCHS = trial.suggest_categorical('EPOCHS', [100, 200, 300])
    BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [128, 256, 512, 1024])
    LEARNING_RATE = trial.suggest_loguniform('LEARNING_RATE', 1e-6, 1e-2)
    PATIENCE = trial.suggest_categorical('PATIENCE', [50, 100])
    MIN_DELTA = trial.suggest_categorical('MIN_DELTA', [1e-5, 1e-4, 1e-3, 1e-2])
    activation_name = trial.suggest_categorical('activation_name', ['Sigmoid','Tanh','ReLU', 'LeakyReLU','ELU','Softmax'])
    position_norm_method = trial.suggest_categorical('position_norm_method', ['StandardScaler','MinMaxScaler', 'RobustScaler'])
    momenta_norm_method = trial.suggest_categorical('momenta_norm_method', ['StandardScaler','MinMaxScaler', 'RobustScaler'])
    use_l1 = trial.suggest_categorical('use_l1', [True, False])
    if use_l1:
        L1_LAMBDA = trial.suggest_categorical('L1_LAMBDA', [0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5])
    else:
        L1_LAMBDA = 0.0
    use_l2 = trial.suggest_categorical('use_l2', [True, False])
    if use_l2:
        L2_LAMBDA = trial.suggest_categorical('L2_LAMBDA', [0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5])
    else:
        L2_LAMBDA = 0.0
    num_hidden_layers = trial.suggest_categorical('num_hidden_layers', [1,2,3,4])
    hidden_layer_size = trial.suggest_categorical('hidden_layer_size', [128,256,512,1024])

    # Restrict maximum hidden layer size and latent dimension to prevent instability
    if LATENT_DIM > 512 or hidden_layer_size > 512:
        # Penalize large dimensions by returning a high MRE
        print(f"Penalizing trial due to large LATENT_DIM ({LATENT_DIM}) or hidden_layer_size ({hidden_layer_size}).")
        return float('inf')

    # Activation function handling
    if activation_name == 'Softmax':
        activation_function = nn.Softmax(dim=1)
    else:
        activation_class = getattr(nn, activation_name)
        activation_function = activation_class()

    # Normalization methods
    if position_norm_method == 'StandardScaler':
        position_scaler = StandardScaler()
    elif position_norm_method == 'MinMaxScaler':
        position_scaler = MinMaxScaler()
    elif position_norm_method == 'RobustScaler':
        position_scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported position_norm_method: {position_norm_method}")

    if momenta_norm_method == 'StandardScaler':
        momenta_scaler = StandardScaler()
    elif momenta_norm_method == 'MinMaxScaler':
        momenta_scaler = MinMaxScaler()
    elif momenta_norm_method == 'RobustScaler':
        momenta_scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported momenta_norm_method: {momenta_norm_method}")

    # Normalize the data
    try:
        train_position_norm = torch.FloatTensor(position_scaler.fit_transform(train_position))
        val_position_norm = torch.FloatTensor(position_scaler.transform(val_position))
        test_position_norm = torch.FloatTensor(position_scaler.transform(test_position))

        train_momenta_norm = torch.FloatTensor(momenta_scaler.fit_transform(train_momenta))
        val_momenta_norm = torch.FloatTensor(momenta_scaler.transform(val_momenta))
        test_momenta_norm = torch.FloatTensor(momenta_scaler.transform(test_momenta))
    except Exception as e:
        # If normalization fails, return a high MRE
        print(f"Normalization failed: {e}")
        return float('inf')

    # Hidden layers configuration
    hidden_layers = [hidden_layer_size * (2 ** i) for i in range(num_hidden_layers)]

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(train_position_norm, train_momenta_norm)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(val_position_norm, val_momenta_norm)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = TensorDataset(test_position_norm, test_momenta_norm)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Instantiate the model
    try:
        model = CVAE(INPUT_DIM, OUTPUT_DIM, LATENT_DIM, hidden_layers, activation_function).to(device)
    except Exception as e:
        print(f"Model instantiation failed: {e}")
        return float('inf')

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Gradient clipping to prevent exploding gradients
    GRAD_CLIP = 5.0

    # Training loop with early stopping
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Variables to compute mean and std of mu
    mu_sum = torch.zeros(LATENT_DIM).to(device)
    mu_squared_sum = torch.zeros(LATENT_DIM).to(device)
    total_samples = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, (data_x, data_c) in enumerate(train_loader):
            data_x = data_x.to(device)
            data_c = data_c.to(device)
            optimizer.zero_grad()
            try:
                x_recon, mu, logvar = model(data_x, data_c)
            except Exception as e:
                print(f"Model forward pass failed: {e}")
                return float('inf')
            loss = loss_function(x_recon, data_x, mu, logvar, model, use_l1, use_l2, L1_LAMBDA, L2_LAMBDA)
            loss.backward()
            # Apply gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()
            # Update running sums for mu
            batch_size_current = data_x.size(0)
            mu_sum += mu.sum(dim=0)
            mu_squared_sum += (mu ** 2).sum(dim=0)
            total_samples += batch_size_current
        train_losses.append(train_loss / len(train_loader.dataset))
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data_x, data_c) in enumerate(val_loader):
                data_x = data_x.to(device)
                data_c = data_c.to(device)
                try:
                    x_recon, mu, logvar = model(data_x, data_c)
                except Exception as e:
                    print(f"Model forward pass during validation failed: {e}")
                    return float('inf')
                loss = loss_function(x_recon, data_x, mu, logvar, model, use_l1, use_l2, L1_LAMBDA, L2_LAMBDA)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader.dataset))
        # Early stopping
        if val_losses[-1] + MIN_DELTA < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve == PATIENCE:
            break

    # Compute latent distribution parameters from training data
    try:
        latent_mu_mean = mu_sum / total_samples
        variance = mu_squared_sum / total_samples - latent_mu_mean ** 2
        # Clamp variance to ensure non-negative and finite values
        variance_clamped = torch.clamp(variance, min=1e-6, max=1e6)
        latent_mu_std = torch.sqrt(variance_clamped)
    except Exception as e:
        print(f"Variance computation failed: {e}")
        return float('inf')

    # Optional: Check for NaNs or Infs in mu_sum and mu_squared_sum
    if torch.isnan(mu_sum).any() or torch.isinf(mu_sum).any():
        print("mu_sum contains NaN or Inf values.")
        return float('inf')
    if torch.isnan(mu_squared_sum).any() or torch.isinf(mu_squared_sum).any():
        print("mu_squared_sum contains NaN or Inf values.")
        return float('inf')

    # Optional: Check for NaNs or Infs after clamping and sqrt
    if torch.isnan(latent_mu_std).any() or torch.isinf(latent_mu_std).any():
        print("latent_mu_std contains NaN or Inf values after clamping.")
        return float('inf')

    # Evaluation on test data without data leakage
    model.eval()
    test_mre_total = 0
    with torch.no_grad():
        for batch_idx, (data_x, data_c) in enumerate(test_loader):
            data_c = data_c.to(device)
            data_x = data_x.to(device)
            # Sample z from N(latent_mu_mean, latent_mu_std)
            try:
                z = torch.normal(latent_mu_mean.expand(data_c.size(0), -1), 
                                 latent_mu_std.expand(data_c.size(0), -1)).to(device)
            except Exception as e:
                print(f"Sampling z failed: {e}")
                return float('inf')
            try:
                x_recon = model.decode(z, data_c)
            except Exception as e:
                print(f"Model decode failed: {e}")
                return float('inf')
            # Add epsilon to avoid division by zero
            epsilon = 1e-8
            mre = (torch.abs(x_recon - data_x) / (torch.abs(data_x) + epsilon)).sum().item()
            test_mre_total += mre

    test_mre = test_mre_total / (len(test_loader.dataset) * INPUT_DIM)

    return test_mre



# Create an Optuna study
study = optuna.create_study(direction='minimize', sampler=TPESampler())

# Optimize the study
study.optimize(objective, n_trials=100)  # You can adjust n_trials as needed

# Print the best hyperparameters
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print(f"  Test MRE: {trial.value}")
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Optionally, save the study results
# optuna.logging.enable_default_handler()
# optuna.logging.disable_default_handler()

# To visualize the optimization history
try:
    import optuna.visualization as vis
    vis.plot_optimization_history(study)
    plt.show()
except:
    pass