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
if torch.cuda.is_available():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load data
FILEPATH = 'random_million_orient.csv'
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
    try:
        # Hyperparameter sampling
        LATENT_DIM = trial.suggest_categorical('LATENT_DIM', [128, 256, 512, 1024])
        EPOCHS = trial.suggest_categorical('EPOCHS', [100, 200, 300])
        BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [128, 256, 512, 1024])
        LEARNING_RATE = trial.suggest_loguniform('LEARNING_RATE', 1e-6, 1e-2)
        PATIENCE = trial.suggest_categorical('PATIENCE', [50, 100])
        MIN_DELTA = trial.suggest_categorical('MIN_DELTA', [1e-5, 1e-4, 1e-3, 1e-2])
        activation_name = trial.suggest_categorical('activation_name', ['Sigmoid','Tanh','ReLU', 'LeakyReLU','ELU'])  # Removed 'Softmax'
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

        # Activation function handling
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
        train_position_norm = torch.FloatTensor(position_scaler.fit_transform(train_position))
        val_position_norm = torch.FloatTensor(position_scaler.transform(val_position))
        test_position_norm = torch.FloatTensor(position_scaler.transform(test_position))

        train_momenta_norm = torch.FloatTensor(momenta_scaler.fit_transform(train_momenta))
        val_momenta_norm = torch.FloatTensor(momenta_scaler.transform(val_momenta))
        test_momenta_norm = torch.FloatTensor(momenta_scaler.transform(test_momenta))

        # Hidden layers configuration
        # Changed to use consistent hidden layer sizes to prevent exponential growth
        hidden_layers = [hidden_layer_size for _ in range(num_hidden_layers)]

        # Create TensorDatasets and DataLoaders
        train_dataset = TensorDataset(train_position_norm, train_momenta_norm)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = TensorDataset(val_position_norm, val_momenta_norm)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        test_dataset = TensorDataset(test_position_norm, test_momenta_norm)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Instantiate the model
        model = CVAE(INPUT_DIM, OUTPUT_DIM, LATENT_DIM, hidden_layers, activation_function).to(device)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
                x_recon, mu, logvar = model(data_x, data_c)
                loss = loss_function(x_recon, data_x, mu, logvar, model, use_l1, use_l2, L1_LAMBDA, L2_LAMBDA)
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    x_recon, mu, logvar = model(data_x, data_c)
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
        latent_mu_mean = mu_sum / total_samples
        variance = mu_squared_sum / total_samples - latent_mu_mean ** 2
        # Clamp variance to ensure non-negative values
        variance_clamped = torch.clamp(variance, min=1e-6, max=1e6)
        latent_mu_std = torch.sqrt(variance_clamped)

        # Optional: Check for NaNs or Infs
        if torch.isnan(latent_mu_std).any() or torch.isinf(latent_mu_std).any():
            # Instead of raising an error, return a large MRE to penalize this trial
            return 1e6

        # Evaluation on test data without data leakage
        model.eval()
        test_mre_total = 0
        with torch.no_grad():
            for batch_idx, (data_x, data_c) in enumerate(test_loader):
                data_c = data_c.to(device)
                data_x = data_x.to(device)
                # Sample z from N(latent_mu_mean, latent_mu_std)
                batch_size_current = data_c.size(0)
                z = torch.normal(latent_mu_mean.expand(batch_size_current, -1), latent_mu_std.expand(batch_size_current, -1)).to(device)
                x_recon = model.decode(z, data_c)
                # Add epsilon to avoid division by zero
                epsilon = 1e-8
                mre = (torch.abs(x_recon - data_x) / (torch.abs(data_x) + epsilon)).sum().item()
                test_mre_total += mre

        test_mre = test_mre_total / (len(test_loader.dataset) * INPUT_DIM)

        return test_mre

    except Exception as e:
        # In case of any unexpected errors, return a large MRE to penalize the trial
        return 1e6

# Create an Optuna study
study = optuna.create_study(direction='minimize', sampler=TPESampler())

# Optimize the study
study.optimize(objective, n_trials=50, timeout=None)  # You can adjust n_trials as needed

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
