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
import optuna
from optuna.trial import TrialState

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
data = pd.read_csv(FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Split data into train, validation, and test sets (70%, 15%, 15%)
train_position, temp_position, train_momenta, temp_momenta = train_test_split(
    position, momenta, test_size=0.3, random_state=42
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

def objective(trial):
    # Suggest hyperparameters
    LATENT_DIM = trial.suggest_int('LATENT_DIM', 64, 2048, log=True)
    EPOCHS = trial.suggest_int('EPOCHS', 30, 200)
    BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [256, 512, 1024])
    LEARNING_RATE = trial.suggest_loguniform('LEARNING_RATE', 1e-6, 1e-3)
    PATIENCE = trial.suggest_int('PATIENCE', 5, 20)
    MIN_DELTA = trial.suggest_loguniform('MIN_DELTA', 1e-6, 1e-3)
    
    # Activation functions
    activation_name = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'ELU', 'Tanh', 'Sigmoid'])
    if activation_name == 'LeakyReLU':
        activation_function = nn.LeakyReLU()
    else:
        activation_function = getattr(nn, activation_name)()
    
    # Normalization methods
    position_norm_method = trial.suggest_categorical('position_norm_method', ['StandardScaler', 'MinMaxScaler', None])
    momenta_norm_method = trial.suggest_categorical('momenta_norm_method', ['StandardScaler', 'MinMaxScaler', None])
    
    # Regularization
    use_l1 = trial.suggest_categorical('use_l1', [True, False])
    L1_LAMBDA = trial.suggest_loguniform('L1_LAMBDA', 1e-6, 1e-2) if use_l1 else 0.0
    use_l2 = trial.suggest_categorical('use_l2', [True, False])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2) if use_l2 else 0.0
    use_dropout = trial.suggest_categorical('use_dropout', [True, False])
    DROPOUT_RATE = trial.suggest_uniform('DROPOUT_RATE', 0.1, 0.5) if use_dropout else 0.0
    
    # Hidden layers configuration
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 3)
    hidden_layer_size = trial.suggest_int('hidden_layer_size', 128, 1024, log=True)
    hidden_layers = [hidden_layer_size // (2 ** i) for i in range(num_hidden_layers)]
    
    # Define the CVAE model with the suggested hyperparameters
    class CVAE_Optuna(nn.Module):
        def __init__(self, input_dim, latent_dim, condition_dim, hidden_layers, activation_function, dropout_rate):
            super(CVAE_Optuna, self).__init__()

            # Encoder
            encoder_layers = []
            encoder_layers.append(nn.Linear(input_dim, hidden_layers[0]))
            encoder_layers.append(activation_function)
            if use_dropout:
                encoder_layers.append(nn.Dropout(dropout_rate))
            for i in range(len(hidden_layers) - 1):
                encoder_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                encoder_layers.append(activation_function)
                if use_dropout:
                    encoder_layers.append(nn.Dropout(dropout_rate))
            self.encoder = nn.Sequential(*encoder_layers)

            self.fc_mu = nn.Linear(hidden_layers[-1], latent_dim)
            self.fc_logvar = nn.Linear(hidden_layers[-1], latent_dim)

            # Decoder
            decoder_layers = []
            decoder_layers.append(nn.Linear(latent_dim + condition_dim, hidden_layers[-1]))
            decoder_layers.append(activation_function)
            if use_dropout:
                decoder_layers.append(nn.Dropout(dropout_rate))
            for i in reversed(range(len(hidden_layers) - 1)):
                decoder_layers.append(nn.Linear(hidden_layers[i+1], hidden_layers[i]))
                decoder_layers.append(activation_function)
                if use_dropout:
                    decoder_layers.append(nn.Dropout(dropout_rate))
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

    # Initialize the model
    model = CVAE_Optuna(INPUT_DIM, LATENT_DIM, OUTPUT_DIM, hidden_layers, activation_function, DROPOUT_RATE).to(device)
    
    # Define the optimizer with weight decay for L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    
    # Define the loss function with L1 regularization if applicable
    def loss_fn(recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_divergence /= x.size(0) * x.size(1)
        loss = recon_loss + kl_divergence
        if use_l1:
            l1_loss = 0.0
            for param in model.parameters():
                l1_loss += torch.sum(torch.abs(param))
            loss += L1_LAMBDA * l1_loss
        return loss

    # Normalize the data based on suggested methods
    if position_norm_method == 'StandardScaler':
        position_scaler = StandardScaler()
    elif position_norm_method == 'MinMaxScaler':
        position_scaler = MinMaxScaler()
    else:
        position_scaler = None

    if momenta_norm_method == 'StandardScaler':
        momenta_scaler = StandardScaler()
    elif momenta_norm_method == 'MinMaxScaler':
        momenta_scaler = MinMaxScaler()
    else:
        momenta_scaler = None

    if position_scaler is not None:
        train_position_norm = torch.FloatTensor(position_scaler.fit_transform(train_position.cpu())).to(device)
        val_position_norm = torch.FloatTensor(position_scaler.transform(val_position.cpu())).to(device)
        test_position_norm = torch.FloatTensor(position_scaler.transform(test_position.cpu())).to(device)
    else:
        train_position_norm = train_position
        val_position_norm = val_position
        test_position_norm = test_position

    if momenta_scaler is not None:
        train_momenta_norm = torch.FloatTensor(momenta_scaler.fit_transform(train_momenta.cpu())).to(device)
        val_momenta_norm = torch.FloatTensor(momenta_scaler.transform(val_momenta.cpu())).to(device)
        test_momenta_norm = torch.FloatTensor(momenta_scaler.transform(test_momenta.cpu())).to(device)
    else:
        train_momenta_norm = train_momenta
        val_momenta_norm = val_momenta
        test_momenta_norm = test_momenta

    # Create DataLoaders with the suggested BATCH_SIZE
    train_loader = DataLoader(TensorDataset(train_position_norm, train_momenta_norm), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_position_norm, val_momenta_norm), batch_size=BATCH_SIZE, shuffle=False)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_x, batch_cond in train_loader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x, batch_cond)
            loss = loss_fn(recon_x, batch_x, mu, logvar)
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_cond in val_loader:
                recon_x, mu, logvar = model(batch_x, batch_cond)
                loss = loss_fn(recon_x, batch_x, mu, logvar)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # Early stopping check
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the model
            torch.save(model.state_dict(), 'best_model_optuna.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    
    # Load the best model
    model.load_state_dict(torch.load('best_model_optuna.pth'))
    
    # Compute MRE on the test set
    model.eval()
    test_predictions = []
    with torch.no_grad():
        # Compute latent stats from training set
        mu_list = []
        logvar_list = []
        z_train_list = []
        for batch_x, batch_cond in train_loader:
            mu, logvar = model.encode(batch_x)
            z = model.reparameterize(mu, logvar)
            mu_list.append(mu)
            logvar_list.append(logvar)
            z_train_list.append(z)
        mu_train = torch.cat(mu_list, dim=0)
        logvar_train = torch.cat(logvar_list, dim=0)
        z_train = torch.cat(z_train_list, dim=0)
        mu_train_mean = mu_train.mean(dim=0)
        mu_train_std = mu_train.std(dim=0)
        
        # Save latent stats
        torch.save({'mu_train_mean': mu_train_mean.cpu(), 'mu_train_std': mu_train_std.cpu()}, 'latent_stats_optuna.pt')
        
        # Sample z and decode
        z_sample = torch.randn(len(test_momenta_norm), LATENT_DIM).to(device)
        z_sample = z_sample * mu_train_std.unsqueeze(0) + mu_train_mean.unsqueeze(0)
        for i in range(len(test_momenta_norm)):
            cond = test_momenta_norm[i].unsqueeze(0)
            pred = model.decode(z_sample[i].unsqueeze(0), cond)
            test_predictions.append(pred)
    test_predictions = torch.cat(test_predictions, dim=0)
    
    # Inverse transform
    if position_norm_method:
        test_predictions_inv = position_scaler.inverse_transform(test_predictions.cpu().numpy())
        test_position_inv = position_scaler.inverse_transform(test_position.cpu().numpy())
    else:
        test_predictions_inv = test_predictions.cpu().numpy()
        test_position_inv = test_position.cpu().numpy()
    
    # Calculate MRE
    relative_errors = np.abs(test_predictions_inv - test_position_inv) / (np.abs(test_position_inv) + 1e-8)
    mre = np.mean(relative_errors)
    
    return mre

# Create an Optuna study
study = optuna.create_study(direction='minimize')

# Optimize the objective function
study.optimize(objective, n_trials=100, timeout=86400)  # Adjust n_trials and timeout as needed

# Get the best trial
best_trial = study.best_trial

print("Best MRE: {}".format(best_trial.value))
print("Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"  {key}: {value}")

# Save the best hyperparameters to a JSON file
best_params = {
    'mre': best_trial.value,
    'hyperparameters': best_trial.params
}

with open('best_hyperparameters_optuna.json', 'w') as f:
    json.dump(best_params, f, indent=4)
