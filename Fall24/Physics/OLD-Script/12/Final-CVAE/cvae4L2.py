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
import copy
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters (other than L2_LAMBDA)
LATENT_DIM = 1024
EPOCHS = 65
BATCH_SIZE = 512
LEARNING_RATE = 2.1210335031751337e-05
PATIENCE = 11
MIN_DELTA = 1.5894218493676975e-05

# List of L2_LAMBDA values to try
L2_LAMBDA_VALUES = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

hidden_layer_size = 512
num_hidden_layers = 1
hidden_layers = [hidden_layer_size // (2 ** i) for i in range(num_hidden_layers)]  # [512]

activation_name = 'Tanh'
if activation_name == 'LeakyReLU':
    activation_function = nn.LeakyReLU()
else:
    activation_function = getattr(nn, activation_name)()

position_norm_method = 'MinMaxScaler'
momenta_norm_method = None

# Load data
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
data = pd.read_csv(FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Normalize the data
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
train_position = torch.FloatTensor(train_position)
val_position = torch.FloatTensor(val_position)
test_position = torch.FloatTensor(test_position)
train_momenta = torch.FloatTensor(train_momenta)
val_momenta = torch.FloatTensor(val_momenta)
test_momenta = torch.FloatTensor(test_momenta)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move data to device
train_position = train_position.to(device)
val_position = val_position.to(device)
test_position = test_position.to(device)
train_momenta = train_momenta.to(device)
val_momenta = val_momenta.to(device)
test_momenta = test_momenta.to(device)

INPUT_DIM = position.shape[1]  # Should be 9
OUTPUT_DIM = momenta.shape[1]  # Should be 9

# Define the CVAE model
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_layers, activation_function):
        super(CVAE, self).__init__()

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_layers[0]))
        encoder_layers.append(activation_function)
        for i in range(len(hidden_layers) - 1):
            encoder_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            encoder_layers.append(activation_function)
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(hidden_layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[-1], latent_dim)

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim + condition_dim, hidden_layers[-1]))
        decoder_layers.append(activation_function)
        for i in reversed(range(len(hidden_layers) - 1)):
            decoder_layers.append(nn.Linear(hidden_layers[i+1], hidden_layers[i]))
            decoder_layers.append(activation_function)
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

# Loss function
def cvae_loss_fn(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_divergence /= x.size(0) * x.size(1)  # Normalize by batch size and input dimensions
    total_loss = recon_loss + kl_divergence
    if torch.isnan(total_loss):
        print("NaN detected in loss computation.")
    return total_loss

# Function to train and evaluate the model for a given L2_LAMBDA
def train_evaluate_model(L2_LAMBDA):
    print(f"\nTraining with L2_LAMBDA = {L2_LAMBDA}")

    # Initialize the model
    model = CVAE(INPUT_DIM, LATENT_DIM, OUTPUT_DIM, hidden_layers, activation_function).to(device)

    # Initialize the optimizer with weight_decay=0 since L2 is manually added
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

    # Create DataLoaders
    train_dataset = TensorDataset(train_position, train_momenta)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(val_position, val_momenta)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Training loop with early stopping and L2 regularization
    cvae_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    model_saved = False  # Flag to check if model was saved

    for epoch in range(EPOCHS):
        model.train()
        train_loss_epoch = 0

        for batch_idx, (position_batch, momenta_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_position, mu, logvar = model(position_batch, momenta_batch)
            loss = cvae_loss_fn(recon_position, position_batch, mu, logvar)

            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx+1}. Skipping this batch.")
                continue  # Skip this batch

            # Compute L2 penalty (only for weights, excluding biases)
            l2_penalty = L2_LAMBDA * sum(param.pow(2).sum() for name, param in model.named_parameters() if 'weight' in name)

            # Total loss
            total_loss = loss + l2_penalty

            total_loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_epoch += total_loss.item()

        # If no batches were processed, break the loop
        if train_loss_epoch == 0:
            print(f"No valid batches in epoch {epoch+1}. Stopping training.")
            break

        train_loss_epoch /= len(train_loader)
        cvae_losses.append(train_loss_epoch)

        # Validation
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for position_batch, momenta_batch in val_loader:
                recon_position, mu, logvar = model(position_batch, momenta_batch)
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
            torch.save(model.state_dict(), 'best_cvae_temp.pth')
            model_saved = True
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Check if a model was saved
    if not model_saved:
        print("No valid model was saved during training.")
        return None

    # Load the best model
    model.load_state_dict(torch.load('best_cvae_temp.pth'))

    # Compute and save Mean and Std of latent variables on training set
    model.eval()
    with torch.no_grad():
        mu_list = []
        logvar_list = []
        z_train_list = []
        for position_batch, momenta_batch in train_loader:
            mu, logvar = model.encode(position_batch)
            z = model.reparameterize(mu, logvar)
            mu_list.append(mu)
            logvar_list.append(logvar)
            z_train_list.append(z)

        mu_train = torch.cat(mu_list, dim=0)
        logvar_train = torch.cat(logvar_list, dim=0)
        z_train = torch.cat(z_train_list, dim=0)

        # Compute mean and std of latent variables
        mu_train_mean = mu_train.mean(dim=0)
        mu_train_std = mu_train.std(dim=0)

        # Save mu_train_mean and mu_train_std
        torch.save({'mu_train_mean': mu_train_mean.cpu(), 'mu_train_std': mu_train_std.cpu()}, 'latent_stats_temp.pt')

    # For test set, sample z from training distribution and decode
    test_predictions = []
    model.eval()
    with torch.no_grad():
        # Load latent stats
        latent_stats = torch.load('latent_stats_temp.pt')
        mu_train_mean = latent_stats['mu_train_mean'].to(device)
        mu_train_std = latent_stats['mu_train_std'].to(device)

        # Sample z from training distribution
        z_sample = torch.randn(len(test_momenta), LATENT_DIM).to(device)
        z_sample = z_sample * mu_train_std.unsqueeze(0) + mu_train_mean.unsqueeze(0)
        for i in range(len(test_momenta)):
            momenta_sample = test_momenta[i].unsqueeze(0)
            predicted_position = model.decode(z_sample[i].unsqueeze(0), momenta_sample)
            test_predictions.append(predicted_position)

    test_predictions = torch.cat(test_predictions, dim=0)

    # Inverse transform the predicted and actual positions
    if position_scaler is not None:
        test_predictions_inverse = position_scaler.inverse_transform(test_predictions.cpu().numpy())
        test_position_inverse = position_scaler.inverse_transform(test_position.cpu().numpy())
    else:
        test_predictions_inverse = test_predictions.cpu().numpy()
        test_position_inverse = test_position.cpu().numpy()

    # Calculate MSE and MRE on test set using original values
    mse = np.mean((test_predictions_inverse - test_position_inverse) ** 2)
    relative_errors = np.abs(test_predictions_inverse - test_position_inverse) / (np.abs(test_position_inverse) + 1e-8)
    mre = np.mean(relative_errors)

    # Clean up temporary files
    if os.path.exists('best_cvae_temp.pth'):
        os.remove('best_cvae_temp.pth')
    if os.path.exists('latent_stats_temp.pt'):
        os.remove('latent_stats_temp.pt')

    print(f"Test MSE: {mse}")
    print(f"Test MRE: {mre}")

    return {
        'L2_LAMBDA': L2_LAMBDA,
        'mse': float(mse),
        'mre': float(mre),
        'hyperparameters': {
            'LATENT_DIM': LATENT_DIM,
            'EPOCHS': EPOCHS,
            'BATCH_SIZE': BATCH_SIZE,
            'LEARNING_RATE': LEARNING_RATE,
            'PATIENCE': PATIENCE,
            'MIN_DELTA': MIN_DELTA,
            'hidden_layer_size': hidden_layer_size,
            'num_hidden_layers': num_hidden_layers,
            'activation': activation_name,
            'position_norm_method': position_norm_method,
            'momenta_norm_method': momenta_norm_method
        }
    }

def plot_loss_curves(cvae_losses, val_losses, L2_LAMBDA, save_prefix):
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cvae_losses) + 1), cvae_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss Curves for L2_LAMBDA = {L2_LAMBDA}')
    plt.savefig(f'{save_prefix}_loss_curves_L2_{L2_LAMBDA}.png')
    plt.close()

def main():
    results = []

    for L2_LAMBDA in L2_LAMBDA_VALUES:
        result = train_evaluate_model(L2_LAMBDA)
        if result is not None:
            results.append(result)

    if not results:
        print("No models were successfully trained.")
        return

    # Find the best L2_LAMBDA with the lowest MRE
    best_result = min(results, key=lambda x: x['mre'])
    best_L2_LAMBDA = best_result['L2_LAMBDA']
    best_mre = best_result['mre']
    best_mse = best_result['mse']

    print(f"\nBest L2_LAMBDA: {best_L2_LAMBDA} with Test MRE: {best_mre}")

    # Save all results to a JSON file
    with open('l2_lambda_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Additionally, save the best hyperparameters and results
    best_results = {
        'best_L2_LAMBDA': best_L2_LAMBDA,
        'best_mre': best_mre,
        'best_mse': best_mse,
        'hyperparameters': best_result['hyperparameters']
    }

    with open('best_l2_lambda_result.json', 'w') as f:
        json.dump(best_results, f, indent=4)

    print("All results have been saved to 'l2_lambda_results.json'.")
    print("Best result has been saved to 'best_l2_lambda_result.json'.")

if __name__ == "__main__":
    main()
