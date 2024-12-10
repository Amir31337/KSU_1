import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import warnings

# Clear GPU memory
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define variables
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
LATENT_DIM = 2048
INPUT_DIM = 9  # Dimension of position
OUTPUT_DIM = 9  # Dimension of momenta
EPOCHS = 182
BATCH_SIZE = 128
LEARNING_RATE = 5e-05
PATIENCE = 5
MIN_DELTA = 5e-3
DROPOUT_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Define AUTOENCODER_LAYERS
AUTOENCODER_LAYERS = [258, 1024]

# Load and preprocess data
data = pd.read_csv(FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Normalize the position using StandardScaler, don't normalize momenta
position_scaler = StandardScaler()
position_normalized = position_scaler.fit_transform(position)
momenta_normalized = momenta

# Split data into train, validation, and test sets
train_val_position, test_position, train_val_momenta, test_momenta = train_test_split(
    position_normalized, momenta_normalized, test_size=0.15, random_state=42
)
train_position, val_position, train_momenta, val_momenta = train_test_split(
    train_val_position, train_val_momenta, test_size=0.1765, random_state=42
)
# The second split ensures the validation set is 15% of the total data

# Convert to PyTorch tensors and move to device
train_position = torch.FloatTensor(train_position).to(device)
val_position = torch.FloatTensor(val_position).to(device)
test_position = torch.FloatTensor(test_position).to(device)
train_momenta = torch.FloatTensor(train_momenta).to(device)
val_momenta = torch.FloatTensor(val_momenta).to(device)
test_momenta = torch.FloatTensor(test_momenta).to(device)

# Define the CVAE model with Dropout
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_layers, dropout_p=0.5):
        super(CVAE, self).__init__()
        # Encoder: Encodes positions X into latent mean and variance
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),  # Added Dropout
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)   # Added Dropout
        )
        self.fc_mu = nn.Linear(hidden_layers[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[1], latent_dim)
        
        # Decoder: Decodes latent Z conditioned on momenta Y
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),  # Added Dropout
            nn.Linear(hidden_layers[1], hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),  # Added Dropout
            nn.Linear(hidden_layers[0], input_dim)
        )

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

# Loss functions for CVAE
def cvae_loss_fn(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_divergence

# Function to train and evaluate CVAE for a given dropout rate
def train_and_evaluate(dropout_p):
    print(f"\n=== Training CVAE with Dropout Rate: {dropout_p} ===")
    
    # Initialize the CVAE model
    cvae = CVAE(INPUT_DIM, LATENT_DIM, OUTPUT_DIM, AUTOENCODER_LAYERS, dropout_p=dropout_p).to(device)
    
    # Optimizer
    cvae_optimizer = optim.Adam(cvae.parameters(), lr=LEARNING_RATE)
    
    # Create DataLoader
    train_dataset = TensorDataset(train_position, train_momenta)
    val_dataset = TensorDataset(val_position, val_momenta)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Training loop for CVAE
    cvae_train_losses = []
    cvae_val_losses = []
    best_cvae_loss = float('inf')
    patience_counter = 0
    model_path = None  # Initialize model_path

    for epoch in range(EPOCHS):
        cvae.train()
        cvae_loss_epoch = 0

        for batch_idx, (position, momenta) in enumerate(train_loader):
            cvae_optimizer.zero_grad()
            recon_position, mu, logvar = cvae(position, momenta)
            loss = cvae_loss_fn(recon_position, position, mu, logvar)
            
            # Check for nan losses
            if torch.isnan(loss):
                warnings.warn(f"NaN loss encountered at epoch {epoch+1}, batch {batch_idx+1}. Stopping training for dropout {dropout_p}.")
                return None  # Skip this dropout rate
            
            loss.backward()
            cvae_optimizer.step()
            cvae_loss_epoch += loss.item()

        cvae_loss_epoch /= len(train_loader)
        cvae_train_losses.append(cvae_loss_epoch)

        # Validation phase
        cvae.eval()
        cvae_val_loss_epoch = 0
        with torch.no_grad():
            for batch_idx, (position, momenta) in enumerate(val_loader):
                recon_position, mu, logvar = cvae(position, momenta)
                loss = cvae_loss_fn(recon_position, position, mu, logvar)
                
                # Check for nan losses
                if torch.isnan(loss):
                    warnings.warn(f"NaN validation loss encountered at epoch {epoch+1}, batch {batch_idx+1}. Stopping training for dropout {dropout_p}.")
                    return None  # Skip this dropout rate
                
                cvae_val_loss_epoch += loss.item()
        cvae_val_loss_epoch /= len(val_loader)
        cvae_val_losses.append(cvae_val_loss_epoch)

        print(f'Epoch {epoch+1}, CVAE loss: {cvae_loss_epoch:.4f}, Val loss: {cvae_val_loss_epoch:.4f}')

        # Early stopping based on validation loss
        if cvae_val_loss_epoch < best_cvae_loss - MIN_DELTA:
            best_cvae_loss = cvae_val_loss_epoch
            patience_counter = 0
            model_path = f'cvae_weights_dropout{dropout_p}.pth'
            torch.save(cvae.state_dict(), model_path)
            print("Saved best CVAE model.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping CVAE training after {epoch+1} epochs')
                break

    if model_path is None:
        warnings.warn(f"No valid model was saved for dropout rate {dropout_p}. Skipping evaluation.")
        return None  # Skip evaluation if no model was saved

    # Load the best model
    try:
        cvae.load_state_dict(torch.load(model_path))
    except Exception as e:
        warnings.warn(f"Failed to load the best model for dropout {dropout_p}: {e}")
        return None  # Skip evaluation if loading fails

    cvae.eval()

    # Save the training and validation losses
    np.save(f'cvae_train_losses_dropout{dropout_p}.npy', np.array(cvae_train_losses))
    np.save(f'cvae_val_losses_dropout{dropout_p}.npy', np.array(cvae_val_losses))

    # Save loss plots
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(cvae_train_losses)+1), cvae_train_losses, label='Training Loss')
    plt.plot(range(1, len(cvae_val_losses)+1), cvae_val_losses, label='Validation Loss')
    plt.title(f'CVAE Loss (Dropout {dropout_p})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'cvae_loss_plot_dropout{dropout_p}.png')
    plt.close()

    # Save the best latent distribution parameters
    with torch.no_grad():
        latent_mu, latent_logvar = cvae.encode(train_position)
        latent_z = cvae.reparameterize(latent_mu, latent_logvar)
        latent_z_np = latent_z.cpu().numpy()
        latent_z_mean = np.mean(latent_z_np, axis=0)
        latent_z_std = np.std(latent_z_np, axis=0)
        # Save the latent distribution parameters
        np.save(f'latent_z_mean_dropout{dropout_p}.npy', latent_z_mean)
        np.save(f'latent_z_std_dropout{dropout_p}.npy', latent_z_std)

    # Testing phase (No encoder on test data)
    print("Evaluating on test set...")
    latent_z_mean = np.load(f'latent_z_mean_dropout{dropout_p}.npy')
    latent_z_std = np.load(f'latent_z_std_dropout{dropout_p}.npy')
    
    latent_z_mean = torch.tensor(latent_z_mean).to(device)
    latent_z_std = torch.tensor(latent_z_std).to(device)
    
    with torch.no_grad():
        cvae.eval()
        z_sample = torch.randn(len(test_momenta), LATENT_DIM).to(device) * latent_z_std + latent_z_mean
        all_predicted_positions = []

        # To speed up, use batching instead of iterating one by one
        test_dataset = TensorDataset(z_sample, test_momenta)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        for batch_idx, (z, momenta) in enumerate(test_loader):
            predicted_position = cvae.decode(z, momenta)
            all_predicted_positions.append(predicted_position)
    
    # Concatenate predicted positions into a single tensor
    all_predicted_positions = torch.cat(all_predicted_positions, dim=0)
    
    # Inverse transform the predicted and actual positions
    all_predicted_positions_inverse = position_scaler.inverse_transform(all_predicted_positions.cpu().numpy())
    test_position_inverse = position_scaler.inverse_transform(test_position.cpu().numpy())
    
    # Calculate evaluation metrics using the inverse transformed positions
    mse = np.mean((all_predicted_positions_inverse - test_position_inverse) ** 2)
    mae = np.mean(np.abs(all_predicted_positions_inverse - test_position_inverse))
    
    # Calculate relative error
    epsilon = 1e-8
    relative_errors = np.abs(all_predicted_positions_inverse - test_position_inverse) / (np.abs(test_position_inverse) + epsilon)
    mean_relative_error = np.mean(relative_errors)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Mean Relative Error: {mean_relative_error:.4f}")
    
    # Save the test metrics with native Python types
    test_metrics = {
        'dropout_p': float(dropout_p),
        'test_mse': float(mse),
        'test_mae': float(mae),
        'mean_relative_error': float(mean_relative_error)
    }
    torch.save(test_metrics, f'cvae_test_metrics_dropout{dropout_p}.pt')

    return test_metrics

# Initialize a list to store all results
all_results = []

# Loop over each dropout rate and train/evaluate the CVAE
for dropout_p in DROPOUT_RATES:
    metrics = train_and_evaluate(dropout_p)
    if metrics is not None:
        all_results.append(metrics)
    else:
        print(f"Skipping dropout rate {dropout_p} due to training issues.")

# After all dropout rates have been evaluated, find the best one
if all_results:
    best_result = min(all_results, key=lambda x: x['mean_relative_error'])
    best_dropout = best_result['dropout_p']
    best_mre = best_result['mean_relative_error']

    print("\n=== Hyperparameter Tuning Results ===")
    for res in all_results:
        print(f"Dropout Rate: {res['dropout_p']}, Test MRE: {res['mean_relative_error']:.4f}")
    
    print(f"\nBest Dropout Rate: {best_dropout} with Test MRE: {best_mre:.4f}")

    # Save all results to a single file with native Python types
    with open('cvae_dropout_tuning_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("All results saved to 'cvae_dropout_tuning_results.json'.")
else:
    print("No successful training runs were completed. Please adjust the dropout rates or check the model configuration.")
