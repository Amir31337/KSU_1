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
import copy

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Clear GPU memory
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define variables
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
LATENT_DIM = 2048
INPUT_DIM = 9  # Dimension of position
OUTPUT_DIM = 9  # Dimension of momenta
EPOCHS = 182
BATCH_SIZE = 1024
LEARNING_RATE = 5e-05
PATIENCE = 5
MIN_DELTA = 5e-3
AUTOENCODER_LAYERS = [258, 1024]
L2_LAMBDA_LIST = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]  # Define a range of L2_LAMBDA values to test

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
train_position_tensor = torch.FloatTensor(train_position).to(device)
val_position_tensor = torch.FloatTensor(val_position).to(device)
test_position_tensor = torch.FloatTensor(test_position).to(device)
train_momenta_tensor = torch.FloatTensor(train_momenta).to(device)
val_momenta_tensor = torch.FloatTensor(val_momenta).to(device)
test_momenta_tensor = torch.FloatTensor(test_momenta).to(device)

# Define the CVAE model
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_layers):
        super(CVAE, self).__init__()
        # Encoder: Encodes positions X into latent mean and variance
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_layers[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[1], latent_dim)
        
        # Decoder: Decodes latent Z conditioned on momenta Y
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], hidden_layers[0]),
            nn.ReLU(),
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

# Loss functions for CVAE with L2 regularization
def cvae_loss_fn(recon_x, x, mu, logvar, model, lambda_l2):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # L2 Regularization
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.sum(param.pow(2))
    
    return recon_loss + kl_divergence + lambda_l2 * l2_reg

# Create DataLoader
train_dataset = TensorDataset(train_position_tensor, train_momenta_tensor)
val_dataset = TensorDataset(val_position_tensor, val_momenta_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize a dictionary to store results for each L2_LAMBDA
results_dict = {}

# Loop over each L2_LAMBDA value
for L2_LAMBDA in L2_LAMBDA_LIST:
    print(f"\nTraining CVAE with L2_LAMBDA = {L2_LAMBDA}")

    # Initialize the CVAE model
    cvae = CVAE(INPUT_DIM, LATENT_DIM, OUTPUT_DIM, AUTOENCODER_LAYERS).to(device)
    
    # Define the optimizer (no weight_decay since L2 is manually added)
    cvae_optimizer = optim.Adam(cvae.parameters(), lr=LEARNING_RATE)
    
    # Training loop for CVAE
    cvae_train_losses = []
    cvae_val_losses = []
    best_cvae_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        cvae.train()
        cvae_loss_epoch = 0

        for batch_idx, (position_batch, momenta_batch) in enumerate(train_loader):
            cvae_optimizer.zero_grad()
            recon_position, mu, logvar = cvae(position_batch, momenta_batch)
            loss = cvae_loss_fn(recon_position, position_batch, mu, logvar, cvae, L2_LAMBDA)
            loss.backward()
            cvae_optimizer.step()
            cvae_loss_epoch += loss.item()

        cvae_loss_epoch /= len(train_loader)
        cvae_train_losses.append(cvae_loss_epoch)

        # Validation phase
        cvae.eval()
        cvae_val_loss_epoch = 0
        with torch.no_grad():
            for batch_idx, (position_batch, momenta_batch) in enumerate(val_loader):
                recon_position, mu, logvar = cvae(position_batch, momenta_batch)
                loss = cvae_loss_fn(recon_position, position_batch, mu, logvar, cvae, L2_LAMBDA)
                cvae_val_loss_epoch += loss.item()
        cvae_val_loss_epoch /= len(val_loader)
        cvae_val_losses.append(cvae_val_loss_epoch)

        print(f'Epoch {epoch+1}, CVAE Loss: {cvae_loss_epoch:.4f}, Val Loss: {cvae_val_loss_epoch:.4f}')

        # Early stopping based on validation loss
        if cvae_val_loss_epoch < best_cvae_loss - MIN_DELTA:
            best_cvae_loss = cvae_val_loss_epoch
            patience_counter = 0
            model_save_path = f'cvae_weights_L2_{L2_LAMBDA}.pth'
            torch.save(cvae.state_dict(), model_save_path)
            print(f"Saved best CVAE model for L2_LAMBDA = {L2_LAMBDA}.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping CVAE training after {epoch+1} epochs for L2_LAMBDA = {L2_LAMBDA}')
                break

    # Load the best model
    cvae.load_state_dict(torch.load(model_save_path))
    cvae.eval()

    # Save the best latent distribution parameters
    with torch.no_grad():
        latent_mu, latent_logvar = cvae.encode(train_position_tensor)
        latent_z = cvae.reparameterize(latent_mu, latent_logvar)
        latent_z_np = latent_z.cpu().numpy()
        latent_z_mean = np.mean(latent_z_np, axis=0)
        latent_z_std = np.std(latent_z_np, axis=0)
        # Save the latent distribution parameters
        latent_mean_save_path = f'latent_z_mean_L2_{L2_LAMBDA}.npy'
        latent_std_save_path = f'latent_z_std_L2_{L2_LAMBDA}.npy'
        np.save(latent_mean_save_path, latent_z_mean)
        np.save(latent_std_save_path, latent_z_std)

    # Testing phase (No encoder on test data)
    print("Evaluating on test set...")
    latent_z_mean = np.load(latent_mean_save_path)
    latent_z_std = np.load(latent_std_save_path)
    
    latent_z_mean = torch.tensor(latent_z_mean).to(device)
    latent_z_std = torch.tensor(latent_z_std).to(device)
    
    with torch.no_grad():
        cvae.eval()
        z_sample = torch.randn(len(test_momenta_tensor), LATENT_DIM).to(device) * latent_z_std + latent_z_mean
        all_predicted_positions = []

        # To optimize memory and speed, process in batches
        test_loader = DataLoader(TensorDataset(test_position_tensor, test_momenta_tensor), batch_size=BATCH_SIZE, shuffle=False)
        for batch_idx, (position_batch, momenta_batch) in enumerate(test_loader):
            batch_size_current = momenta_batch.size(0)
            z_batch = torch.randn(batch_size_current, LATENT_DIM).to(device) * latent_z_std + latent_z_mean
            predicted_position_batch = cvae.decode(z_batch, momenta_batch)
            all_predicted_positions.append(predicted_position_batch.cpu())

    # Concatenate predicted positions into a single tensor
    all_predicted_positions = torch.cat(all_predicted_positions, dim=0)
    
    # Inverse transform the predicted and actual positions
    all_predicted_positions_inverse = position_scaler.inverse_transform(all_predicted_positions.numpy())
    test_position_inverse = position_scaler.inverse_transform(test_position_tensor.cpu().numpy())
    
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
    
    # Visualize CVAE losses for first 10 epochs
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, min(11, len(cvae_train_losses)+1)), cvae_train_losses[:10], label='Training Loss')
    plt.plot(range(1, min(11, len(cvae_val_losses)+1)), cvae_val_losses[:10], label='Validation Loss')
    plt.title(f'CVAE Loss (First 10 Epochs) L2_LAMBDA={L2_LAMBDA}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'cvae_loss_plot_first10_L2_{L2_LAMBDA}.png')
    plt.close()
    
    # Visualize CVAE losses for remaining epochs
    if len(cvae_train_losses) > 10:
        plt.figure(figsize=(6, 4))
        plt.plot(range(11, len(cvae_train_losses)+1), cvae_train_losses[10:], label='Training Loss')
        plt.plot(range(11, len(cvae_val_losses)+1), cvae_val_losses[10:], label='Validation Loss')
        plt.title(f'CVAE Loss (Epochs 11 Onwards) L2_LAMBDA={L2_LAMBDA}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'cvae_loss_plot_rest_L2_{L2_LAMBDA}.png')
        plt.close()
    
    # Save results for this L2_LAMBDA
    results = {
        'cvae_train_losses': cvae_train_losses,
        'cvae_val_losses': cvae_val_losses,
        'test_mse': mse,
        'test_mae': mae,
        'mean_relative_error': mean_relative_error
    }
    
    results_save_path = f'cvae_results_L2_{L2_LAMBDA}.pt'
    torch.save(results, results_save_path)
    print(f"Results saved for L2_LAMBDA = {L2_LAMBDA}.\n")
    
    # Store the results in the dictionary
    results_dict[L2_LAMBDA] = results

# After all L2_LAMBDA values have been tested, identify the best one
best_L2_LAMBDA = None
lowest_mre = float('inf')

for L2_LAMBDA, res in results_dict.items():
    if res['mean_relative_error'] < lowest_mre:
        lowest_mre = res['mean_relative_error']
        best_L2_LAMBDA = L2_LAMBDA

print(f"\nBest L2_LAMBDA: {best_L2_LAMBDA} with Test MRE: {lowest_mre:.4f}")

# Optionally, save all results to a single file
all_results_save_path = 'all_cvae_results.pt'
torch.save(results_dict, all_results_save_path)
print(f"All results saved to {all_results_save_path}.")

# Plot Mean Relative Error vs L2_LAMBDA
plt.figure(figsize=(8, 6))
L2_values = list(results_dict.keys())
MRE_values = [results_dict[L2]['mean_relative_error'] for L2 in L2_values]
plt.plot(L2_values, MRE_values, marker='o')
plt.xscale('log')
plt.xlabel('L2_LAMBDA')
plt.ylabel('Mean Relative Error (MRE)')
plt.title('MRE vs L2_LAMBDA')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('MRE_vs_L2_LAMBDA.png')
plt.close()
print("MRE vs L2_LAMBDA plot saved as 'MRE_vs_L2_LAMBDA.png'.")
