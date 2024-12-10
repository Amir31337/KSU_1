'''
===== Cross-Validation Results =====
Fold 1: Test MSE = 0.7939, Mean Relative Error = 1.6674
Fold 2: Test MSE = 0.7829, Mean Relative Error = 1.0861
Fold 3: Test MSE = 0.7676, Mean Relative Error = 1.1567
Fold 4: Test MSE = 0.7695, Mean Relative Error = 0.9325
Fold 5: Test MSE = 3.6450, Mean Relative Error = 5.5142

Average Test MSE: 1.3518 ± 1.1467
Average Mean Relative Error: 2.0714 ± 1.7390
'''
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="No artists with labels found to put in legend.")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

# Clear GPU memory
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define variables
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
LATENT_DIM = 1920
INPUT_DIM = 9  # Dimension of position
OUTPUT_DIM = 9  # Dimension of momenta
EPOCHS = 182
BATCH_SIZE = 128
LEARNING_RATE = 5.772971044368426e-05
PATIENCE = 5
MIN_DELTA = 5e-3
NUM_FOLDS = 5

# Define AUTOENCODER_LAYERS
AUTOENCODER_LAYERS = [1152, 1152]

# Create directory to save fold results
os.makedirs('fold_results', exist_ok=True)

# Load and preprocess data
data = pd.read_csv(FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Normalize the position using StandardScaler, don't normalize momenta
position_scaler = StandardScaler()
position_normalized = position_scaler.fit_transform(position)
momenta_normalized = momenta

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

# Loss functions for CVAE
def cvae_loss_fn(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence

# Initialize K-Fold cross-validation
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Lists to store overall results
fold_mse_list = []
fold_mre_list = []

for fold, (train_val_indices, test_indices) in enumerate(kf.split(position_normalized), 1):
    print(f'\n===== Fold {fold} =====')
    
    # Split data into training+validation and test sets
    train_val_position = position_normalized[train_val_indices]
    train_val_momenta = momenta_normalized[train_val_indices]
    test_position = position_normalized[test_indices]
    test_momenta = momenta_normalized[test_indices]
    
    # Further split training into training and validation sets
    train_position, val_position, train_momenta, val_momenta = train_test_split(
        train_val_position, train_val_momenta, test_size=0.1765, random_state=42
    )
    # The test_size=0.1765 ensures validation set is approximately 15% of the total data
    
    # Convert to PyTorch tensors and move to device
    train_position_tensor = torch.FloatTensor(train_position).to(device)
    val_position_tensor = torch.FloatTensor(val_position).to(device)
    test_position_tensor = torch.FloatTensor(test_position).to(device)
    train_momenta_tensor = torch.FloatTensor(train_momenta).to(device)
    val_momenta_tensor = torch.FloatTensor(val_momenta).to(device)
    test_momenta_tensor = torch.FloatTensor(test_momenta).to(device)
    
    # Create DataLoader
    train_dataset = TensorDataset(train_position_tensor, train_momenta_tensor)
    val_dataset = TensorDataset(val_position_tensor, val_momenta_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Build the CVAE model
    cvae = CVAE(INPUT_DIM, LATENT_DIM, OUTPUT_DIM, AUTOENCODER_LAYERS).to(device)
    
    # Optimizer
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
            loss = cvae_loss_fn(recon_position, position_batch, mu, logvar)
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
                loss = cvae_loss_fn(recon_position, position_batch, mu, logvar)
                cvae_val_loss_epoch += loss.item()
        cvae_val_loss_epoch /= len(val_loader)
        cvae_val_losses.append(cvae_val_loss_epoch)

        print(f'Epoch {epoch+1}, CVAE loss: {cvae_loss_epoch:.4f}, Val loss: {cvae_val_loss_epoch:.4f}')

        # Early stopping based on validation loss
        if cvae_val_loss_epoch < best_cvae_loss - MIN_DELTA:
            best_cvae_loss = cvae_val_loss_epoch
            patience_counter = 0
            torch.save(cvae.state_dict(), f'fold_results/cvae_weights_fold{fold}.pth')
            print("Saved best CVAE model for this fold.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping CVAE training after {epoch+1} epochs for this fold.')
                break
    
    # Plot and save CVAE losses for the current fold
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, min(11, len(cvae_train_losses)+1)), cvae_train_losses[:10], label='Training Loss')
    plt.plot(range(1, min(11, len(cvae_val_losses)+1)), cvae_val_losses[:10], label='Validation Loss')
    plt.title(f'Fold {fold} CVAE Loss (First 10 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'fold_results/cvae_loss_plot_fold{fold}_first10.png')
    plt.close()
    
    if len(cvae_train_losses) > 10:
        plt.figure(figsize=(6, 4))
        plt.plot(range(11, len(cvae_train_losses)+1), cvae_train_losses[10:], label='Training Loss')
        plt.plot(range(11, len(cvae_val_losses)+1), cvae_val_losses[10:], label='Validation Loss')
        plt.title(f'Fold {fold} CVAE Loss (Epochs 11 Onwards)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'fold_results/cvae_loss_plot_fold{fold}_rest.png')
        plt.close()
    
    # Load the best CVAE model for this fold
    cvae.load_state_dict(torch.load(f'fold_results/cvae_weights_fold{fold}.pth', map_location=device))
    cvae.eval()
    
    # Compute latent distribution parameters and z_train
    with torch.no_grad():
        mu_train, logvar_train = cvae.encode(train_position_tensor)
        z_train = cvae.reparameterize(mu_train, logvar_train)
        latent_z_np = z_train.cpu().numpy()
        latent_z_mean = np.mean(latent_z_np, axis=0)
        latent_z_std = np.std(latent_z_np, axis=0)
    
    # Save latent distribution parameters and z_train using pickle
    with open(f'fold_results/latent_z_mean_fold{fold}.pkl', 'wb') as f:
        pickle.dump(latent_z_mean, f)
    with open(f'fold_results/latent_z_std_fold{fold}.pkl', 'wb') as f:
        pickle.dump(latent_z_std, f)
    with open(f'fold_results/z_train_fold{fold}.pkl', 'wb') as f:
        pickle.dump(latent_z_np, f)
    
    # Testing phase
    print(f"Evaluating on test set for Fold {fold}...")
    # Load latent distribution parameters
    with open(f'fold_results/latent_z_mean_fold{fold}.pkl', 'rb') as f:
        latent_z_mean_loaded = pickle.load(f)
    with open(f'fold_results/latent_z_std_fold{fold}.pkl', 'rb') as f:
        latent_z_std_loaded = pickle.load(f)
    
    # Convert to torch tensors
    latent_z_mean_tensor = torch.tensor(latent_z_mean_loaded).to(device)
    latent_z_std_tensor = torch.tensor(latent_z_std_loaded).to(device)
    
    # Create DataLoader for test set
    test_dataset = TensorDataset(test_position_tensor, test_momenta_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_predicted_positions = []
    all_actual_positions = []
    
    with torch.no_grad():
        for batch_idx, (test_pos_batch, test_mom_batch) in enumerate(test_loader):
            batch_size_current = test_mom_batch.size(0)
            # Sample z from N(mu_train, sigma_train) for each sample in the batch
            z_sample = torch.randn(batch_size_current, LATENT_DIM).to(device) * latent_z_std_tensor + latent_z_mean_tensor
            # Decode to get predicted positions
            predicted_pos_batch = cvae.decode(z_sample, test_mom_batch)
            all_predicted_positions.append(predicted_pos_batch.cpu().numpy())
            all_actual_positions.append(test_pos_batch.cpu().numpy())
    
    # Concatenate all predicted and actual positions
    all_predicted_positions = np.vstack(all_predicted_positions)
    all_actual_positions = np.vstack(all_actual_positions)
    
    # Inverse transform the predicted and actual positions
    all_predicted_positions_inverse = position_scaler.inverse_transform(all_predicted_positions)
    all_actual_positions_inverse = position_scaler.inverse_transform(all_actual_positions)
    
    # Calculate evaluation metrics using the inverse transformed positions
    mse = np.mean((all_predicted_positions_inverse - all_actual_positions_inverse) ** 2)
    
    # Calculate relative error
    epsilon = 1e-8
    relative_errors = np.abs(all_predicted_positions_inverse - all_actual_positions_inverse) / (np.abs(all_actual_positions_inverse) + epsilon)
    mean_relative_error = np.mean(relative_errors)
    
    print(f"Fold {fold} Test MSE: {mse:.4f}")
    print(f"Fold {fold} Mean Relative Error: {mean_relative_error:.4f}")
    
    # Append results to the overall list
    fold_mse_list.append(mse)
    fold_mre_list.append(mean_relative_error)
    
    # Save fold results using pickle
    fold_results = {
        'cvae_train_losses': cvae_train_losses,
        'cvae_val_losses': cvae_val_losses,
        'test_mse': mse,
        'mean_relative_error': mean_relative_error
    }
    with open(f'fold_results/cvae_results_fold{fold}.pkl', 'wb') as f:
        pickle.dump(fold_results, f)
    print(f"Fold {fold} results saved.")

# After all folds, calculate average and standard deviation of MSE and MRE
average_mse = np.mean(fold_mse_list)
std_mse = np.std(fold_mse_list)
average_mre = np.mean(fold_mre_list)
std_mre = np.std(fold_mre_list)

print("\n===== Cross-Validation Results =====")
for fold in range(1, NUM_FOLDS + 1):
    print(f"Fold {fold}: Test MSE = {fold_mse_list[fold-1]:.4f}, Mean Relative Error = {fold_mre_list[fold-1]:.4f}")
print(f"\nAverage Test MSE: {average_mse:.4f} ± {std_mse:.4f}")
print(f"Average Mean Relative Error: {average_mre:.4f} ± {std_mre:.4f}")

# Save overall cross-validation results using pickle
overall_results = {
    'fold_mse': fold_mse_list,
    'fold_mre': fold_mre_list,
    'average_mse': average_mse,
    'std_mse': std_mse,
    'average_mre': average_mre,
    'std_mre': std_mre
}
with open('fold_results/cvae_overall_results.pkl', 'wb') as f:
    pickle.dump(overall_results, f)
print("Overall cross-validation results saved.")

# Optionally, plot average losses across folds (if needed)
# This requires collecting loss curves from all folds

# Plot training and validation loss for first 10 epochs across all folds
plt.figure(figsize=(8, 6))
for fold in range(1, NUM_FOLDS + 1):
    with open(f'fold_results/cvae_results_fold{fold}.pkl', 'rb') as f:
        fold_results = pickle.load(f)
    fold_train_losses = fold_results['cvae_train_losses']
    fold_val_losses = fold_results['cvae_val_losses']
    plt.plot(range(1, min(11, len(fold_train_losses)+1)), fold_train_losses[:10], 
             alpha=0.5, label='Training Loss' if fold ==1 else "")
    plt.plot(range(1, min(11, len(fold_val_losses)+1)), fold_val_losses[:10], 
             alpha=0.5, label='Validation Loss' if fold ==1 else "")
plt.title('CVAE Training and Validation Loss (First 10 Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('fold_results/cvae_loss_plot_first10_all_folds.png')
plt.close()

# Plot training and validation loss for remaining epochs across all folds
plt.figure(figsize=(8, 6))
for fold in range(1, NUM_FOLDS + 1):
    with open(f'fold_results/cvae_results_fold{fold}.pkl', 'rb') as f:
        fold_results = pickle.load(f)
    fold_train_losses = fold_results['cvae_train_losses']
    fold_val_losses = fold_results['cvae_val_losses']
    if len(fold_train_losses) > 10:
        plt.plot(range(11, len(fold_train_losses)+1), fold_train_losses[10:], 
                 alpha=0.5, label='Training Loss' if fold ==1 else "")
        plt.plot(range(11, len(fold_val_losses)+1), fold_val_losses[10:], 
                 alpha=0.5, label='Validation Loss' if fold ==1 else "")
plt.title('CVAE Training and Validation Loss (Epochs 11 Onwards)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('fold_results/cvae_loss_plot_rest_all_folds.png')
plt.close()

print("All plots saved.")
