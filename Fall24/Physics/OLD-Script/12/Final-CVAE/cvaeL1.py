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
from tqdm import tqdm  # For progress bars

# Clear GPU memory
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
LATENT_DIM = 1024
INPUT_DIM = 9  # Dimension of position
OUTPUT_DIM = 9  # Dimension of momenta
EPOCHS = 182
BATCH_SIZE = 128
LEARNING_RATE = 5e-05
PATIENCE = 5
MIN_DELTA = 5e-3
# L1_LAMBDA will be tuned

# Define AUTOENCODER_LAYERS
AUTOENCODER_LAYERS = [256, 1024]

# Define the range of L1_LAMBDA values to explore
L1_LAMBDA_VALUES = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

# Directory to save models and results
SAVE_DIR = 'cvae_hyperparameter_tuning'
os.makedirs(SAVE_DIR, exist_ok=True)

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
        print(hidden_layers)
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
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_divergence

# Create DataLoader
train_dataset = TensorDataset(train_position_tensor, train_momenta_tensor)
val_dataset = TensorDataset(val_position_tensor, val_momenta_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Function to train and evaluate CVAE for a given L1_LAMBDA
def train_evaluate_cvae(l1_lambda, run_id):
    print(f"\n=== Training CVAE with L1_LAMBDA = {l1_lambda} ===")
    
    # Initialize the CVAE model
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

        for batch_idx, (position, momenta) in enumerate(train_loader):
            cvae_optimizer.zero_grad()
            recon_position, mu, logvar = cvae(position, momenta)
            loss = cvae_loss_fn(recon_position, position, mu, logvar)
            
            # ------------------ L1 Regularization ------------------
            if l1_lambda > 0.0:
                l1_loss = 0.0
                for name, param in cvae.named_parameters():
                    if 'bias' not in name and param.requires_grad:
                        l1_loss += torch.sum(torch.abs(param))
                loss += l1_lambda * l1_loss
            # -------------------------------------------------------

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
                if l1_lambda > 0.0:
                    l1_loss = 0.0
                    for name, param in cvae.named_parameters():
                        if 'bias' not in name and param.requires_grad:
                            l1_loss += torch.sum(torch.abs(param))
                    loss += l1_lambda * l1_loss
                cvae_val_loss_epoch += loss.item()
        cvae_val_loss_epoch /= len(val_loader)
        cvae_val_losses.append(cvae_val_loss_epoch)

        print(f'Epoch {epoch+1}/{EPOCHS}, CVAE Loss: {cvae_loss_epoch:.4f}, Val Loss: {cvae_val_loss_epoch:.4f}')

        # Early stopping based on validation loss
        if cvae_val_loss_epoch < best_cvae_loss - MIN_DELTA:
            best_cvae_loss = cvae_val_loss_epoch
            patience_counter = 0
            model_save_path = os.path.join(SAVE_DIR, f'cvae_weights_L1_{l1_lambda}.pth')
            torch.save(cvae.state_dict(), model_save_path)
            print("Saved best CVAE model for this L1_LAMBDA.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping CVAE training after {epoch+1} epochs for L1_LAMBDA = {l1_lambda}')
                break

    # Load the best model
    cvae.load_state_dict(torch.load(model_save_path))
    cvae.eval()

    # Save training and validation losses
    losses = {
        'train_losses': cvae_train_losses,
        'val_losses': cvae_val_losses
    }
    loss_save_path = os.path.join(SAVE_DIR, f'cvae_losses_L1_{l1_lambda}.npz')
    np.savez(loss_save_path, train_losses=cvae_train_losses, val_losses=cvae_val_losses)

    # Save the best latent distribution parameters
    with torch.no_grad():
        latent_mu, latent_logvar = cvae.encode(train_position_tensor)
        latent_z = cvae.reparameterize(latent_mu, latent_logvar)
        latent_z_np = latent_z.cpu().numpy()
        latent_z_mean = np.mean(latent_z_np, axis=0)
        latent_z_std = np.std(latent_z_np, axis=0)
        # Save the latent distribution parameters
        np.save(os.path.join(SAVE_DIR, f'latent_z_mean_L1_{l1_lambda}.npy'), latent_z_mean)
        np.save(os.path.join(SAVE_DIR, f'latent_z_std_L1_{l1_lambda}.npy'), latent_z_std)

    # Testing phase (No encoder on test data)
    print("Evaluating on test set...")
    latent_z_mean = np.load(os.path.join(SAVE_DIR, f'latent_z_mean_L1_{l1_lambda}.npy'))
    latent_z_std = np.load(os.path.join(SAVE_DIR, f'latent_z_std_L1_{l1_lambda}.npy'))
    
    latent_z_mean = torch.tensor(latent_z_mean).to(device)
    latent_z_std = torch.tensor(latent_z_std).to(device)
    
    with torch.no_grad():
        cvae.eval()
        z_sample = torch.randn(len(test_momenta_tensor), LATENT_DIM).to(device) * latent_z_std + latent_z_mean
        all_predicted_positions = []

        for i in tqdm(range(len(test_momenta_tensor)), desc="Decoding Test Set"):
            momenta = test_momenta_tensor[i].unsqueeze(0)
            z = z_sample[i].unsqueeze(0)
            predicted_position = cvae.decode(z, momenta)
            all_predicted_positions.append(predicted_position)

    # Concatenate predicted positions into a single tensor
    all_predicted_positions = torch.cat(all_predicted_positions, dim=0)
    
    # Inverse transform the predicted and actual positions
    all_predicted_positions_inverse = position_scaler.inverse_transform(all_predicted_positions.cpu().numpy())
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
    print(f"Mean Relative Error (MRE): {mean_relative_error:.4f}")
    
    # Visualize CVAE losses for first 10 epochs
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, min(11, len(cvae_train_losses)+1)), cvae_train_losses[:10], label='Training Loss')
    plt.plot(range(1, min(11, len(cvae_val_losses)+1)), cvae_val_losses[:10], label='Validation Loss')
    plt.title(f'CVAE Loss (First 10 Epochs) - L1_LAMBDA={l1_lambda}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'cvae_loss_plot_first10_L1_{l1_lambda}.png'))
    plt.close()
    
    # Visualize CVAE losses for remaining epochs
    if len(cvae_train_losses) > 10:
        plt.figure(figsize=(6, 4))
        plt.plot(range(11, len(cvae_train_losses)+1), cvae_train_losses[10:], label='Training Loss')
        plt.plot(range(11, len(cvae_val_losses)+1), cvae_val_losses[10:], label='Validation Loss')
        plt.title(f'CVAE Loss (Epochs 11 Onwards) - L1_LAMBDA={l1_lambda}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f'cvae_loss_plot_rest_L1_{l1_lambda}.png'))
        plt.close()
    
    # Save evaluation metrics
    results = {
        'cvae_train_losses': cvae_train_losses,
        'cvae_val_losses': cvae_val_losses,
        'test_mse': mse,
        'test_mae': mae,
        'mean_relative_error': mean_relative_error
    }
    
    torch.save(results, os.path.join(SAVE_DIR, f'cvae_results_L1_{l1_lambda}.pt'))
    print(f"Results saved for L1_LAMBDA = {l1_lambda}.\n")
    
    return {
        'L1_LAMBDA': l1_lambda,
        'Test_MSE': mse,
        'Test_MAE': mae,
        'Mean_Relative_Error': mean_relative_error
    }

# Initialize a list to store all results
all_results = []

# Iterate over all L1_LAMBDA values and train/evaluate the model
for idx, l1_lambda in enumerate(L1_LAMBDA_VALUES):
    run_id = idx + 1
    result = train_evaluate_cvae(l1_lambda, run_id)
    all_results.append(result)

# Convert results to a DataFrame for easy analysis
results_df = pd.DataFrame(all_results)
print("\n=== Hyperparameter Tuning Results ===")
print(results_df)

# Save the summary of all results
results_summary_path = os.path.join(SAVE_DIR, 'hyperparameter_tuning_summary.csv')
results_df.to_csv(results_summary_path, index=False)
print(f"Hyperparameter tuning summary saved to {results_summary_path}.")

# Identify the best L1_LAMBDA based on lowest Mean Relative Error
best_result = results_df.loc[results_df['Mean_Relative_Error'].idxmin()]
best_l1_lambda = best_result['L1_LAMBDA']
print(f"\nBest L1_LAMBDA: {best_l1_lambda} with Mean Relative Error: {best_result['Mean_Relative_Error']:.4f}")

# Optionally, you can load the best model for further use
# best_model_path = os.path.join(SAVE_DIR, f'cvae_weights_L1_{best_l1_lambda}.pth')
# cvae_best = CVAE(INPUT_DIM, LATENT_DIM, OUTPUT_DIM, AUTOENCODER_LAYERS).to(device)
# cvae_best.load_state_dict(torch.load(best_model_path))
# cvae_best.eval()
