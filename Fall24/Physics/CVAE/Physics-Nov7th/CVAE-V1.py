import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
FILEPATH = '/content/drive/MyDrive/PhysicsProject-KSU/cei_traning_orient_1.csv'
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

# Set hyperparameters
LATENT_DIM = 128
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.0012538192581474535
PATIENCE = 100
MIN_DELTA = 1e-5
activation_name = 'LeakyReLU'
position_norm_method = 'MinMaxScaler'
momenta_norm_method = 'StandardScaler'
use_l1 = True
L1_LAMBDA = 0.4
use_l2 = True
L2_LAMBDA = 0.2
num_hidden_layers = 2
hidden_layer_size = 256

activation_function = getattr(nn, activation_name)()

# Normalization methods
if position_norm_method == 'StandardScaler':
    position_scaler = StandardScaler()
elif position_norm_method == 'MinMaxScaler':
    position_scaler = MinMaxScaler()
elif position_norm_method == 'None':
    position_scaler = None

if momenta_norm_method == 'StandardScaler':
    momenta_scaler = StandardScaler()
elif momenta_norm_method == 'MinMaxScaler':
    momenta_scaler = MinMaxScaler()
elif momenta_norm_method == 'None':
    momenta_scaler = None

# Normalize the data
if position_scaler is not None:
    train_position_norm = torch.FloatTensor(position_scaler.fit_transform(train_position))
    val_position_norm = torch.FloatTensor(position_scaler.transform(val_position))
    test_position_norm = torch.FloatTensor(position_scaler.transform(test_position))
else:
    train_position_norm = torch.FloatTensor(train_position)
    val_position_norm = torch.FloatTensor(val_position)
    test_position_norm = torch.FloatTensor(test_position)

if momenta_scaler is not None:
    train_momenta_norm = torch.FloatTensor(momenta_scaler.fit_transform(train_momenta))
    val_momenta_norm = torch.FloatTensor(momenta_scaler.transform(val_momenta))
    test_momenta_norm = torch.FloatTensor(momenta_scaler.transform(test_momenta))
else:
    train_momenta_norm = torch.FloatTensor(train_momenta)
    val_momenta_norm = torch.FloatTensor(val_momenta)
    test_momenta_norm = torch.FloatTensor(test_momenta)

# Hidden layers configuration
hidden_layers = [hidden_layer_size * (2 ** i) for i in range(num_hidden_layers)]

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(train_position_norm, train_momenta_norm)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(val_position_norm, val_momenta_norm)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = TensorDataset(test_position_norm, test_momenta_norm)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

# Instantiate the model
model = CVAE(INPUT_DIM, OUTPUT_DIM, LATENT_DIM, hidden_layers, activation_function).to(device)

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
    total_loss = recon_loss + kl_div + L1_LAMBDA * l1_reg + L2_LAMBDA * l2_reg
    return total_loss

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
        optimizer.step()
        train_loss += loss.item()
        # Update running sums for mu
        batch_size = data_x.size(0)
        mu_sum += mu.sum(dim=0)
        mu_squared_sum += (mu ** 2).sum(dim=0)
        total_samples += batch_size
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
        print(f'Early stopping at epoch {epoch+1}')
        break
    print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')

# Compute latent distribution parameters from training data
latent_mu_mean = mu_sum / total_samples
latent_mu_std = torch.sqrt(mu_squared_sum / total_samples - latent_mu_mean ** 2)

# Evaluation on training data
model.eval()
train_mse_total = 0
train_mre_total = 0
with torch.no_grad():
    for batch_idx, (data_x, data_c) in enumerate(train_loader):
        data_x = data_x.to(device)
        data_c = data_c.to(device)
        x_recon, _, _ = model(data_x, data_c)
        mse = nn.functional.mse_loss(x_recon, data_x, reduction='sum').item()
        # Add epsilon to avoid division by zero
        epsilon = 1e-8
        mre = (torch.abs(x_recon - data_x) / (torch.abs(data_x) + epsilon)).sum().item()
        train_mse_total += mse
        train_mre_total += mre
train_mse = train_mse_total / len(train_loader.dataset)
train_mre = train_mre_total / (len(train_loader.dataset) * INPUT_DIM)

# Evaluation on validation data
val_mse_total = 0
val_mre_total = 0
with torch.no_grad():
    for batch_idx, (data_x, data_c) in enumerate(val_loader):
        data_x = data_x.to(device)
        data_c = data_c.to(device)
        x_recon, _, _ = model(data_x, data_c)
        mse = nn.functional.mse_loss(x_recon, data_x, reduction='sum').item()
        epsilon = 1e-8
        mre = (torch.abs(x_recon - data_x) / (torch.abs(data_x) + epsilon)).sum().item()
        val_mse_total += mse
        val_mre_total += mre
val_mse = val_mse_total / len(val_loader.dataset)
val_mre = val_mre_total / (len(val_loader.dataset) * INPUT_DIM)

# Evaluation on test data without data leakage
test_mse_total = 0
test_mre_total = 0
test_recon_positions = []
with torch.no_grad():
    for batch_idx, (data_x, data_c) in enumerate(test_loader):
        data_c = data_c.to(device)
        data_x = data_x.to(device)
        # Sample z from N(latent_mu_mean, latent_mu_std)
        batch_size = data_c.size(0)
        z = torch.normal(latent_mu_mean.expand(batch_size, -1), latent_mu_std.expand(batch_size, -1)).to(device)
        x_recon = model.decode(z, data_c)
        mse = nn.functional.mse_loss(x_recon, data_x, reduction='sum').item()
        epsilon = 1e-8
        mre = (torch.abs(x_recon - data_x) / (torch.abs(data_x) + epsilon)).sum().item()
        test_mse_total += mse
        test_mre_total += mre
        test_recon_positions.append(x_recon.cpu())

test_mse = test_mse_total / len(test_loader.dataset)
test_mre = test_mre_total / (len(test_loader.dataset) * INPUT_DIM)

# Concatenate all reconstructed positions
test_recon_positions = torch.cat(test_recon_positions, dim=0)

# Print the errors
print(f'\nTrain MSE: {train_mse:.6f}, Train MRE: {train_mre:.6f}')
print(f'Validation MSE: {val_mse:.6f}, Validation MRE: {val_mre:.6f}')
print(f'Test MSE: {test_mse:.6f}, Test MRE: {test_mre:.6f}')

# Plot learning curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Print 5 random samples from the test set
print("\nSampled Test Data:")
num_samples = 5
indices = np.random.choice(len(test_position_norm), num_samples, replace=False)

for idx in indices:
    pos_actual = test_position_norm[idx].numpy()
    mom = test_momenta_norm[idx].numpy()
    pos_recon = test_recon_positions[idx].numpy()
    print(f"Sample {idx}:")
    print(f"Position (Actual): {pos_actual}")
    print(f"Momenta: {mom}")
    print(f"Position (Reconstructed): {pos_recon}\n")




# Evaluation on test data without data leakage
test_mse_total = 0
test_mre_total = 0
test_recon_positions = []
with torch.no_grad():
    for batch_idx, (data_x, data_c) in enumerate(test_loader):
        data_c = data_c.to(device)
        data_x = data_x.to(device)
        # Sample z from N(latent_mu_mean, latent_mu_std)
        batch_size = data_c.size(0)
        z = torch.normal(latent_mu_mean.expand(batch_size, -1), latent_mu_std.expand(batch_size, -1)).to(device)
        x_recon = model.decode(z, data_c)
        mse = nn.functional.mse_loss(x_recon, data_x, reduction='sum').item()
        epsilon = 1e-8
        mre = (torch.abs(x_recon - data_x) / (torch.abs(data_x) + epsilon)).sum().item()
        test_mse_total += mse
        test_mre_total += mre
        test_recon_positions.append(x_recon.cpu())

test_mse = test_mse_total / len(test_loader.dataset)
test_mre = test_mre_total / (len(test_loader.dataset) * INPUT_DIM)

# Concatenate all reconstructed positions
test_recon_positions = torch.cat(test_recon_positions, dim=0)

# Invert normalization to get original scale for positions and reconstructed positions
if position_scaler is not None:
    test_position_actual = position_scaler.inverse_transform(test_position_norm.cpu().numpy())
    test_recon_position_original = position_scaler.inverse_transform(test_recon_positions.numpy())
else:
    test_position_actual = test_position_norm.cpu().numpy()
    test_recon_position_original = test_recon_positions.numpy()

# Invert normalization for momenta
if momenta_scaler is not None:
    test_momenta_actual = momenta_scaler.inverse_transform(test_momenta_norm.cpu().numpy())
else:
    test_momenta_actual = test_momenta_norm.cpu().numpy()

# Create a DataFrame
import pandas as pd

# Define the number of dimensions
INPUT_DIM = position.shape[1]  # Should be 9
OUTPUT_DIM = momenta.shape[1]  # Should be 9

# Create column names for actual positions, momenta, and reconstructed positions
position_cols = [f'Position_Actual_{i+1}' for i in range(INPUT_DIM)]
momenta_cols = [f'Momenta_{i+1}' for i in range(OUTPUT_DIM)]
recon_cols = [f'Position_Reconstructed_{i+1}' for i in range(INPUT_DIM)]

# Initialize a dictionary to hold the data
data_dict = {}

# Populate the dictionary with actual positions
for i, col in enumerate(position_cols):
    data_dict[col] = test_position_actual[:, i]

# Populate the dictionary with momenta
for i, col in enumerate(momenta_cols):
    data_dict[col] = test_momenta_actual[:, i]

# Populate the dictionary with reconstructed positions
for i, col in enumerate(recon_cols):
    data_dict[col] = test_recon_position_original[:, i]

# Create the DataFrame
test_results_df = pd.DataFrame(data_dict)

# Specify the file path where you want to save the CSV
OUTPUT_CSV_PATH = '/content/drive/MyDrive/PhysicsProject-KSU/Colab/CVAE/test_results.csv'

# Save the DataFrame to a CSV file
test_results_df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"Test results saved to {OUTPUT_CSV_PATH}")

# Print the errors
print(f'\nTrain MSE: {train_mse:.6f}, Train MRE: {train_mre:.6f}')
print(f'Validation MSE: {val_mse:.6f}, Validation MRE: {val_mre:.6f}')
print(f'Test MSE: {test_mse:.6f}, Test MRE: {test_mre:.6f}')


# Plot learning curves
plt.figure(figsize=(100, 50))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()