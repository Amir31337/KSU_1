import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import random
import os
import itertools

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters and Configuration
DATA_FILE = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
POSITION_COLUMNS = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
MOMENTA_COLUMNS = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']
POSITION_DIM = len(POSITION_COLUMNS)
MOMENTA_DIM = len(MOMENTA_COLUMNS)
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
NUM_FOLDS = 5

# Hyperparameter ranges
LATENT_DIMS = [16, 32, 64, 128, 256, 512, 1024, 2048]
ENCODER_HIDDEN_DIMS = [16, 32, 64, 128, 256, 512, 1024, 2048]
DECODER_HIDDEN_DIMS = [16, 32, 64, 128, 256, 512, 1024, 2048]
COUPLING_HIDDEN_DIMS = [16, 32, 64, 128, 256, 512, 1024, 2048]

# Random Seed for reproducibility
RANDOM_SEED = 24
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load and preprocess the data
data = pd.read_csv(DATA_FILE)
position = data[POSITION_COLUMNS].values
momenta = data[MOMENTA_COLUMNS].values

# Normalize the data
scaler_pos = StandardScaler()
scaler_mom = StandardScaler()
position_normalized = scaler_pos.fit_transform(position)
momenta_normalized = scaler_mom.fit_transform(momenta)

# Define the dataset
class CoulombExplosionDataset(Dataset):
    def __init__(self, positions, momenta, original_positions):
        self.positions = torch.FloatTensor(positions)
        self.momenta = torch.FloatTensor(momenta)
        self.original_positions = torch.FloatTensor(original_positions)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.momenta[idx], self.original_positions[idx]

# Create the full dataset
full_dataset = CoulombExplosionDataset(position_normalized, momenta_normalized, position)

# Custom Conditional Invertible block
class CouplingLayer(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.net = nn.Sequential(
            nn.Linear(condition_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * input_dim)
        )

    def forward(self, x, condition):
        net_input = torch.cat([x, condition], dim=1)
        params = self.net(net_input)
        scale = torch.sigmoid(params[:, :self.input_dim])
        shift = params[:, self.input_dim:]
        x_modified = scale * x + shift
        return x_modified

# Encoder and Decoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, condition_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z, condition):
        z_cond = torch.cat([z, condition], dim=1)
        return self.net(z_cond)

# Define the CINN model
class CINN(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim, encoder_hidden_dim, decoder_hidden_dim, coupling_hidden_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, encoder_hidden_dim)
        self.decoder = Decoder(latent_dim, input_dim, condition_dim, decoder_hidden_dim)
        self.coupling_layer = CouplingLayer(latent_dim, condition_dim, coupling_hidden_dim)

    def forward(self, x, condition):
        z = self.encoder(x)
        transformed = self.coupling_layer(z, condition)
        reconstructed = self.decoder(transformed, condition)
        return reconstructed, transformed

    def inverse(self, latent, condition):
        return self.decoder(latent, condition)

def train_and_evaluate(latent_dim, encoder_hidden_dim, decoder_hidden_dim, coupling_hidden_dim):
    # Initialize K-fold cross-validation
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Initialize lists to store results
    mse_scores = []

    # K-fold cross-validation loop
    for fold, (train_indices, test_indices) in enumerate(kfold.split(full_dataset)):
        print(f"Fold {fold + 1}/{NUM_FOLDS}")
        
        # Create train and test datasets for the current fold
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        # Initialize data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model
        model = CINN(input_dim=POSITION_DIM, condition_dim=MOMENTA_DIM, latent_dim=latent_dim,
                     encoder_hidden_dim=encoder_hidden_dim, decoder_hidden_dim=decoder_hidden_dim,
                     coupling_hidden_dim=coupling_hidden_dim).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        
        # Training loop
        latent_representations = []
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            for positions, momenta, _ in train_loader:
                # Move data to the device
                positions, momenta = positions.to(device), momenta.to(device)
                
                optimizer.zero_grad()
                reconstructed, latent_z = model(positions, momenta)
                loss = criterion(reconstructed, positions)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Save latent representation of the training data
            if epoch == NUM_EPOCHS - 1:
                for positions, momenta, _ in train_loader:
                    positions, momenta = positions.to(device), momenta.to(device)
                    _, latent_z = model(positions, momenta)
                    latent_representations.append(latent_z.cpu().detach().numpy())
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")
        
        # Save latent representations for the current fold
        latent_representations = np.concatenate(latent_representations, axis=0)
        
        # Testing phase
        model.eval()
        all_true_positions = []
        all_predicted_positions = []
        
        with torch.no_grad():
            latent_distribution = latent_representations
            
            for i, (positions, momenta, original_positions) in enumerate(test_loader):
                positions, momenta = positions.to(device), momenta.to(device)
                
                # Sample latent variable from the saved distribution efficiently
                sampled_latent_np = np.array(random.choices(latent_distribution, k=len(positions)))
                sampled_latent = torch.from_numpy(sampled_latent_np).float().to(device)
                
                # Predict positions using the decoder and test momenta as condition
                reconstructed_positions = model.inverse(sampled_latent, momenta)
                
                predicted_positions_original = scaler_pos.inverse_transform(reconstructed_positions.cpu().numpy())
                
                all_true_positions.append(original_positions.numpy())
                all_predicted_positions.append(predicted_positions_original)
        
        # Concatenate all batches for the current fold
        true_positions = np.concatenate(all_true_positions, axis=0)
        predicted_positions = np.concatenate(all_predicted_positions, axis=0)
        
        # Calculate MSE on original scale for the current fold
        mse_original = mean_squared_error(true_positions, predicted_positions)
        mse_scores.append(mse_original)
        print(f"Mean Squared Error (original scale) for Fold {fold + 1}: {mse_original:.4f}")

    # Calculate average MSE across all folds
    avg_mse = np.mean(mse_scores)
    print(f"\nAverage Mean Squared Error (original scale) across all folds: {avg_mse:.4f}")
    
    return avg_mse

# Hyperparameter tuning
best_mse = float('inf')
best_params = None

for latent_dim, encoder_hidden_dim, decoder_hidden_dim, coupling_hidden_dim in itertools.product(
    LATENT_DIMS, ENCODER_HIDDEN_DIMS, DECODER_HIDDEN_DIMS, COUPLING_HIDDEN_DIMS):
    
    print(f"\nTesting parameters: LATENT_DIM={latent_dim}, ENCODER_HIDDEN_DIM={encoder_hidden_dim}, "
          f"DECODER_HIDDEN_DIM={decoder_hidden_dim}, COUPLING_HIDDEN_DIM={coupling_hidden_dim}")
    
    avg_mse = train_and_evaluate(latent_dim, encoder_hidden_dim, decoder_hidden_dim, coupling_hidden_dim)
    
    if avg_mse < best_mse:
        best_mse = avg_mse
        best_params = (latent_dim, encoder_hidden_dim, decoder_hidden_dim, coupling_hidden_dim)

print("\nHyperparameter tuning completed.")
print(f"Best parameters: LATENT_DIM={best_params[0]}, ENCODER_HIDDEN_DIM={best_params[1]}, "
      f"DECODER_HIDDEN_DIM={best_params[2]}, COUPLING_HIDDEN_DIM={best_params[3]}")
print(f"Lowest Average Mean Squared Error (original scale) across all folds: {best_mse:.4f}")