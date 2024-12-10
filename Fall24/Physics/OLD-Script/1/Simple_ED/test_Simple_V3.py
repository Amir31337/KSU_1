import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np

# Parameters
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
data_path = "Physics/1M.csv"
position_columns = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
momentum_columns = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']
batch_size = 2048
input_dim = 9
output_dim = 9
hidden_dim = 512
latent_dim = 128
dropout_rate = 0.03686336695055423

print(f"Using {device}")

# Load and preprocess data
data = pd.read_csv(data_path)
positions = data[position_columns].values
momenta = data[momentum_columns].values

# Normalize data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_normalized = scaler_X.fit_transform(momenta)
y_normalized = scaler_y.fit_transform(positions)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_normalized)
y_tensor = torch.FloatTensor(y_normalized)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
data_loader = DataLoader(dataset, batch_size=batch_size)

# Define model architecture (same as in the training script)
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout_rate):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.decoder(z)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, dropout_rate):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim, dropout_rate)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Load the trained model
model = Autoencoder(input_dim, hidden_dim, latent_dim, output_dim, dropout_rate).to(device)
model.load_state_dict(torch.load('best_model.pth', weights_only=True , map_location=device))
model.eval()

# Predict and evaluate
y_true = []
y_pred = []

with torch.no_grad():
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        y_true.extend(target.cpu().numpy())
        y_pred.extend(output.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Inverse transform the predictions and true values
y_true = scaler_y.inverse_transform(y_true)
y_pred = scaler_y.inverse_transform(y_pred)

# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print(f"Test MSE: {mse:.4f}")
print(f"Test R-squared: {r2:.4f}")
print(f"Test MAE: {mae:.4f}")

# Print 5 random samples
random_indices = np.random.choice(len(y_true), 5, replace=False)
for idx in random_indices:
    print(f"\nSample {idx}:")
    print(f"Real:      {' '.join([f'{val:.3f}' for val in y_true[idx]])}")
    print(f"Predicted: {' '.join([f'{val:.3f}' for val in y_pred[idx]])}")