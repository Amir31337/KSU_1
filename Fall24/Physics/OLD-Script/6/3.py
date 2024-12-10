'''######################
Mean Squared Error: 1.4693
#####################'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from FrEIA import framework as fr
from FrEIA import modules as fm

# Set random seed for reproducibility
random_seed = 42

# Data file path
data_file = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'

# Dimensions of position and momenta
POSITION_DIM = 9
MOMENTA_DIM = 9

# Model hyperparameters
hidden_dim = 99
num_layers = 5

# Training hyperparameters
batch_size = 32
learning_rate = 0.0045
num_epochs = 50

# Test set size
test_size = 0.2

# Number of samples for testing
num_test_samples = 2000

# Model save path
model_save_path = 'cinn_model.pth'

# Set random seeds
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CINN(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, num_layers):
        super(CINN, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        
        # Define the subnet for coupling layers
        def subnet_fc(c_in, c_out):
            return nn.Sequential(
                nn.Linear(c_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, c_out)
            )
        
        # Build the invertible network
        self.cinn = fr.SequenceINN(input_dim)
        for _ in range(num_layers):
            self.cinn.append(fm.GLOWCouplingBlock, subnet_constructor=subnet_fc, clamp=2.0)
            self.cinn.append(fm.PermuteRandom)
        
        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, c, rev=False):
        c_encoded = self.condition_encoder(c)
        return self.cinn(x, c_encoded, rev=rev)

def train_cinn(model, train_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (positions, momenta) in enumerate(train_loader):
            positions, momenta = positions.to(device), momenta.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            z, log_jac_det = model(positions, momenta)
            
            # Compute loss
            loss = torch.mean(0.5 * torch.sum(z**2, dim=1) - log_jac_det)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

def test_cinn(model, test_momenta, num_samples=2000):
    model.eval()
    with torch.no_grad():
        # Generate latent samples
        z = torch.randn(num_samples, POSITION_DIM).to(device)
        
        # Use the model to generate positions from momenta
        predicted_positions, _ = model(z, test_momenta, rev=True)
    
    return predicted_positions.cpu().numpy()

# Load and prepare data
data = pd.read_csv(data_file)
positions = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Split data into train and test sets
train_positions, test_positions, train_momenta, test_momenta = train_test_split(
    positions, momenta, test_size=test_size, random_state=random_seed)

# Convert to PyTorch tensors
train_positions = torch.FloatTensor(train_positions)
train_momenta = torch.FloatTensor(train_momenta)
test_momenta = torch.FloatTensor(test_momenta)

# Create DataLoader for training data
train_dataset = TensorDataset(train_positions, train_momenta)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = CINN(POSITION_DIM, MOMENTA_DIM, hidden_dim, num_layers).to(device)

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_cinn(model, train_loader, optimizer, num_epochs)

# Generate predictions using only momenta
test_momenta_tensor = torch.FloatTensor(test_momenta).to(device)
predicted_positions = test_cinn(model, test_momenta_tensor, num_samples=num_test_samples)

# Compute mean squared error
mse = np.mean((predicted_positions - test_positions)**2)
print(f"Mean Squared Error: {mse:.4f}")

# Optional: Save the model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")