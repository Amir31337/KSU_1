'''######################
Average Test MSE: 1.4727
#####################'''

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ************************** HYPERPARAMETERS **************************

# File path
DATA_FILE_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'

# Dimensions
POSITION_DIM = 9  # Dimension of position vectors (for Carbon, Oxygen, and Sulfur)
MOMENTA_DIM = 9   # Dimension of momenta vectors (for Carbon, Oxygen, and Sulfur)

# Model hyperparameters
NUM_LAYERS = 4         # Number of layers in the model
HIDDEN_DIM = 256        # Dimension of the hidden layers in the model

# Training hyperparameters
BATCH_SIZE = 64        # Batch size for training  
NUM_EPOCHS = 20       # Number of training epochs
LEARNING_RATE = 3e-5   # Learning rate for optimizer
TEST_SIZE = 0.2        # Proportion of dataset to be used as the test set 
RANDOM_STATE = 42      # Random seed for reproducibility

# Force CUDA device
DEVICE = torch.device("cuda")  
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
print(f"Using device: {DEVICE}")

# Model saving path
MODEL_SAVE_PATH = "cinn_model.pth"

# *********************************************************************

# Load the dataset
data = pd.read_csv(DATA_FILE_PATH)

# Extract positions and momenta
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Split into training and testing sets
train_momenta, test_momenta, train_position, test_position = train_test_split(
    momenta, position, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Create DataLoader for training
train_dataset = TensorDataset(torch.tensor(train_momenta, dtype=torch.float32).cuda(), 
                              torch.tensor(train_position, dtype=torch.float32).cuda())
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(torch.tensor(test_momenta, dtype=torch.float32).cuda(),
                             torch.tensor(test_position, dtype=torch.float32).cuda())  
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define ConditionalCouplingLayer
class ConditionalCouplingLayer(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_cond_inputs, mask):
        super().__init__()
        self.num_inputs = num_inputs
        self.mask = mask.cuda()

        self.net_s = nn.Sequential(
            nn.Linear(num_inputs + num_cond_inputs, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden), 
            nn.ReLU(),
            nn.Linear(num_hidden, num_inputs),
            nn.Tanh()
        ).cuda()

        self.net_t = nn.Sequential(
            nn.Linear(num_inputs + num_cond_inputs, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_inputs)
        ).cuda()

    def forward(self, x, cond_inputs):
        x_masked = x * self.mask
        s = self.net_s(torch.cat([x_masked, cond_inputs], dim=1)) * (1 - self.mask)
        t = self.net_t(torch.cat([x_masked, cond_inputs], dim=1)) * (1 - self.mask)
        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def inverse(self, z, cond_inputs):
        z_masked = z * self.mask
        s = self.net_s(torch.cat([z_masked, cond_inputs], dim=1)) * (1 - self.mask)
        t = self.net_t(torch.cat([z_masked, cond_inputs], dim=1)) * (1 - self.mask)
        x = z_masked + (1 - self.mask) * (z - t) * torch.exp(-s)
        log_det = -torch.sum(s, dim=1)
        return x, log_det

# Define CINN Model 
class CINNModel(nn.Module):
    def __init__(self, num_inputs, num_cond_inputs, num_hidden, num_layers):
        super().__init__()

        masks = [torch.arange(0, num_inputs) % 2 == (i % 2) for i in range(num_layers)]
        self.layers = nn.ModuleList([
            ConditionalCouplingLayer(
                num_inputs=num_inputs,
                num_hidden=num_hidden,
                num_cond_inputs=num_cond_inputs,
                mask=masks[i].float() 
            ) for i in range(num_layers)
        ]).cuda()

    def forward(self, x, cond_inputs):
        log_det_total = torch.zeros(x.shape[0], device=DEVICE)
        for layer in self.layers:
            x, log_det = layer(x, cond_inputs)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, z, cond_inputs):
        x = z
        for layer in reversed(self.layers):
            x, _ = layer.inverse(x, cond_inputs)
        return x

# Initialize the model
model = CINNModel(num_inputs=POSITION_DIM,
                  num_cond_inputs=MOMENTA_DIM,
                  num_hidden=HIDDEN_DIM,
                  num_layers=NUM_LAYERS).cuda()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, (momenta_batch, position_batch) in enumerate(train_loader):
        momenta_batch, position_batch = momenta_batch.cuda(), position_batch.cuda()

        optimizer.zero_grad()
        z, log_det = model(position_batch, cond_inputs=momenta_batch)

        # Compute loss
        log_likelihood = torch.sum(-0.5 * z**2 - 0.5 * torch.log(torch.tensor(2 * torch.pi).cuda()), dim=1) 
        loss = -(log_likelihood + log_det).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# Testing loop: Evaluating the model
model.eval()
with torch.no_grad():
    total_mse = 0
    for momenta_batch, position_batch in test_loader:
        momenta_batch, position_batch = momenta_batch.cuda(), position_batch.cuda()

        # Generate random noise for sampling
        z = torch.randn(momenta_batch.size(0), POSITION_DIM, device=DEVICE)

        # Use the inverse function to predict positions
        predicted_positions = model.inverse(z, cond_inputs=momenta_batch)

        total_mse += torch.mean((predicted_positions - position_batch)**2).item()

    avg_mse = total_mse / len(test_loader)
    print(f"Average Test MSE: {avg_mse:.4f}")

# Example of using the model for prediction
sample_momenta = torch.randn(1, MOMENTA_DIM, device=DEVICE)  # Random momenta for demonstration
z = torch.randn(1, POSITION_DIM, device=DEVICE)
predicted_position = model.inverse(z, cond_inputs=sample_momenta)