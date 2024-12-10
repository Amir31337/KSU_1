import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# cuda setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == "cuda" else {}

# hyper params
batch_size = 256
latent_size = 256
epochs = 100
learning_rate = 1e-3

# Load and preprocess the dataset
data = pd.read_csv('/home/g/ghanaatian/MYFILES/FALL24/Physics/10K.csv')

# Separate features and targets
positions = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Standardize the data
scaler_momenta = StandardScaler().fit(momenta)
momenta = scaler_momenta.transform(momenta)

scaler_positions = StandardScaler().fit(positions)
positions = scaler_positions.transform(positions)

# Split the data
momenta_train, momenta_test, positions_train, positions_test = train_test_split(
    momenta, positions, test_size=0.2, random_state=42)
momenta_train, momenta_val, positions_train, positions_val = train_test_split(
    momenta_train, positions_train, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and create DataLoader
train_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_train, dtype=torch.float32), torch.tensor(positions_train, dtype=torch.float32))
val_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_val, dtype=torch.float32), torch.tensor(positions_val, dtype=torch.float32))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_test, dtype=torch.float32), torch.tensor(positions_test, dtype=torch.float32))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

class CVAE(nn.Module):
    def __init__(self, input_size, latent_size, output_size, condition_size):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.condition_size = condition_size

        # encode
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc21 = nn.Linear(256, latent_size)
        self.fc22 = nn.Linear(256, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + condition_size, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, output_size)

        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.relu(self.bn1(self.fc1(x)))
        h2 = self.relu(self.bn2(self.fc2(h1)))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z_c = torch.cat([z, c], dim=1)
        h3 = self.relu(self.bn3(self.fc3(z_c)))
        h4 = self.relu(self.bn4(self.fc4(h3)))
        return self.fc5(h4)

    def forward(self, x, c=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if c is None:
            c = torch.zeros(x.size(0), self.condition_size, device=x.device)
        return self.decode(z, c), mu, logvar

# Define loss function
def loss_function(recon_x, x, mu, logvar, kl_weight=0.01):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + kl_weight * KLD

# Create model
input_size = momenta.shape[1]
output_size = positions.shape[1]
condition_size = positions.shape[1]
model = CVAE(input_size, latent_size, output_size, condition_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

# Lists to store losses
train_losses = []
val_losses = []

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, condition) in enumerate(train_loader):
        data, condition = data.to(device), condition.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, condition)
        loss = loss_function(recon_batch, condition, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    if (epoch % 10) == 0:
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}')

def validate(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, condition in val_loader:
            data, condition = data.to(device), condition.to(device)
            recon_batch, mu, logvar = model(data, condition)
            val_loss += loss_function(recon_batch, condition, mu, logvar).item()

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    if (epoch % 10) == 0:
        print(f'Epoch: {epoch}, Validation Loss: {val_loss:.4f}')
    return val_loss

# Early stopping
best_val_loss = float('inf')
patience = 20
counter = 0

for epoch in range(1, epochs + 1):
    train(epoch)
    val_loss = validate(epoch)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_cvae_momentum_to_position.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

print("Training completed.")

# Load the best model
model.load_state_dict(torch.load("best_cvae_momentum_to_position.pth"))
model.eval()

# Evaluate on the test dataset
predictions = []
targets = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        recon_batch, _, _ = model(data, target)  # Pass the target as the condition
        predictions.append(recon_batch.cpu().numpy())
        targets.append(target.cpu().numpy())

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)

# Inverse transform the predictions and targets
predictions = scaler_positions.inverse_transform(predictions)
targets = scaler_positions.inverse_transform(targets)

mse = mean_squared_error(targets, predictions)
mae = mean_absolute_error(targets, predictions)
r2 = r2_score(targets, predictions)

print(f'Test MSE: {mse:.4f}')
print(f'Test MAE: {mae:.4f}')
print(f'Test R2 Score: {r2:.4f}')

print("Evaluation completed.")