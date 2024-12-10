#condition is the initial position in cVAE

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# cuda setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == "cuda" else {}

# hyper params
batch_size = 256
latent_size = 9
epochs = 200

# Load and preprocess the dataset
data = pd.read_csv('1M.csv')

# Separate features and targets
positions = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Standardize the data
scaler_momenta = MinMaxScaler().fit(momenta)
momenta = scaler_momenta.transform(momenta)

# Split the data into training and testing sets
momenta_train, momenta_test, positions_train, positions_test = train_test_split(
    momenta, positions, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and create DataLoader
'''
condition is the initial position in cVAE
'''
# condition is the initial position in cVAE#
train_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_train, dtype=torch.float32), torch.tensor(positions_train, dtype=torch.float32), torch.tensor(positions_train, dtype=torch.float32))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_test, dtype=torch.float32), torch.tensor(positions_test, dtype=torch.float32), torch.tensor(positions_test, dtype=torch.float32))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


class CVAE(nn.Module):
    def __init__(self, input_size, latent_size, output_size, condition_size):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.condition_size = condition_size

        # encode
        self.fc1 = nn.Linear(input_size + condition_size, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + condition_size, 400)
        self.fc4 = nn.Linear(400, output_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h3 = self.elu(self.fc3(inputs))
        return self.fc4(h3)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# Define loss function (Reconstruction + KL divergence losses)
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

# Create model
input_size = momenta.shape[1]  # Number of features for final momenta (9)
output_size = positions.shape[1]  # Number of features for initial positions (9)
condition_size = momenta.shape[1]  # Number of conditional features (final momenta)
model = CVAE(input_size, latent_size, output_size, condition_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Lists to store train and test losses
train_losses = []
test_losses = []

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target, condition) in enumerate(train_loader):
        data, target, condition = data.to(device), target.to(device), condition.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, condition)
        loss = loss_function(recon_batch, target, mu, logvar)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        train_loss += loss.item()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    if epoch % 25 == 0:
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}')

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, target, condition) in enumerate(test_loader):
            data, target, condition = data.to(device), target.to(device), condition.to(device)
            recon_batch, mu, logvar = model(data, condition)
            test_loss += loss_function(recon_batch, target, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    if epoch % 25 == 0:
        print(f'Epoch: {epoch}, Test Loss: {test_loss:.4f}')

    return test_loss

for epoch in range(1, epochs + 1):
    train(epoch)
    test_loss = test(epoch)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('cVAE V3 Learning Curve positions Condition')
plt.legend()
plt.savefig('V3.png')
plt.show()

# Save the model
torch.save(model.state_dict(), "V3.pth")
print("Model saved as V3.pth")
