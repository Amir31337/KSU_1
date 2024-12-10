import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Configuration and hyperparameters
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 2048,
    "latent_size": 64,
    "epochs": 50,
    "learning_rate": 1e-3,
    "use_condition": True,
    "data_path": "/home/g/ghanaatian/MYFILES/FALL24/Physics/1M.csv",
    "test_size": 0.2,
    "validation_split": 0.2,
    "random_seed": 42,
    "model_path": "CVAE_V1.pth",
    "scheduler_patience": 7,
    "scheduler_factor": 0.35,
    "early_stopping_patience": 20,
    "hidden_layer_sizes": {
        "encode_layer1": 128,
        "encode_layer2": 32,
        "decode_layer1": 1024,
        "decode_layer2": 16
    },
    "kl_weight": 1
}

kwargs = {'num_workers': 1, 'pin_memory': True} if config["device"].type == "cuda" else {}

# Load and preprocess the dataset
data = pd.read_csv(config["data_path"])

# Separate features and targets
positions = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Standardize the data
scaler_momenta = StandardScaler().fit(momenta)
momenta = scaler_momenta.transform(momenta)

scaler_positions = StandardScaler().fit(positions)
positions = scaler_positions.transform(positions)

'''
Here
'''
# Calculate absolute value of momenta as the condition
condition = np.abs(momenta)

'''
Here
'''

# Split the data
momenta_train, momenta_test, positions_train, positions_test, condition_train, condition_test = train_test_split(
    momenta, positions, condition, test_size=config["test_size"], random_state=config["random_seed"])
momenta_train, momenta_val, positions_train, positions_val, condition_train, condition_val = train_test_split(
    momenta_train, positions_train, condition_train, test_size=config["validation_split"], random_state=config["random_seed"])

# Convert to PyTorch tensors and create DataLoader
train_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_train, dtype=torch.float32),
                                               torch.tensor(positions_train, dtype=torch.float32),
                                               torch.tensor(condition_train, dtype=torch.float32))
val_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_val, dtype=torch.float32),
                                             torch.tensor(positions_val, dtype=torch.float32),
                                             torch.tensor(condition_val, dtype=torch.float32))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_test, dtype=torch.float32),
                                              torch.tensor(positions_test, dtype=torch.float32),
                                              torch.tensor(condition_test, dtype=torch.float32))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, **kwargs)

# Define the CVAE model
class CVAE(nn.Module):
    def __init__(self, input_size, latent_size, output_size, condition_size, hidden_sizes):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.condition_size = condition_size

        # encode
        self.fc1 = nn.Linear(input_size + condition_size, hidden_sizes["encode_layer1"])
        self.bn1 = nn.BatchNorm1d(hidden_sizes["encode_layer1"])
        self.fc2 = nn.Linear(hidden_sizes["encode_layer1"], hidden_sizes["encode_layer2"])
        self.bn2 = nn.BatchNorm1d(hidden_sizes["encode_layer2"])
        self.fc21 = nn.Linear(hidden_sizes["encode_layer2"], latent_size)
        self.fc22 = nn.Linear(hidden_sizes["encode_layer2"], latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + condition_size, hidden_sizes["decode_layer1"])
        self.bn3 = nn.BatchNorm1d(hidden_sizes["decode_layer1"])
        self.fc4 = nn.Linear(hidden_sizes["decode_layer1"], hidden_sizes["decode_layer2"])
        self.bn4 = nn.BatchNorm1d(hidden_sizes["decode_layer2"])
        self.fc5 = nn.Linear(hidden_sizes["decode_layer2"], output_size)

        self.relu = nn.ReLU()

    def encode(self, x, c):
        x_c = torch.cat([x, c], dim=1)
        h1 = self.relu(self.bn1(self.fc1(x_c)))
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

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# Define loss function
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + config["kl_weight"] * KLD

# Create model and optimizer
input_size = momenta.shape[1]
output_size = positions.shape[1]
condition_size = condition.shape[1]
model = CVAE(input_size, config["latent_size"], output_size, condition_size, config["hidden_layer_sizes"]).to(config["device"])

optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config["scheduler_patience"], factor=config["scheduler_factor"])

# Training and validation processes follow as previously described

train_losses = []
val_losses = []

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target, condition) in enumerate(train_loader):
        data, target, condition = data.to(config["device"]), target.to(config["device"]), condition.to(config["device"])
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, condition)
        loss = loss_function(recon_batch, target, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    if (epoch % 10) == 0:
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}')

def validate(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target, condition in val_loader:
            data, target, condition = data.to(config["device"]), target.to(config["device"]), condition.to(config["device"])
            recon_batch, mu, logvar = model(data, condition)
            val_loss += loss_function(recon_batch, target, mu, logvar).item()

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    if (epoch % 10) == 0:
        print(f'Epoch: {epoch}, Validation Loss: {val_loss:.4f}')
    return val_loss

# Early stopping logic
best_val_loss = float('inf')
counter = 0

for epoch in range(1, config["epochs"] + 1):
    train(epoch)
    val_loss = validate(epoch)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), config["model_path"])
    else:
        counter += 1
        if counter >= config["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

print("Training completed.")

# Load the best model
model.load_state_dict(torch.load(config["model_path"]))
model.eval()

# Evaluation on the test dataset
predictions = []
targets = []
with torch.no_grad():
    for data, target, condition in test_loader:
        data, target, condition = data.to(config["device"]), target.to(config["device"]), condition.to(config["device"])
        recon_batch, _, _ = model(data, condition)
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
