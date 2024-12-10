import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration and hyperparameters
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 4096,
    "latent_size": 128,
    "hidden_size": 512,  # Soft-coded hidden layer size
    "epochs": 50,
    "data_path": "/home/g/ghanaatian/MYFILES/FALL24/Physics/1M.csv",
    "test_size": 0.2,
    "validation_split": 0.2,
    "random_seed": 42,
    "learning_rate": 1e-3,
    "clip_grad": 1.0,
    "model_save_path": "simpleVAE_optimized.pth"
}

kwargs = {'num_workers': 1, 'pin_memory': True} if config["device"].type == "cuda" else {}

# Load and preprocess the dataset
data = pd.read_csv(config["data_path"])

# Separate features and targets
features_columns = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
target_columns = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']
positions = data[features_columns].values
momenta = data[target_columns].values

# Standardize the data
scaler_momenta = StandardScaler().fit(momenta)
momenta = scaler_momenta.transform(momenta)

# Split the data into training, validation, and testing sets
momenta_train, momenta_test, positions_train, positions_test = train_test_split(
    momenta, positions, test_size=config["test_size"], random_state=config["random_seed"])
momenta_train, momenta_val, positions_train, positions_val = train_test_split(
    momenta_train, positions_train, test_size=config["validation_split"], random_state=config["random_seed"])

# Convert to PyTorch tensors and create DataLoader
train_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_train, dtype=torch.float32), torch.tensor(positions_train, dtype=torch.float32))
val_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_val, dtype=torch.float32), torch.tensor(positions_val, dtype=torch.float32))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_test, dtype=torch.float32), torch.tensor(positions_test, dtype=torch.float32))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, **kwargs)

class VAE(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, output_size):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Encoder
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.elu = nn.ELU()

    def encode(self, x):
        h1 = self.elu(self.fc1(x))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.elu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Define loss function
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

# Create and train the model
model = VAE(momenta.shape[1], config["latent_size"], config["hidden_size"], positions.shape[1]).to(config["device"])
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Lists to store train and validation losses
train_losses = []
val_losses = []

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(config["device"]), target.to(config["device"])
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, target, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
        train_loss += loss.item()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}')

def validate(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(config["device"]), target.to(config["device"])
            recon_batch, mu, logvar = model(data)
            val_loss += loss_function(recon_batch, target, mu, logvar).item()

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Validation Loss: {val_loss:.4f}')

    return val_loss

for epoch in range(1, config["epochs"] + 1):
    train(epoch)
    val_loss = validate(epoch)

# Save and load the model
torch.save(model.state_dict(), config["model_save_path"])
model.load_state_dict(torch.load(config["model_save_path"]))
model.eval()

# Evaluate on the test dataset
mse_losses = []
mae_losses = []
r2_scores = []
with torch.no_grad():
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(config["device"]), target.to(config["device"])
        recon_batch, _, _ = model(data)
        mse_losses.append(mean_squared_error(target.cpu().numpy(), recon_batch.cpu().numpy()))
        mae_losses.append(mean_absolute_error(target.cpu().numpy(), recon_batch.cpu().numpy()))
        r2_scores.append(r2_score(target.cpu().numpy(), recon_batch.cpu().numpy()))

print(f'Test MSE: {sum(mse_losses) / len(mse_losses):.4f}')
print(f'Test MAE: {sum(mae_losses) / len(mae_losses):.4f}')
print(f'Test R2 Score: {sum(r2_scores) / len(r2_scores):.4f}')

