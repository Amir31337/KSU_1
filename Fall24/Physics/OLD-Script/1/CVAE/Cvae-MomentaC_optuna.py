import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import optuna

# Configuration and hyperparameters
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 2048,
    "epochs": 100,
    "learning_rate": 1e-3,
    "use_condition": True,
    "data_path": "/home/g/ghanaatian/MYFILES/FALL24/Physics/1M.csv",
    "test_size": 0.2,
    "validation_split": 0.2,
    "random_seed": 42,
    "model_path": "best_cvae_momentum_to_position.pth",
    "early_stopping_patience": 20,
}

# Load and preprocess the dataset (same as before)
data = pd.read_csv(config["data_path"])
positions = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values
scaler_momenta = StandardScaler().fit(momenta)
momenta = scaler_momenta.transform(momenta)
scaler_positions = StandardScaler().fit(positions)
positions = scaler_positions.transform(positions)
condition = np.sqrt(np.abs(momenta))

# Split the data (same as before)
momenta_train, momenta_test, positions_train, positions_test, condition_train, condition_test = train_test_split(
    momenta, positions, condition, test_size=config["test_size"], random_state=config["random_seed"])
momenta_train, momenta_val, positions_train, positions_val, condition_train, condition_val = train_test_split(
    momenta_train, positions_train, condition_train, test_size=config["validation_split"], random_state=config["random_seed"])

# CVAE model definition (same as before)
class CVAE(nn.Module):
    def __init__(self, input_size, latent_size, output_size, condition_size, hidden_sizes):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.condition_size = condition_size

        self.fc1 = nn.Linear(input_size + condition_size, hidden_sizes["encode_layer1"])
        self.bn1 = nn.BatchNorm1d(hidden_sizes["encode_layer1"])
        self.fc2 = nn.Linear(hidden_sizes["encode_layer1"], hidden_sizes["encode_layer2"])
        self.bn2 = nn.BatchNorm1d(hidden_sizes["encode_layer2"])
        self.fc21 = nn.Linear(hidden_sizes["encode_layer2"], latent_size)
        self.fc22 = nn.Linear(hidden_sizes["encode_layer2"], latent_size)

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

# Loss function
def loss_function(recon_x, x, mu, logvar, kl_weight):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + kl_weight * KLD

# Training function
def train_and_validate(model, train_loader, val_loader, optimizer, scheduler, num_epochs, kl_weight, device):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target, condition) in enumerate(train_loader):
            data, target, condition = data.to(device), target.to(device), condition.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, condition)
            loss = loss_function(recon_batch, target, mu, logvar, kl_weight)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target, condition in val_loader:
                data, target, condition = data.to(device), target.to(device), condition.to(device)
                recon_batch, mu, logvar = model(data, condition)
                val_loss += loss_function(recon_batch, target, mu, logvar, kl_weight).item()

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss

# Optuna objective function
def objective(trial):
    # Define the hyperparameters to optimize
    latent_size = trial.suggest_int("latent_size", 16, 1024)
    encode_layer1 = trial.suggest_int("encode_layer1", 16, 1024)
    encode_layer2 = trial.suggest_int("encode_layer2", 16, 1024)
    decode_layer1 = trial.suggest_int("decode_layer1", 16, 1024)
    decode_layer2 = trial.suggest_int("decode_layer2", 16, 1024)
    kl_weight = trial.suggest_loguniform("kl_weight", 1e-5, 1e-1)
    scheduler_patience = trial.suggest_int("scheduler_patience", 5, 20)
    scheduler_factor = trial.suggest_uniform("scheduler_factor", 0.1, 0.5)

    hidden_layer_sizes = {
        "encode_layer1": encode_layer1,
        "encode_layer2": encode_layer2,
        "decode_layer1": decode_layer1,
        "decode_layer2": decode_layer2
    }

    # Create model and optimizer
    input_size = momenta.shape[1]
    output_size = positions.shape[1]
    condition_size = condition.shape[1]
    model = CVAE(input_size, latent_size, output_size, condition_size, hidden_layer_sizes).to(config["device"])

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience, factor=scheduler_factor)

    # Create DataLoader
    kwargs = {'num_workers': 1, 'pin_memory': True} if config["device"].type == "cuda" else {}
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_train, dtype=torch.float32),
                                                   torch.tensor(positions_train, dtype=torch.float32),
                                                   torch.tensor(condition_train, dtype=torch.float32))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_val, dtype=torch.float32),
                                                 torch.tensor(positions_val, dtype=torch.float32),
                                                 torch.tensor(condition_val, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, **kwargs)

    # Train and validate
    best_val_loss = train_and_validate(model, train_loader, val_loader, optimizer, scheduler, config["epochs"], kl_weight, config["device"])

    return best_val_loss

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)  # You can adjust the number of trials

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Use the best parameters to train the final model and evaluate on the test set
best_params = study.best_params
best_model = CVAE(input_size, best_params["latent_size"], output_size, condition_size, {
    "encode_layer1": best_params["encode_layer1"],
    "encode_layer2": best_params["encode_layer2"],
    "decode_layer1": best_params["decode_layer1"],
    "decode_layer2": best_params["decode_layer2"]
}).to(config["device"])

best_optimizer = optim.Adam(best_model.parameters(), lr=config["learning_rate"])
best_scheduler = optim.lr_scheduler.ReduceLROnPlateau(best_optimizer, 'min', patience=best_params["scheduler_patience"], factor=best_params["scheduler_factor"])

# Train the best model
train_and_validate(best_model, train_loader, val_loader, best_optimizer, best_scheduler, config["epochs"], best_params["kl_weight"], config["device"])

# Evaluate on test set
test_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_test, dtype=torch.float32),
                                              torch.tensor(positions_test, dtype=torch.float32),
                                              torch.tensor(condition_test, dtype=torch.float32))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, **kwargs)

best_model.eval()
predictions = []
targets = []
with torch.no_grad():
    for data, target, condition in test_loader:
        data, target, condition = data.to(config["device"]), target.to(config["device"]), condition.to(config["device"])
        recon_batch, _, _ = best_model(data, condition)
        predictions.append(recon_batch.cpu().numpy())
        targets.append(target.cpu().numpy())

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)

# Inverse transform the predictions and targets
predictions = scaler_positions.inverse_transform(predictions)
targets = scaler_positions.inverse_transform(targets)

mse = mean_squared_error(targets, predictions)
print(f'Test MSE with best parameters: {mse:.4f}')
