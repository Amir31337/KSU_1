import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import optuna

# Configuration
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "epochs": 200,
    "data_path": "/home/g/ghanaatian/MYFILES/FALL24/Physics/1M.csv",
    "test_size": 0.2,
    "validation_split": 0.2,
    "random_seed": 42,
    "clip_grad": 1.0,
    "n_trials": 50
}

'''# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True'''

# Load and preprocess the dataset
data = pd.read_csv(config["data_path"])

features_columns = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
target_columns = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']
positions = data[features_columns].values
momenta = data[target_columns].values

scaler_momenta = MinMaxScaler().fit(momenta)
momenta = scaler_momenta.transform(momenta)

momenta_train, momenta_test, positions_train, positions_test = train_test_split(
    momenta, positions, test_size=config["test_size"], random_state=config["random_seed"])
momenta_train, momenta_val, positions_train, positions_val = train_test_split(
    momenta_train, positions_train, test_size=config["validation_split"], random_state=config["random_seed"])

class VAE(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, output_size):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.elu = nn.ELU()

    def encode(self, x):
        h1 = self.elu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

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

def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

# Create DataLoaders
kwargs = {'num_workers': 1, 'pin_memory': True} if config["device"].type == "cuda" else {}
train_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_train, dtype=torch.float32), 
                                               torch.tensor(positions_train, dtype=torch.float32))
val_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_val, dtype=torch.float32), 
                                             torch.tensor(positions_val, dtype=torch.float32))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(momenta_test, dtype=torch.float32), 
                                              torch.tensor(positions_test, dtype=torch.float32))

def train_and_evaluate(trial):
    # Suggest values for hyperparameters
    batch_size = 1024
    latent_size = trial.suggest_int("latent_size", 16, 1024)
    hidden_size = trial.suggest_int("hidden_size", 64, 1024)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    # Create and train the model
    model = VAE(momenta.shape[1], latent_size, hidden_size, positions.shape[1]).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config["device"]), target.to(config["device"])
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, target, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config["device"]), target.to(config["device"])
                recon_batch, mu, logvar = model(data)
                val_loss += loss_function(recon_batch, target, mu, logvar).item()
        val_loss /= len(val_loader.dataset)

        # Report intermediate value
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Evaluate on test set
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(config["device"]), target.to(config["device"])
            recon_batch, _, _ = model(data)
            test_loss += mean_squared_error(target.cpu().numpy(), recon_batch.cpu().numpy())
    test_mse = test_loss / len(test_loader)

    return test_mse

# Create a study object and optimize the objective function
study = optuna.create_study(direction="minimize")

try:
    study.optimize(train_and_evaluate, n_trials=config["n_trials"])
except RuntimeError as e:
    if "CUDA error: initialization error" in str(e):
        print("CUDA initialization error occurred. Retrying with CPU...")
        config["device"] = torch.device("cpu")
        study.optimize(train_and_evaluate, n_trials=config["n_trials"])
    else:
        raise e

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Optional: Plot optimization history
try:
    import matplotlib.pyplot as plt
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
except ImportError:
    print("Matplotlib is not installed. Skipping plot generation.")
