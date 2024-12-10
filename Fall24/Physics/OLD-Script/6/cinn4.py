import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# File path
DATA_FILE_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'

# Dimensions
POSITION_DIM = 9
MOMENTA_DIM = 9

# Force CUDA device
DEVICE = torch.device("cuda")
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
print(f"Using device: {DEVICE}")

# Load and preprocess data
data = pd.read_csv(DATA_FILE_PATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Split into training and testing sets
train_momenta, test_momenta, train_position, test_position = train_test_split(
    momenta, position, test_size=0.2, random_state=42
)

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

def create_data_loaders(batch_size):
    train_dataset = TensorDataset(torch.tensor(train_momenta, dtype=torch.float32).cuda(), 
                                  torch.tensor(train_position, dtype=torch.float32).cuda())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.tensor(test_momenta, dtype=torch.float32).cuda(),
                                 torch.tensor(test_position, dtype=torch.float32).cuda())  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def objective(trial):
    # Define hyperparameters to optimize
    num_layers = trial.suggest_int('num_layers', 2, 10)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    num_epochs = trial.suggest_int('num_epochs', 10, 50)

    # Create data loaders
    train_loader, test_loader = create_data_loaders(batch_size)

    # Initialize the model with trial parameters
    model = CINNModel(num_inputs=POSITION_DIM,
                      num_cond_inputs=MOMENTA_DIM,
                      num_hidden=hidden_dim,
                      num_layers=num_layers).cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for momenta_batch, position_batch in train_loader:
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
        
        # Report intermediate metric
        trial.report(avg_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Evaluation
    model.eval()
    with torch.no_grad():
        total_mse = 0
        for momenta_batch, position_batch in test_loader:
            momenta_batch, position_batch = momenta_batch.cuda(), position_batch.cuda()
            z = torch.randn(momenta_batch.size(0), POSITION_DIM, device=DEVICE)
            predicted_positions = model.inverse(z, cond_inputs=momenta_batch)
            total_mse += torch.mean((predicted_positions - position_batch)**2).item()

        avg_mse = total_mse / len(test_loader)

    return avg_mse

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)  # You can adjust the number of trials

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Save the best model
best_model = CINNModel(num_inputs=POSITION_DIM,
                       num_cond_inputs=MOMENTA_DIM,
                       num_hidden=trial.params['hidden_dim'],
                       num_layers=trial.params['num_layers']).cuda()

best_model_path = "best_cinn_model.pth"
torch.save(best_model.state_dict(), best_model_path)
print(f"Best model saved to {best_model_path}")