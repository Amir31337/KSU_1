import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from FrEIA import framework as fr
from FrEIA import modules as fm

# Data file path
data_file = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'

# Dimensions of position and momenta
POSITION_DIM = 9
MOMENTA_DIM = 9

# Set random seed for reproducibility
random_seed = 42
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
        
        def subnet_fc(c_in, c_out):
            return nn.Sequential(
                nn.Linear(c_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, c_out)
            )
        
        self.cinn = fr.SequenceINN(input_dim)
        for _ in range(num_layers):
            self.cinn.append(fm.GLOWCouplingBlock, subnet_constructor=subnet_fc, clamp=2.0)
            self.cinn.append(fm.PermuteRandom)
        
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
    total_loss = 0
    for epoch in range(num_epochs):
        for batch_idx, (positions, momenta) in enumerate(train_loader):
            positions, momenta = positions.to(device), momenta.to(device)
            
            optimizer.zero_grad()
            
            z, log_jac_det = model(positions, momenta)
            loss = torch.mean(0.5 * torch.sum(z**2, dim=1) - log_jac_det)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss / (num_epochs * len(train_loader))

def objective(trial):
    # Define hyperparameters to optimize
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    num_layers = trial.suggest_int('num_layers', 4, 16)
    batch_size = trial.suggest_int('batch_size', 32, 256)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    num_epochs = trial.suggest_int('num_epochs', 5, 50)

    # Load and prepare data
    data = pd.read_csv(data_file)
    positions = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
    momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

    train_positions, test_positions, train_momenta, test_momenta = train_test_split(
        positions, momenta, test_size=0.2, random_state=random_seed)

    train_positions = torch.FloatTensor(train_positions)
    train_momenta = torch.FloatTensor(train_momenta)

    train_dataset = TensorDataset(train_positions, train_momenta)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = CINN(POSITION_DIM, MOMENTA_DIM, hidden_dim, num_layers).to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    avg_loss = train_cinn(model, train_loader, optimizer, num_epochs)

    return avg_loss

def optimize_hyperparameters(n_trials=100):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return study.best_params

if __name__ == "__main__":
    best_params = optimize_hyperparameters(n_trials=100)
    print("Best hyperparameters:", best_params)