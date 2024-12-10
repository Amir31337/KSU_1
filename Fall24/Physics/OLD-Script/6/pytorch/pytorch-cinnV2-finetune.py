import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from FrEIA.framework import *
from FrEIA.modules import *
import optuna

######################
#  Hyperparameters   #
######################
DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
N_EPOCHS = 100
POSITION_DIM = 9
MOMENTA_DIM = 9
GRAD_CLAMP = 15
K_FOLDS = 5
DATA_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
PRINT_INTERVAL = 10

######################
#  Model Definition  #
######################

def subnet_fc(c_in, c_out, hidden_dim):
    return nn.Sequential(
        nn.Linear(c_in, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, c_out)
    )

def create_cinn(latent_dim, num_coupling_blocks, subnet_hidden_dim, coupling_clamp):
    cond_node = ConditionNode(MOMENTA_DIM)
    nodes = [InputNode(POSITION_DIM, name='input')]

    for k in range(num_coupling_blocks):
        nodes.append(Node(nodes[-1], GLOWCouplingBlock,
                          {'subnet_constructor': lambda c_in, c_out: subnet_fc(c_in, c_out, subnet_hidden_dim), 
                           'clamp': coupling_clamp},
                          conditions=cond_node, name=f'coupling_{k}'))
        nodes.append(Node(nodes[-1], PermuteRandom, {'seed': k}, name=f'permute_{k}'))

    nodes.append(OutputNode(nodes[-1], name='output'))
    nodes.append(cond_node)

    model = ReversibleGraphNet(nodes, verbose=False).to(DEVICE)
    return model

######################
#  Helper Functions  #
######################

def mse_loss(pred_pos, true_pos):
    return torch.mean((pred_pos - true_pos) ** 2)

def train_epoch(model, train_loader, optimizer, grad_clamp):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        z, log_jac_det = model(x, c=[y])

        loss = 0.5 * torch.sum(z**2, dim=1) - log_jac_det
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clamp)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def test_epoch(model, test_loader, x_test):
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for i, y in enumerate(test_loader):
            y = y[0].to(DEVICE)

            z_sampled = torch.randn(y.size(0), POSITION_DIM).to(DEVICE)

            x_pred, _ = model(z_sampled, c=[y], rev=True)

            mse = mse_loss(x_pred, x_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].to(DEVICE))
            total_mse += mse.item()

    return total_mse / len(test_loader)

def k_fold_cross_validation(data, k_folds, batch_size, latent_dim, num_coupling_blocks, subnet_hidden_dim, coupling_clamp, lr_init):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
    momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values
    
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(position)):
        print(f"Fold {fold + 1}/{k_folds}")

        x_train, x_val = position[train_idx], position[val_idx]
        y_train, y_val = momenta[train_idx], momenta[val_idx]

        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(y_val), batch_size=batch_size, shuffle=False)

        model = create_cinn(latent_dim, num_coupling_blocks, subnet_hidden_dim, coupling_clamp)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)

        for epoch in range(N_EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, GRAD_CLAMP)
            if (epoch + 1) % PRINT_INTERVAL == 0:
                print(f"Epoch {epoch + 1}/{N_EPOCHS}, Loss: {train_loss:.4f}")

        val_mse = test_epoch(model, val_loader, x_val)
        print(f"Validation MSE: {val_mse:.4f}")
        fold_results.append(val_mse)

    return np.mean(fold_results)

###############
#  Optuna Objective Function  #
###############

def objective(trial):
    # Hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    latent_dim = trial.suggest_categorical('latent_dim', [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    subnet_hidden_dim = trial.suggest_categorical('subnet_hidden_dim', [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    num_coupling_blocks = trial.suggest_categorical('num_coupling_blocks', [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    coupling_clamp = trial.suggest_categorical('coupling_clamp', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lr_init = trial.suggest_categorical('lr_init', [1e-1, 5e-1, 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4])

    data = pd.read_csv(DATA_PATH)

    # Perform K-Fold Cross Validation and return the average MSE
    average_mse = k_fold_cross_validation(data, K_FOLDS, batch_size, latent_dim, num_coupling_blocks, subnet_hidden_dim, coupling_clamp, lr_init)
    return average_mse

###############
#  Main Optuna Loop  #
###############

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)  # Adjust the number of trials

    print("\nBest hyperparameters:")
    print(study.best_params)

    print(f"\nBest Average MSE: {study.best_value:.4f}")
