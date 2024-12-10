import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from FrEIA.framework import *
from FrEIA.modules import *
from itertools import product
import random

######################
#  Hyperparameters   #
######################

# General settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
K_FOLDS = 5

# Hyperparameter ranges
BATCH_SIZES = [512, 1024, 2048]
N_EPOCHS_RANGE = [500, 1000, 1500]
LR_RANGE = [1e-4, 1e-3, 1e-2]
LATENT_DIMS = [8, 9, 10]
GRAD_CLAMP_RANGE = [10, 15, 20]

SUBNET_HIDDEN_DIMS = [128, 256, 512]
NUM_COUPLING_BLOCKS_RANGE = [4, 6, 8]
COUPLING_CLAMP_RANGE = [1.5, 2.0, 2.5]

# Number of random combinations to try
N_RANDOM_SEARCHES = 20

######################
#  Model Definition  #
######################

def subnet_fc(c_in, c_out, hidden_dim):
    return nn.Sequential(
        nn.Linear(c_in, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, c_out)
    )

def create_cinn(latent_dim, subnet_hidden_dim, num_coupling_blocks, coupling_clamp):
    cond_node = ConditionNode(latent_dim)
    nodes = [InputNode(latent_dim, name='input')]

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

def test_epoch(model, test_loader, x_test, latent_dim):
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for i, y in enumerate(test_loader):
            y = y[0].to(DEVICE)

            z_sampled = torch.randn(y.size(0), latent_dim).to(DEVICE)

            x_pred, _ = model(z_sampled, c=[y], rev=True)

            mse = mse_loss(x_pred, x_test[i * y.size(0):(i + 1) * y.size(0)].to(DEVICE))
            total_mse += mse.item()

    return total_mse / len(test_loader)

def k_fold_cross_validation(data, k_folds, hyperparams):
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

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(y_val), batch_size=hyperparams['batch_size'], shuffle=False)

        model = create_cinn(hyperparams['latent_dim'], hyperparams['subnet_hidden_dim'], 
                            hyperparams['num_coupling_blocks'], hyperparams['coupling_clamp'])
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])

        for epoch in range(hyperparams['n_epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, hyperparams['grad_clamp'])
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{hyperparams['n_epochs']}, Loss: {train_loss:.4f}")

        val_mse = test_epoch(model, val_loader, x_val, hyperparams['latent_dim'])
        print(f"Validation MSE: {val_mse:.4f}")
        fold_results.append(val_mse)

    return np.mean(fold_results)

def random_search(data):
    best_mse = float('inf')
    best_hyperparams = None

    for _ in range(N_RANDOM_SEARCHES):
        hyperparams = {
            'batch_size': random.choice(BATCH_SIZES),
            'n_epochs': random.choice(N_EPOCHS_RANGE),
            'lr': random.choice(LR_RANGE),
            'latent_dim': random.choice(LATENT_DIMS),
            'grad_clamp': random.choice(GRAD_CLAMP_RANGE),
            'subnet_hidden_dim': random.choice(SUBNET_HIDDEN_DIMS),
            'num_coupling_blocks': random.choice(NUM_COUPLING_BLOCKS_RANGE),
            'coupling_clamp': random.choice(COUPLING_CLAMP_RANGE)
        }

        print("\nTesting hyperparameters:", hyperparams)
        mse = k_fold_cross_validation(data, K_FOLDS, hyperparams)
        print(f"Average MSE: {mse:.4f}")

        if mse < best_mse:
            best_mse = mse
            best_hyperparams = hyperparams

    return best_hyperparams, best_mse

###############
#  Main Loop  #
###############

if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)

    best_hyperparams, best_mse = random_search(data)

    print("\nBest Hyperparameters:")
    for key, value in best_hyperparams.items():
        print(f"{key}: {value}")
    print(f"\nBest Average MSE: {best_mse:.4f}")