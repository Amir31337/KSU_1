import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from FrEIA.framework import *
from FrEIA.modules import *

######################
#  Hyperparameters   #
######################

# General settings
DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1024
N_EPOCHS = 100
LR_INIT = 1.0e-3
LATENT_DIM = 9
POSITION_DIM = 9
MOMENTA_DIM = 9
GRAD_CLAMP = 15
K_FOLDS = 5

# Model architecture
SUBNET_HIDDEN_DIM = 256
NUM_COUPLING_BLOCKS = 6
COUPLING_CLAMP = 2.0

# Data
DATA_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'

# Training
PRINT_INTERVAL = 10

######################
#  Model Definition  #
######################

def subnet_fc(c_in, c_out):
    return nn.Sequential(
        nn.Linear(c_in, SUBNET_HIDDEN_DIM), nn.ReLU(),
        nn.Linear(SUBNET_HIDDEN_DIM, c_out)
    )

def create_cinn():
    cond_node = ConditionNode(MOMENTA_DIM)
    nodes = [InputNode(POSITION_DIM, name='input')]

    for k in range(NUM_COUPLING_BLOCKS):
        nodes.append(Node(nodes[-1], GLOWCouplingBlock,
                          {'subnet_constructor': subnet_fc, 'clamp': COUPLING_CLAMP},
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

def train_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        z, log_jac_det = model(x, c=[y])

        loss = 0.5 * torch.sum(z**2, dim=1) - log_jac_det
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLAMP)

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

def k_fold_cross_validation(data, k_folds):
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

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(y_val), batch_size=BATCH_SIZE, shuffle=False)

        model = create_cinn()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)

        for epoch in range(N_EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer)
            if (epoch + 1) % PRINT_INTERVAL == 0:
                print(f"Epoch {epoch + 1}/{N_EPOCHS}, Loss: {train_loss:.4f}")

        val_mse = test_epoch(model, val_loader, x_val)
        print(f"Validation MSE: {val_mse:.4f}")
        fold_results.append(val_mse)

    return fold_results

###############
#  Main Loop  #
###############

if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)

    fold_results = k_fold_cross_validation(data, K_FOLDS)

    print("\nK-Fold Cross-Validation Results:")
    for fold, mse in enumerate(fold_results):
        print(f"Fold {fold + 1} MSE: {mse:.4f}")

    average_mse = np.mean(fold_results)
    print(f"\nAverage MSE across {K_FOLDS} folds: {average_mse:.4f}")