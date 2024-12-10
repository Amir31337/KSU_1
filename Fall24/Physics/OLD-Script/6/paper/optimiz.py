import sys
import subprocess

# Check if Optuna is installed
try:
    import optuna
except ImportError:
    print("Optuna is not installed. Attempting to install it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import the original model and related functions
from cinnV1 import AtomicReconstructionModel, ConditionalNet, CouplingLayer, CINN, load_and_preprocess_data

# Configuration
BASE_CONFIG = {
    'position_dim': 9,
    'momenta_dim': 9,
    'data_file': '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv',
    'model_save_path': 'optimized_atomic_reconstruction_model.pth',
    'latent_save_path': 'optimized_latent_representations.pt'
}

def objective(trial):
    # Define the hyperparameters to optimize
    config = {
        'latent_dim': trial.suggest_int('latent_dim', 32, 256),
        'conditional_dim': trial.suggest_int('conditional_dim', 32, 256),
        'num_coupling_layers': trial.suggest_int('num_coupling_layers', 4, 16),
        'batch_size': trial.suggest_categorical('batch_size', [1024, 2048, 4096, 8192]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'num_epochs': trial.suggest_int('num_epochs', 50, 200),
        'kl_weight': trial.suggest_loguniform('kl_weight', 1e-4, 1e-1),
        'hidden_dim': trial.suggest_int('hidden_dim', 128, 512),
    }
    
    # Merge with base config
    config.update(BASE_CONFIG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train, y_train, X_test, y_test, _, _ = load_and_preprocess_data(config['data_file'])
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = AtomicReconstructionModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x, y)
            
            recon_loss = nn.MSELoss()(x_recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + config['kl_weight'] * kl_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Early stopping
        if epoch % 10 == 0:
            model.eval()
            test_mse = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    x_recon, _, _ = model(x, y)
                    test_mse += nn.MSELoss()(x_recon, x).item()
            test_mse /= len(test_loader)
            
            trial.report(test_mse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return test_mse

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=3600*8)  # Run for 8 hours

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save the best model
    best_config = BASE_CONFIG.copy()
    best_config.update(trial.params)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test, _, _ = load_and_preprocess_data(best_config['data_file'])
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=best_config['batch_size'], shuffle=True)

    best_model = AtomicReconstructionModel(best_config).to(device)
    optimizer = optim.Adam(best_model.parameters(), lr=best_config['learning_rate'])

    for epoch in range(best_config['num_epochs']):
        best_model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = best_model(x, y)
            
            recon_loss = nn.MSELoss()(x_recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + best_config['kl_weight'] * kl_loss

            loss.backward()
            optimizer.step()

    torch.save(best_model.state_dict(), best_config['model_save_path'])
    print(f"Best model saved to {best_config['model_save_path']}")

if __name__ == "__main__":
    main()