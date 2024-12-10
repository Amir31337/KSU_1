import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import optuna
import os

# Configuration
CONFIG = {
    'position_dim': 9,
    'momenta_dim': 9,
    'num_epochs': 100,
    'data_file': 'cei_traning_orient_1.csv',
    'model_save_path': 'atomic_reconstruction_model.pth',
    'latent_save_path': 'latent_representations.pt',
    'n_splits': 5  # Number of folds for cross-validation
}

# Set up multi-GPU environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable cuDNN auto-tuner
torch.backends.cudnn.benchmark = True

class ConditionalNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class CouplingLayer(nn.Module):
    def __init__(self, dim, conditional_dim, hidden_dim):
        super().__init__()
        self.net = ConditionalNet(dim // 2 + conditional_dim, dim // 2, hidden_dim)

    def forward(self, x, c, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        if not reverse:
            t = self.net(torch.cat([x1, c], dim=1))
            y1, y2 = x1, x2 + t
        else:
            t = self.net(torch.cat([x1, c], dim=1))
            y1, y2 = x1, x2 - t
        return torch.cat([y1, y2], dim=1)

class CINN(nn.Module):
    def __init__(self, dim, conditional_dim, num_layers, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList([CouplingLayer(dim, conditional_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, x, c, reverse=False):
        if not reverse:
            for layer in self.layers:
                x = layer(x, c)
        else:
            for layer in reversed(self.layers):
                x = layer(x, c, reverse=True)
        return x

class AtomicReconstructionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config['position_dim'], config['hidden_dim']),
            nn.BatchNorm1d(config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.BatchNorm1d(config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['latent_dim'] * 2)
        )
        self.conditional_net = ConditionalNet(config['momenta_dim'], config['conditional_dim'], config['hidden_dim'])
        self.cinn = CINN(config['latent_dim'], config['conditional_dim'], config['num_coupling_layers'], config['hidden_dim'])
        self.decoder = nn.Linear(config['latent_dim'], config['position_dim'])

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z_transformed = self.cinn(z, c, reverse=True)
        return self.decoder(z_transformed)

    def forward(self, x, y):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        c = self.conditional_net(y)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
    momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

    scaler_position = StandardScaler()
    scaler_momenta = StandardScaler()

    position_scaled = scaler_position.fit_transform(position)
    momenta_scaled = scaler_momenta.fit_transform(momenta)

    return torch.FloatTensor(position_scaled), torch.FloatTensor(momenta_scaled)

def train(model, train_loader, optimizer, kl_weight):
    model.train()
    train_loss = 0
    latent_representations = []
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x, y)
        
        z = model.module.reparameterize(mu, logvar)
        latent_representations.append(z.detach().cpu())
        
        recon_loss = nn.MSELoss()(x_recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_weight * kl_loss

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    latent_representations = torch.cat(latent_representations, dim=0)
    return train_loss / len(train_loader), latent_representations

def evaluate(model, test_loader, latent_representations):
    model.eval()
    test_mse = 0
    mse_loss = nn.MSELoss()
    latent_idx = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            batch_size = x.size(0)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            z = latent_representations[latent_idx:latent_idx+batch_size].to(device, non_blocking=True)
            latent_idx += batch_size
            
            c = model.module.conditional_net(y)
            x_recon = model.module.decode(z, c)
            
            test_mse += mse_loss(x_recon, x).item()
    
    return test_mse / len(test_loader)

def cross_validate(config):
    X, y = load_and_preprocess_data(config['data_file'])

    kf = KFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
    
    fold_mses = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        model = AtomicReconstructionModel(config).to(device)
        model = nn.DataParallel(model)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        for epoch in range(config['num_epochs']):
            train_loss, latent_representations = train(model, train_loader, optimizer, config['kl_weight'])

        val_mse = evaluate(model, val_loader, latent_representations)
        fold_mses.append(val_mse)
    
    avg_mse = np.mean(fold_mses)
    return avg_mse

def objective(trial):
    config = {
        'latent_dim': trial.suggest_categorical('latent_dim', [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
        'conditional_dim': trial.suggest_categorical('conditional_dim', [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
        'num_coupling_layers': trial.suggest_categorical('num_coupling_layers', [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'kl_weight': trial.suggest_float('kl_weight', 1e-4, 1, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    }

    avg_mse = cross_validate({**CONFIG, **config})
    return avg_mse

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print(f"Best configuration: {study.best_params}")
    print(f"Best average MSE: {study.best_value:.4f}")