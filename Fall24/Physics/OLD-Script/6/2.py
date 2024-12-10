'''######################
Test MSE: 3.9668
#####################'''

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, momenta_dim, hidden_dims):
        super(Autoencoder, self).__init__()
        self.encoder = self._build_encoder(input_dim, latent_dim, hidden_dims)
        self.decoder = self._build_decoder(latent_dim, momenta_dim, input_dim, hidden_dims[::-1])

    def _build_encoder(self, input_dim, latent_dim, hidden_dims):
        layers = []
        in_features = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, latent_dim * 2))
        return nn.Sequential(*layers)

    def _build_decoder(self, latent_dim, momenta_dim, output_dim, hidden_dims):
        layers = []
        in_features = latent_dim + momenta_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))
        return nn.Sequential(*layers)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        latent_params = self.encoder(x)
        mu, log_var = latent_params.chunk(2, dim=-1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(torch.cat((z, y), dim=-1))
        return x_recon, mu, log_var

class CINN(nn.Module):
    def __init__(self, input_dim, condition_dim, output_dim, num_layers):
        super(CINN, self).__init__()
        self.layers = nn.ModuleList([CouplingLayer(input_dim, condition_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, z, y):
        log_det_jacobian = 0
        for layer in self.layers:
            z, ldj = layer(z, y)
            log_det_jacobian += ldj
        return z, log_det_jacobian

    def inverse(self, x, y):
        log_det_jacobian = 0
        for layer in reversed(self.layers):
            x, ldj = layer.inverse(x, y)
            log_det_jacobian += ldj
        x = self.output_layer(x)
        return x, log_det_jacobian

class CouplingLayer(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(CouplingLayer, self).__init__()
        self.split_dim = input_dim // 2
        self.scale_net = self._build_net(self.split_dim + condition_dim, self.split_dim)
        self.translate_net = self._build_net(self.split_dim + condition_dim, self.split_dim)
        self.init_weights()

    def _build_net(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def init_weights(self):
        for layer in self.scale_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.translate_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, z, y):
        z1, z2 = z.chunk(2, dim=-1)
        scale = torch.clamp(self.scale_net(torch.cat((z1, y), dim=-1)), min=-5, max=5)
        translate = self.translate_net(torch.cat((z1, y), dim=-1))
        z2 = z2 * torch.exp(scale) + translate
        z = torch.cat((z1, z2), dim=-1)
        log_det_jacobian = scale.sum(dim=-1)
        return z, log_det_jacobian

    def inverse(self, x, y):
        x1, x2 = x.chunk(2, dim=-1)
        scale = torch.clamp(self.scale_net(torch.cat((x1, y), dim=-1)), min=-5, max=5)
        translate = self.translate_net(torch.cat((x1, y), dim=-1))
        x2 = (x2 - translate) * torch.exp(-scale)
        x = torch.cat((x1, x2), dim=-1)
        log_det_jacobian = -scale.sum(dim=-1)
        return x, log_det_jacobian

def load_data(file_path):
    data = pd.read_csv(file_path)
    positions = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
    momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values
    return positions, momenta

def preprocess_data(positions, momenta, train_ratio=0.8):
    num_samples = positions.shape[0]
    train_size = int(num_samples * train_ratio)
    device = get_device()

    positions_train = positions[:train_size]
    momenta_train = momenta[:train_size]
    positions_test = positions[train_size:]
    momenta_test = momenta[train_size:]

    positions_train_tensor = torch.tensor(positions_train, dtype=torch.float32).to(device)
    momenta_train_tensor = torch.tensor(momenta_train, dtype=torch.float32).to(device)
    positions_test_tensor = torch.tensor(positions_test, dtype=torch.float32).to(device)
    momenta_test_tensor = torch.tensor(momenta_test, dtype=torch.float32).to(device)

    return positions_train_tensor, momenta_train_tensor, positions_test_tensor, momenta_test_tensor

def train(autoencoder, cinn, train_loader, optimizer_ae, optimizer_cinn, epochs, grad_clip):
    device = get_device()
    for epoch in range(epochs):
        autoencoder_loss_sum = 0
        cinn_loss_sum = 0
        num_batches = 0

        for batch_idx, (positions, momenta) in enumerate(train_loader):
            positions, momenta = positions.to(device), momenta.to(device)

            recon_positions, mu, log_var = autoencoder(positions, momenta)
            recon_loss = nn.MSELoss()(recon_positions, positions)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / positions.size(0)
            autoencoder_loss = recon_loss + kld_loss
            
            optimizer_ae.zero_grad()
            autoencoder_loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), grad_clip)
            optimizer_ae.step()

            z = autoencoder.reparameterize(mu.detach(), log_var.detach())
            z = z.detach()
            z, log_det_jacobian = cinn(z, momenta)
            cinn_loss = -torch.mean(log_det_jacobian)

            if torch.isnan(cinn_loss).any():
                print(f"NaN detected in CINN loss at epoch {epoch}, batch {batch_idx}")
                continue

            optimizer_cinn.zero_grad()
            cinn_loss.backward()
            torch.nn.utils.clip_grad_norm_(cinn.parameters(), grad_clip)
            optimizer_cinn.step()

            autoencoder_loss_sum += autoencoder_loss.item()
            cinn_loss_sum += cinn_loss.item()
            num_batches += 1

        avg_autoencoder_loss = autoencoder_loss_sum / num_batches
        avg_cinn_loss = cinn_loss_sum / num_batches
        print(f"Epoch [{epoch+1}/{epochs}], Autoencoder Loss: {avg_autoencoder_loss:.4f}, CINN Loss: {avg_cinn_loss:.4f}")

def test(autoencoder, cinn, test_loader):
    device = get_device()
    mse_sum = 0
    num_batches = 0
    with torch.no_grad():
        for positions, momenta in test_loader:
            positions, momenta = positions.to(device), momenta.to(device)
            mu, _ = autoencoder.encoder(positions).chunk(2, dim=-1)
            z = mu
            predicted_positions, _ = cinn.inverse(z, momenta)
            mse = nn.MSELoss()(predicted_positions, positions)
            mse_sum += mse.item()
            num_batches += 1
    avg_mse = mse_sum / num_batches
    print(f"Test MSE: {avg_mse:.4f}")
    return avg_mse

def main(config):
    device = get_device()
    print(f"Using device: {device}")

    positions, momenta = load_data(config['data_path'])
    positions_train, momenta_train, positions_test, momenta_test = preprocess_data(positions, momenta, config['train_ratio'])

    train_dataset = TensorDataset(positions_train, momenta_train)
    test_dataset = TensorDataset(positions_test, momenta_test)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    autoencoder = Autoencoder(config['input_dim'], config['latent_dim'], config['momenta_dim'], config['hidden_dims']).to(device)
    cinn = CINN(config['latent_dim'], config['condition_dim'], config['position_dim'], config['num_layers']).to(device)

    optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=config['learning_rate'])
    optimizer_cinn = torch.optim.Adam(cinn.parameters(), lr=config['learning_rate'])

    train(autoencoder, cinn, train_loader, optimizer_ae, optimizer_cinn, config['epochs'], config['grad_clip'])

    torch.save(autoencoder.state_dict(), 'autoencoder.pth')
    autoencoder.load_state_dict(torch.load('autoencoder.pth'))
    autoencoder.eval()

    test_mse = test(autoencoder, cinn, test_loader)
    return test_mse

if __name__ == '__main__':
    config = {
        'data_path': '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv',
        'input_dim': 9,
        'position_dim': 9,
        'momenta_dim': 9,
        'latent_dim': 2,
        'condition_dim': 9,
        'hidden_dims': [165, 94],
        'num_layers': 3,
        'batch_size': 156,
        'epochs': 50,
        'learning_rate': 0.0001,
        'grad_clip': 1.2,
        'train_ratio': 0.8
    }
    main(config)