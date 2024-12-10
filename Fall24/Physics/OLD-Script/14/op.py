import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
from itertools import product
import datetime
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_layers, activation_function):
        super(CVAE, self).__init__()
        
        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_layers[0]))
        encoder_layers.append(activation_function)
        for i in range(len(hidden_layers) - 1):
            encoder_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            encoder_layers.append(activation_function)
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(hidden_layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim + condition_dim, hidden_layers[-1]))
        decoder_layers.append(activation_function)
        for i in reversed(range(len(hidden_layers) - 1)):
            decoder_layers.append(nn.Linear(hidden_layers[i+1], hidden_layers[i]))
            decoder_layers.append(activation_function)
        decoder_layers.append(nn.Linear(hidden_layers[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        combined = torch.cat((z, condition), dim=1)
        return self.decoder(combined)

    def forward(self, x, condition):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, condition), mu, logvar

class CVAETrainer:
    def __init__(self, params, input_dim, output_dim, device):
        self.params = params
        self.device = device
        hidden_layers = [params['hidden_layer_size'] * (2 ** i) for i in range(params['num_hidden_layers'])]
        activation_function = getattr(nn, params['activation_name'])()
        self.model = CVAE(input_dim, params['LATENT_DIM'], output_dim, hidden_layers, activation_function).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params['LEARNING_RATE'])

    def compute_metrics(self, loader, position_data, position_scaler, name):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x, batch_cond in loader:
                recon_x, _, _ = self.model(batch_x, batch_cond)
                predictions.append(recon_x)
        
        predictions = torch.cat(predictions, dim=0)
        
        # Convert tensor to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(position_data, torch.Tensor):
            position_data = position_data.cpu().numpy()
        
        if position_scaler is not None:
            predictions_inv = position_scaler.inverse_transform(predictions)
            targets_inv = position_scaler.inverse_transform(position_data)
        else:
            predictions_inv = predictions
            targets_inv = position_data
            
        relative_errors = np.abs(predictions_inv - targets_inv) / (np.abs(targets_inv) + 1e-8)
        mre = np.mean(relative_errors)
        mse = np.mean((predictions_inv - targets_inv) ** 2)
        
        return {f"{name}_mre": float(mre), f"{name}_mse": float(mse)}

    def loss_fn(self, recon_x, x, mu, logvar, return_components=False):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_divergence /= x.size(0) * x.size(1)
        loss = recon_loss + kl_divergence
        
        if self.params['use_l1']:
            l1_loss = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
            loss += self.params['L1_LAMBDA'] * l1_loss
            
        if self.params['use_l2']:
            l2_loss = sum(torch.sum(param ** 2) for param in self.model.parameters())
            loss += self.params['L2_LAMBDA'] * l2_loss
            
        if return_components:
            return loss, recon_loss, kl_divergence
        return loss

    def train_and_evaluate(self, train_loader, val_loader, test_loader, 
                          train_position, val_position, test_position, position_scaler):
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.params['EPOCHS']):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_cond in train_loader:
                self.optimizer.zero_grad()
                recon_x, mu, logvar = self.model(batch_x, batch_cond)
                loss = self.loss_fn(recon_x, batch_x, mu, logvar)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                train_loss += loss.item()
                
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_cond in val_loader:
                    recon_x, mu, logvar = self.model(batch_x, batch_cond)
                    loss = self.loss_fn(recon_x, batch_x, mu, logvar)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.params['EPOCHS']} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss - self.params['MIN_DELTA']:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.params['PATIENCE']:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Evaluation
        metrics = {}
        metrics.update(self.compute_metrics(train_loader, train_position, position_scaler, "train"))
        metrics.update(self.compute_metrics(val_loader, val_position, position_scaler, "val"))
        metrics.update(self.compute_metrics(test_loader, test_position, position_scaler, "test"))
        
        return metrics, train_losses, val_losses

def prepare_data(data, scaler_method):
    """
    Prepare data with optional normalization
    """
    if scaler_method == "StandardScaler":
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
    elif scaler_method == "None":
        scaler = None
        data_normalized = data
    return torch.FloatTensor(data_normalized), scaler

def prepare_data(data, scaler_method):
    """
    Prepare data with optional normalization
    """
    if scaler_method == "StandardScaler":
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
    elif scaler_method == "None":
        scaler = None
        data_normalized = data
    return torch.FloatTensor(data_normalized), scaler

def main():
    # Modified hyperparameter combinations
    param_grid = {
        "LATENT_DIM": [512, 1024, 2048],
        "EPOCHS": [100, 200, 300],
        "BATCH_SIZE": [256, 512, 1024],
        "LEARNING_RATE": [0.01, 0.001, 0.0001, 0.00001],
        "PATIENCE": [100],
        "MIN_DELTA": [1e-5],
        "activation_name": ["Sigmoid", "ReLU", "Tanh", "ELU"],
        "position_norm_method": ["StandardScaler", "None"],  # Added None option
        "momenta_norm_method": ["StandardScaler", "None"],   # Added None option
        "use_l1": [True, False],
        "L1_LAMBDA": [0.001, 0.01, 0.1, 0.5],
        "use_l2": [True, False],
        "L2_LAMBDA": [0.001, 0.01, 0.1, 0.5],
        "num_hidden_layers": [1, 2, 3, 4],
        "hidden_layer_size": [64, 128, 256]
    }

    # Generate all possible combinations and shuffle them
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    random.shuffle(combinations)
    
    # Load and prepare data
    FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
    data = pd.read_csv(FILEPATH)
    position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
    momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values
    
    # Split data
    train_position, temp_position, train_momenta, temp_momenta = train_test_split(
        position, momenta, test_size=0.3, random_state=85, shuffle=True
    )
    val_position, test_position, val_momenta, test_momenta = train_test_split(
        temp_position, temp_momenta, test_size=0.5, random_state=42, shuffle=True
    )
    
    # Create output directory and save parameter configurations (unchanged)
    BASE_OUTPUT_DIR = '/home/g/ghanaatian/MYFILES/FALL24/Physics/14'
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f'grid_search_results_{timestamp}')
    
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Created output directory: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error creating directory: {str(e)}")
        raise
    
    all_results = []
    
    # Save initial combination order and parameter grid
    with open(os.path.join(OUTPUT_DIR, 'combination_order.json'), 'w') as f:
        json.dump([{f"combination_{i+1}": comb} for i, comb in enumerate(combinations)], f, indent=4)
    
    with open(os.path.join(OUTPUT_DIR, 'parameter_grid.json'), 'w') as f:
        json.dump(param_grid, f, indent=4)
    
    # Iterate through shuffled combinations
    for i, params in enumerate(combinations):
        print(f"\nTraining combination {i+1}/{len(combinations)}")
        print(json.dumps(params, indent=2))
        
        try:
            # Prepare position data with optional normalization
            train_position_norm, position_scaler = prepare_data(train_position, params['position_norm_method'])
            val_position_norm = (torch.FloatTensor(position_scaler.transform(val_position)) 
                               if position_scaler else torch.FloatTensor(val_position))
            test_position_norm = (torch.FloatTensor(position_scaler.transform(test_position))
                                if position_scaler else torch.FloatTensor(test_position))
            
            # Prepare momenta data with optional normalization
            train_momenta_norm, momenta_scaler = prepare_data(train_momenta, params['momenta_norm_method'])
            val_momenta_norm = (torch.FloatTensor(momenta_scaler.transform(val_momenta))
                              if momenta_scaler else torch.FloatTensor(val_momenta))
            test_momenta_norm = (torch.FloatTensor(momenta_scaler.transform(test_momenta))
                               if momenta_scaler else torch.FloatTensor(test_momenta))
            
            # Move data to device
            train_position_norm = train_position_norm.to(device)
            val_position_norm = val_position_norm.to(device)
            test_position_norm = test_position_norm.to(device)
            train_momenta_norm = train_momenta_norm.to(device)
            val_momenta_norm = val_momenta_norm.to(device)
            test_momenta_norm = test_momenta_norm.to(device)
            
            # Create data loaders
            train_loader = DataLoader(TensorDataset(train_position_norm, train_momenta_norm), 
                                    batch_size=params['BATCH_SIZE'], shuffle=True)
            val_loader = DataLoader(TensorDataset(val_position_norm, val_momenta_norm), 
                                  batch_size=params['BATCH_SIZE'], shuffle=False)
            test_loader = DataLoader(TensorDataset(test_position_norm, test_momenta_norm), 
                                   batch_size=params['BATCH_SIZE'], shuffle=False)
            
            # Train and evaluate
            trainer = CVAETrainer(params, position.shape[1], momenta.shape[1], device)
            metrics, train_losses, val_losses = trainer.train_and_evaluate(
                train_loader, val_loader, test_loader,
                train_position, val_position, test_position,
                position_scaler
            )
            
            
            # Save results
            result = {
                'combination_number': i + 1,
                'params': params,
                'metrics': metrics,
                'train_losses': [float(x) for x in train_losses],
                'val_losses': [float(x) for x in val_losses]
            }
            
            # Save individual result
            filename = f'result_{i+1}.json'
            with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
                json.dump(result, f, indent=4)
            
            all_results.append(result)
            
            # Plot learning curves
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Learning Curves - Combination {i+1}')
            plt.legend()
            plt.savefig(os.path.join(OUTPUT_DIR, f'learning_curves_{i+1}.png'))
            plt.close()
            
            # Save progress checkpoint
            with open(os.path.join(OUTPUT_DIR, 'progress_results.json'), 'w') as f:
                json.dump(all_results, f, indent=4)
            
        except Exception as e:
            error_msg = f"Error in combination {i+1}: {str(e)}"
            print(error_msg)
            
            # Save error log
            with open(os.path.join(OUTPUT_DIR, 'error_log.txt'), 'a') as f:
                f.write(f"\n{datetime.datetime.now()}: {error_msg}")
            continue
    
    # Save all results
    with open(os.path.join(OUTPUT_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Find and save best combination
    best_result = min(all_results, key=lambda x: x['metrics']['val_mse'])
    with open(os.path.join(OUTPUT_DIR, 'best_result.json'), 'w') as f:
        json.dump(best_result, f, indent=4)
    
    print("\nGrid search completed!")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("\nBest combination:")
    print(json.dumps(best_result['params'], indent=2))
    print("\nBest metrics:")
    print(json.dumps(best_result['metrics'], indent=2))

if __name__ == "__main__":
    main()