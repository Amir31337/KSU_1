import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import optuna

# Configuration and hyperparameters
config = {
    'epochs': 50,
    'train_ratio': 0.7,
    'validation_ratio': 0.15,
    'test_ratio': 0.15,
    'early_stopping_patience': 10,
    'early_stopping_delta': 0.0001,
    'data_file_path': '/home/g/ghanaatian/MYFILES/FALL24/Physics/10K.csv',
    'input_size': 9,
    'output_size': 9,
    'n_trials': 100,  # Number of trials for Optuna
}

class INN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(INN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output

def train(model, device, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def load_data(filepath):
    data = pd.read_csv(filepath)
    features = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values
    targets = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
    return features, targets

def preprocess_data(features, targets):
    scaler_features = StandardScaler()
    scaler_targets = StandardScaler()
    features_scaled = scaler_features.fit_transform(features)
    targets_scaled = scaler_targets.fit_transform(targets)
    return features_scaled, targets_scaled, scaler_features, scaler_targets

class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def objective(trial):
    # Suggest values for the hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    hidden_layer_size = trial.suggest_categorical('hidden_layer_size', [16, 32, 64, 128, 256, 512, 1024])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    features, targets = load_data(config['data_file_path'])
    features_scaled, targets_scaled, _, _ = preprocess_data(features, targets)

    # Convert to tensors
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    targets_tensor = torch.tensor(targets_scaled, dtype=torch.float32).to(device)

    # Prepare DataLoader
    dataset = TensorDataset(features_tensor, targets_tensor)
    train_size = int(len(dataset) * config['train_ratio'])
    val_size = int(len(dataset) * config['validation_ratio'])
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss, optimizer, and early stopping
    model = INN(config['input_size'], hidden_layer_size, config['output_size']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(config['early_stopping_patience'], config['early_stopping_delta'])

    # Training loop
    for epoch in range(config['epochs']):
        train_loss = train(model, device, train_loader, criterion, optimizer)
        val_loss = validate(model, device, val_loader, criterion)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    return val_loss

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=config['n_trials'])

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Train the final model with the best parameters
    best_params = study.best_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    features, targets = load_data(config['data_file_path'])
    features_scaled, targets_scaled, scaler_features, scaler_targets = preprocess_data(features, targets)

    # Convert to tensors
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    targets_tensor = torch.tensor(targets_scaled, dtype=torch.float32).to(device)

    # Prepare DataLoader
    dataset = TensorDataset(features_tensor, targets_tensor)
    train_size = int(len(dataset) * config['train_ratio'])
    val_size = int(len(dataset) * config['validation_ratio'])
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])

    # Initialize model with best parameters
    model = INN(config['input_size'], best_params['hidden_layer_size'], config['output_size']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])

    # Train the final model
    for epoch in range(config['epochs']):
        train_loss = train(model, device, train_loader, criterion, optimizer)
        val_loss = validate(model, device, val_loader, criterion)
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Test the model
    model.eval()
    test_loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    test_loss /= len(test_loader)
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    # Inverse transform the scaled data
    predictions = scaler_targets.inverse_transform(predictions)
    targets = scaler_targets.inverse_transform(targets)

    mse = mean_squared_error(targets, predictions)
    print(f"Test MSE: {mse:.4f}")

if __name__ == "__main__":
    main()