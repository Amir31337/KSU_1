import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

# Configuration
config = {
    'data_file_path': '/home/g/ghanaatian/MYFILES/FALL24/Physics/1M.csv',
    'input_size': 9,
    'condition_size': 1,
    'output_size': 9,
    'train_ratio': 0.7,
    'validation_ratio': 0.15,
    'test_ratio': 0.15,
    'early_stopping_patience': 15,
    'early_stopping_delta': 0.0001,
}

class CINN(nn.Module):
    def __init__(self, input_size, condition_size, hidden_size, num_layers, output_size):
        super(CINN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size + condition_size, hidden_size),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(num_layers - 1)],
            nn.Linear(hidden_size, output_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(output_size + condition_size, hidden_size),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(num_layers - 1)],
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, c):
        latent = self.encoder(torch.cat((x, c), dim=1))
        output = self.decoder(torch.cat((latent, c), dim=1))
        return output

def train(model, device, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for data, condition, target in train_loader:
        data, condition, target = data.to(device), condition.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, condition)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, condition, target in val_loader:
            data, condition, target = data.to(device), condition.to(device), target.to(device)
            output = model(data, condition)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def test(model, device, test_loader, scaler_targets):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, condition, target in test_loader:
            data, condition, target = data.to(device), condition.to(device), target.to(device)
            output = model(data, condition)
            output = output.cpu()
            all_predictions.append(output.numpy())
            all_targets.append(target.cpu().numpy())

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    predictions = scaler_targets.inverse_transform(predictions)
    targets = scaler_targets.inverse_transform(targets)

    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return mse, mae, r2

def load_data(filepath):
    data = pd.read_csv(filepath)
    features = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values
    targets = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
    conditions = features + 1e-9
    conditions = conditions.sum(axis=1, keepdims=True)
    return features, targets, conditions

def preprocess_data(features, targets, conditions):
    scaler_features = StandardScaler()
    scaler_targets = StandardScaler()
    scaler_conditions = StandardScaler()
    
    features_scaled = scaler_features.fit_transform(features)
    targets_scaled = scaler_targets.fit_transform(targets)
    conditions_scaled = scaler_conditions.fit_transform(conditions)
    
    return features_scaled, targets_scaled, conditions_scaled, scaler_features, scaler_targets, scaler_conditions

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
    # Hyperparameters to optimize
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    hidden_layer_size = trial.suggest_categorical('hidden_layer_size', [16, 32, 64, 128, 256, 512, 1024])
    num_layers = trial.suggest_int('num_layers', 1, 100)

    # Load and preprocess data
    features, targets, conditions = load_data(config['data_file_path'])
    features_scaled, targets_scaled, conditions_scaled, scaler_features, scaler_targets, scaler_conditions = preprocess_data(features, targets, conditions)

    # Convert to tensors
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_scaled, dtype=torch.float32)
    conditions_tensor = torch.tensor(conditions_scaled, dtype=torch.float32)

    # Prepare DataLoader
    dataset = TensorDataset(features_tensor, conditions_tensor, targets_tensor)
    train_size = int(len(dataset) * config['train_ratio'])
    val_size = int(len(dataset) * config['validation_ratio'])
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model, loss, optimizer, and early stopping
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CINN(config['input_size'], config['condition_size'], hidden_layer_size, num_layers, config['output_size']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    early_stopping = EarlyStopping(config['early_stopping_patience'], config['early_stopping_delta'])

    # Training loop
    for epoch in range(200):  # Set to 200 epochs as specified
        train_loss = train(model, device, train_loader, criterion, optimizer)
        val_loss = validate(model, device, val_loader, criterion)
        scheduler.step(val_loss)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    # Test the model
    mse, _, _ = test(model, device, test_loader, scaler_targets)
    return mse

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)  # You can adjust the number of trials

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Train the final model with the best hyperparameters
    best_params = study.best_params
    features, targets, conditions = load_data(config['data_file_path'])
    features_scaled, targets_scaled, conditions_scaled, scaler_features, scaler_targets, scaler_conditions = preprocess_data(features, targets, conditions)

    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_scaled, dtype=torch.float32)
    conditions_tensor = torch.tensor(conditions_scaled, dtype=torch.float32)

    dataset = TensorDataset(features_tensor, conditions_tensor, targets_tensor)
    train_size = int(len(dataset) * config['train_ratio'])
    val_size = int(len(dataset) * config['validation_ratio'])
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CINN(config['input_size'], config['condition_size'], best_params['hidden_layer_size'], best_params['num_layers'], config['output_size']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    early_stopping = EarlyStopping(config['early_stopping_patience'], config['early_stopping_delta'])

    for epoch in range(200):
        train_loss = train(model, device, train_loader, criterion, optimizer)
        val_loss = validate(model, device, val_loader, criterion)
        scheduler.step(val_loss)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    # Save the best model
    torch.save(model.state_dict(), 'best_cinn_model.pth')

    # Test the best model
    mse, mae, r2 = test(model, device, test_loader, scaler_targets)
    print(f"Final Test Results - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    main()