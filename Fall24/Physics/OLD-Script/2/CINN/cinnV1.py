import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Configuration and hyperparameters
config = {
    'batch_size': 512,
    'epochs': 100,
    'learning_rate': 1e-4,
    'train_ratio': 0.7,
    'validation_ratio': 0.15,
    'test_ratio': 0.15,
    'early_stopping_patience': 10,
    'early_stopping_delta': 0.001,
    'data_file_path': '/home/g/ghanaatian/Documents/PyCharm/pythonProject/FALL24/Physics/10K.csv',
    'input_size': 9,
    'hidden_layer_size': 1024,  # Increased width
    'output_size': 9,
    'condition_size': 9,
    'noise_factor': 0.01  # Noise factor for augmentation
}


# Custom Loss Function: Log-Cosh Loss
class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(loss)


# Residual block to be used in deeper networks
class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out + residual  # Skip connection


class cINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, condition_size):
        super(cINN, self).__init__()

        # Calculate the combined input size based on feature engineering
        # input_size = 9 (original features), 9 (squared), 9 (sqrt), so 27 + 9 (condition) = 36
        combined_input_size = (input_size * 3) + condition_size  # Adjusted for feature engineering

        self.encoder = nn.Sequential(
            nn.Linear(combined_input_size, hidden_size),  # Adjusted input size
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            ResidualBlock(hidden_size, hidden_size // 2),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, condition):
        x_combined = torch.cat((x, condition), dim=1)
        return self.encoder(x_combined)


def train(model, device, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for data, target, condition in train_loader:
        data, target, condition = data.to(device), target.to(device), condition.to(device)

        # Adding noise for data augmentation
        noise = torch.randn_like(data) * config['noise_factor']
        data += noise.to(device)

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
        for data, target, condition in val_loader:
            data, target, condition = data.to(device), target.to(device), condition.to(device)
            output = model(data, condition)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def test(model, device, test_loader, scaler_targets):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target, condition in test_loader:
            data, condition = data.to(device), condition.to(device)
            output = model(data, condition)
            output = output.cpu()
            all_predictions.append(output.numpy())
            all_targets.append(target.numpy())

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

    # Feature interactions
    features = np.hstack([features, features ** 2, np.sqrt(np.abs(features))])

    return features, targets


def preprocess_data(features, targets):
    scaler_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()
    scaler_condition = MinMaxScaler()

    features_scaled = scaler_features.fit_transform(features)
    targets_scaled = scaler_targets.fit_transform(targets)

    condition = np.log1p(features_scaled)  # Changed condition definition to add complexity
    condition_scaled = scaler_condition.fit_transform(condition)

    return features_scaled, targets_scaled, condition_scaled, scaler_features, scaler_targets, scaler_condition


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    features, targets = load_data(config['data_file_path'])
    features_scaled, targets_scaled, condition_scaled, scaler_features, scaler_targets, scaler_condition = preprocess_data(
        features, targets)

    # Convert to tensors
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_scaled, dtype=torch.float32)
    condition_tensor = torch.tensor(condition_scaled, dtype=torch.float32)

    # Prepare DataLoader
    dataset = TensorDataset(features_tensor, targets_tensor, condition_tensor)
    train_size = int(len(dataset) * config['train_ratio'])
    val_size = int(len(dataset) * config['validation_ratio'])
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # Initialize model, loss, optimizer, and early stopping
    model = cINN(config['input_size'], config['hidden_layer_size'], config['output_size'], config['condition_size']).to(
        device)
    criterion = LogCoshLoss()  # Using Log-Cosh Loss
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])  # Using AdamW optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    early_stopping = EarlyStopping(config['early_stopping_patience'], config['early_stopping_delta'])

    # Training loop
    for epoch in range(config['epochs']):
        train_loss = train(model, device, train_loader, criterion, optimizer)
        val_loss = validate(model, device, val_loader, criterion)
        scheduler.step(val_loss)
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Save the model
    torch.save(model.state_dict(), 'cinn_model.pth')

    # Test the model
    model.load_state_dict(torch.load('cinn_model.pth', map_location=device))
    mse, mae, r2 = test(model, device, test_loader, scaler_targets)
    print(f"Test Results - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


if __name__ == "__main__":
    main()
