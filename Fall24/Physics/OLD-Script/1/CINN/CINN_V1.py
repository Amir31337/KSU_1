'''
TRAIN SCRIPT
'''


import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration and hyperparameters
config = {
    'batch_size': 2048,
    'epochs': 20,
    'learning_rate': 1e-3,
    'train_ratio': 0.7,
    'validation_ratio': 0.15,
    'test_ratio': 0.15,
    'early_stopping_patience': 15,
    'early_stopping_delta': 0.0001,
    'data_file_path': '/home/g/ghanaatian/MYFILES/FALL24/Physics/1M.csv',
    'input_size': 9,
    'condition_size': 9,
    'hidden_layer_size': 256,
    'num_layers': 1,
    'output_size': 9,
}

# Define the CINN model architecture with conditioning
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

    # Inverse transform the scaled data
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
    
    '''
    HERE
    '''
    # Calculate the condition as abs(position)
    conditions = np.abs(features)
    
    '''
    HERE
    '''
    
    return features, targets, conditions

# The rest of your code remains the same

def preprocess_data(features, targets, conditions):
    scaler_features = StandardScaler()
    scaler_targets = StandardScaler()
    scaler_conditions = StandardScaler()
    
    features_scaled = scaler_features.fit_transform(features)
    targets_scaled = scaler_targets.fit_transform(targets)
    conditions_scaled = scaler_conditions.fit_transform(conditions)
    
    return features_scaled, targets_scaled, conditions_scaled, scaler_features, scaler_targets, scaler_conditions

# Early stopping class
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

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # Initialize model, loss, optimizer, and early stopping
    model = CINN(config['input_size'], config['condition_size'], config['hidden_layer_size'], config['num_layers'], config['output_size']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
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
    torch.save(model.state_dict(), 'CINN_V1.pth')

    # Test the model
    model.load_state_dict(torch.load('CINN_V1.pth'))
    mse, mae, r2 = test(model, device, test_loader, scaler_targets)
    print(f"Test Results - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    main()