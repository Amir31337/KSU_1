'''
TEST SCRIPT
'''
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random

# Configuration and hyperparameters
config = {
    'MODEL_PATH': '/home/g/ghanaatian/MYFILES/FALL24/Physics/NEW/CINN/CINN_V1.pth',
    'data_file_path': '/home/g/ghanaatian/MYFILES/FALL24/Physics/generated_cos3d_check.csv',
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

    return mse, mae, r2, predictions, targets

def load_data(filepath):
    data = pd.read_csv(filepath)
    features = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values
    targets = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
    
    # Calculate the condition as abs(position)
    conditions = np.abs(features)
    
    return features, targets, conditions

def preprocess_data(features, targets, conditions):
    scaler_features = StandardScaler()
    scaler_targets = StandardScaler()
    scaler_conditions = StandardScaler()
    
    features_scaled = scaler_features.fit_transform(features)
    targets_scaled = scaler_targets.fit_transform(targets)
    conditions_scaled = scaler_conditions.fit_transform(conditions)
    
    return features_scaled, targets_scaled, conditions_scaled, scaler_features, scaler_targets, scaler_conditions

def print_random_row(features, targets, predictions):
    num_rows = len(features)
    random_index = random.randint(0, num_rows - 1)
    
    print(f"Random Row Index: {random_index}")
    print("Momenta (pcx, pcy, pcz, pox, poy, poz, psx, psy, psz):")
    print(np.array2string(features[random_index], formatter={'float_kind':lambda x: "%.2f" % x}))
    print("\nReal Position (cx, cy, cz, ox, oy, oz, sx, sy, sz):")
    print(np.array2string(targets[random_index], formatter={'float_kind':lambda x: "%.2f" % x}))
    print("\nPredicted Position (cx, cy, cz, ox, oy, oz, sx, sy, sz):")
    print(np.array2string(predictions[random_index], formatter={'float_kind':lambda x: "%.2f" % x}))

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
    test_loader = DataLoader(dataset)

    # Initialize model, loss, optimizer, and early stopping
    model = CINN(config['input_size'], config['condition_size'], config['hidden_layer_size'], config['num_layers'], config['output_size']).to(device)
    # Test the model
    model.load_state_dict(torch.load(config['MODEL_PATH']))
    mse, mae, r2, predictions, targets = test(model, device, test_loader, scaler_targets)
    print(f"Test Results - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    # Print a random row
    print_random_row(features, targets, predictions)

if __name__ == "__main__":
    main()