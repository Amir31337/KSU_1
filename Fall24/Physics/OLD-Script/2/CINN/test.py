import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

config = {
    'batch_size': 2048,
    'input_size': 9,
    'hidden_layer_size': 64,
    'output_size': 9,
    'condition_size': 9
}

class cINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, condition_size):
        super(cINN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size + condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size + condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x, condition):
        x = torch.cat((x, condition), dim=1)
        latent = self.encoder(x)
        latent_with_condition = torch.cat((latent, condition), dim=1)
        output = self.decoder(latent_with_condition)
        return output


def load_data(filepath):
    data = pd.read_csv(filepath)
    data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']] = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].fillna(1)
    features = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values
    targets = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
    # target = output, features = input
    return features, targets

def preprocess_data(features, targets):
    scaler_features = StandardScaler()
    scaler_targets = StandardScaler()
    features_scaled = scaler_features.fit_transform(features)
    targets_scaled = scaler_targets.fit_transform(targets)
    return features_scaled, targets_scaled, scaler_features, scaler_targets


def test(model, device, test_loader, scaler_targets):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target, condition in test_loader:
            data, target, condition = data.to(device), target.to(device), condition.to(device)
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, targets = load_data('/home/g/ghanaatian/MYFILES/FALL24/Physics/19Aug/CINN/test.csv')
    #features, targets = load_data('random_cos3d_10000.csv')
    features_scaled, targets_scaled, scaler_features, scaler_targets = preprocess_data(features, targets)

    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    targets_tensor = torch.tensor(targets_scaled, dtype=torch.float32).to(device)

    # Using targets as condition
    # target = output, features = input
    dataset = TensorDataset(features_tensor, targets_tensor, targets_tensor)
    test_loader = DataLoader(dataset, batch_size=config['batch_size'])

    model = cINN(config['input_size'], config['hidden_layer_size'], config['output_size'], config['condition_size']).to(device)

    model.load_state_dict(torch.load('/home/g/ghanaatian/MYFILES/FALL24/Physics/19Aug/CINN/cinn_model.pth', weights_only=True))
    mse, mae, r2 = test(model, device, test_loader, scaler_targets)
    print(f"Test Results - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


    test_df = pd.read_csv('/home/g/ghanaatian/MYFILES/FALL24/Physics/19Aug/CINN/test.csv')
    test_df[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']] = test_df[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].fillna(1)
    #test_df = pd.read_csv('random_cos3d_10000.csv')
    rand_row = np.random.randint(0, len(test_df))
    test_row = test_df.iloc[rand_row]
    print(f'Row number: {int(test_row.iloc[0])}')
    test_target = test_row[1:10].values
    test_features = test_row[10:].values
    print('Final momanta:\n', np.array2string(test_features, formatter={'float_kind': '{:.2f}'.format}))
    print('Real Position:\n', np.array2string(test_target, formatter={'float_kind': '{:.2f}'.format}))


    test_features_scaled = scaler_features.transform(test_features.reshape(1, -1))
    test_features_tensor = torch.tensor(test_features_scaled, dtype=torch.float32).to(device)
    test_target_scaled = scaler_targets.transform(test_target.reshape(1, -1))
    test_condition_tensor = torch.tensor(test_target_scaled, dtype=torch.float32).to(device)
    output = model(test_features_tensor, test_condition_tensor)
    output = output.cpu()
    predicted_output = scaler_targets.inverse_transform(output.detach().numpy())
    print('Predicted Position:\n', np.array2string(predicted_output, formatter={'float_kind': '{:.2f}'.format}))


if __name__ == "__main__":
    main()
