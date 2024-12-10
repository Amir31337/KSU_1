import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configuration variables
file_path = 'Physics/1M.csv'
batch_size = 2048
epochs = 100
lr = 0.001
patience = 16
delta = 1e-6
hidden_dim = 128
best_model_path = 'VDM_V2_BEST.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load and preprocess the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    initial_positions = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']]
    final_momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']]

    scaler_positions = MinMaxScaler()
    scaler_momenta = MinMaxScaler()

    initial_positions_scaled = scaler_positions.fit_transform(initial_positions)
    final_momenta_scaled = scaler_momenta.fit_transform(final_momenta)

    return initial_positions_scaled, final_momenta_scaled, scaler_positions, scaler_momenta

# Neural network model to predict initial positions from final momenta
class VDM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VDM, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Output is predicted initial positions
        return x

# Loss function to directly minimize error between predicted and actual initial positions
def position_loss(predicted_positions, true_positions):
    return nn.functional.mse_loss(predicted_positions, true_positions)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
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

# Train the model
def train_model(model, train_loader, val_loader, epochs, lr, patience, delta, best_model_path):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    early_stopping = EarlyStopping(patience=patience, delta=delta)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            final_momenta, initial_positions = batch
            final_momenta, initial_positions = final_momenta.to(device), initial_positions.to(device)

            optimizer.zero_grad()
            predicted_positions = model(final_momenta)
            loss = position_loss(predicted_positions, initial_positions)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_momenta, val_positions = val_batch
                val_momenta, val_positions = val_momenta.to(device), val_positions.to(device)
                predicted_positions = model(val_momenta)
                val_loss = position_loss(predicted_positions, val_positions)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch + 1} with validation loss {avg_val_loss:.6f}")

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return train_losses, val_losses

# Plot learning curve
def plot_learning_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curve')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('VDM_V2_Learning_Curve.png')
    plt.close()

# Evaluate model on test set
def evaluate_model(model, test_loader, scaler_positions):
    model.eval()
    true_positions = []
    pred_positions = []
    with torch.no_grad():
        for test_batch in test_loader:
            test_momenta, test_positions = test_batch
            test_momenta, test_positions = test_momenta.to(device), test_positions.to(device)
            predicted_positions = model(test_momenta)
            pred_positions.append(scaler_positions.inverse_transform(predicted_positions.cpu().numpy()))
            true_positions.append(scaler_positions.inverse_transform(test_positions.cpu().numpy()))

    true_positions = np.concatenate(true_positions)
    pred_positions = np.concatenate(pred_positions)

    mse = mean_squared_error(true_positions, pred_positions)
    mae = mean_absolute_error(true_positions, pred_positions)
    r2 = r2_score(true_positions, pred_positions)

    return mse, mae, r2

# Main function to run the training
def main():
    initial_positions, final_momenta, scaler_positions, scaler_momenta = load_data(file_path)
    input_dim = final_momenta.shape[1]
    output_dim = initial_positions.shape[1]

    dataset = TensorDataset(torch.tensor(final_momenta, dtype=torch.float32),
                            torch.tensor(initial_positions, dtype=torch.float32))

    # Split dataset into train, validation, and test sets
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = VDM(input_dim, hidden_dim, output_dim).to(device)
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs, lr, patience, delta, best_model_path)

    plot_learning_curve(train_losses, val_losses)

    # Load the best model and evaluate on the test set
    best_model = VDM(input_dim, hidden_dim, output_dim).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    mse, mae, r2 = evaluate_model(best_model, test_loader, scaler_positions)

    print(f'Test MSE: {mse:.6f}')
    print(f'Test MAE: {mae:.6f}')
    print(f'Test R2: {r2:.6f}')

    # Save both scalers
    joblib.dump(scaler_positions, 'scaler_positions.joblib')
    joblib.dump(scaler_momenta, 'scaler_momenta.joblib')
    print("Scalers saved successfully.")

if __name__ == '__main__':
    main()
