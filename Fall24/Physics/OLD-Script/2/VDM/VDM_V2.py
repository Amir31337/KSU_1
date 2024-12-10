# TRAIN AND SAVE THE BEST MODEL WITH A 1M ROWS DATASET
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import joblib

# Configuration variables
file_path = 'Physics/1M.csv'
batch_size = 2048
epochs = 100
lr = 0.0001
beta = 0.12437626128847809
T = 822
patience = 16
delta = 1.0363604392842692e-06
hidden_dim = 64
k_folds = 5
best_model_path = 'VDM_V2_BEST.pth'
device = torch.device('cpu')
#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load and preprocess the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    initial_positions = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']]
    final_momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']]

    # Use separate scalers for initial_positions and final_momenta
    scaler_positions = MinMaxScaler()
    scaler_momenta = MinMaxScaler()

    initial_positions_scaled = scaler_positions.fit_transform(initial_positions)
    final_momenta_scaled = scaler_momenta.fit_transform(final_momenta)

    return initial_positions_scaled, final_momenta_scaled, scaler_positions, scaler_momenta


# Forward diffusion process (adding noise)
def forward_diffusion_process(data, T, beta):
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha * np.ones(T))
    noise = np.random.randn(*data.shape)
    noisy_data = np.sqrt(alpha_bar[-1]) * data + np.sqrt(1 - alpha_bar[-1]) * noise
    return noisy_data, noise

# Reverse diffusion process (removing noise)
class ReverseDiffusionProcess(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ReverseDiffusionProcess, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Variational Diffusion Model
class VDM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VDM, self).__init__()
        self.reverse_diffusion = ReverseDiffusionProcess(input_dim, hidden_dim)

    def forward(self, x):
        return self.reverse_diffusion(x)

# Loss function (Variational Lower Bound)
def vlb_loss(noisy_data, noise_pred, noise):
    mse_loss = nn.functional.mse_loss(noise_pred, noise, reduction='sum')
    return mse_loss

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
def train_model(model, train_loader, val_loader, epochs, lr, beta, T, patience, delta, best_model_path):
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
            noisy_data, noise = forward_diffusion_process(final_momenta.cpu().numpy(), T, beta)
            noisy_data = torch.tensor(noisy_data, dtype=torch.float32).to(device)
            noise = torch.tensor(noise, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            noise_pred = model(noisy_data)
            loss = vlb_loss(noisy_data, noise_pred, noise)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_losses.append(total_train_loss / len(train_loader.dataset))

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_momenta, val_positions = val_batch
                val_momenta, val_positions = val_momenta.to(device), val_positions.to(device)
                noisy_data, noise = forward_diffusion_process(val_momenta.cpu().numpy(), T, beta)
                noisy_data = torch.tensor(noisy_data, dtype=torch.float32).to(device)
                noise = torch.tensor(noise, dtype=torch.float32).to(device)
                noise_pred = model(noisy_data)
                val_loss = vlb_loss(noisy_data, noise_pred, noise)
                total_val_loss += val_loss.item()

        val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.6f}, Validation Loss: {val_losses[-1]:.6f}')

        scheduler.step(val_loss)
        early_stopping(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch + 1} with validation loss {val_loss:.6f}")

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return train_losses, val_losses

# Plot learning curve
def plot_learning_curve(train_losses, val_losses, fold):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Learning Curve - Fold {fold}')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f'VDM_V2_Fold_{fold}.png')
    plt.close()

# Evaluate model on test set
def evaluate_model(model, test_loader, beta, T):
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for test_batch in test_loader:
            test_momenta, test_positions = test_batch
            test_momenta, test_positions = test_momenta.to(device), test_positions.to(device)
            noisy_data, noise = forward_diffusion_process(test_momenta.cpu().numpy(), T, beta)
            noisy_data = torch.tensor(noisy_data, dtype=torch.float32).to(device)
            noise = torch.tensor(noise, dtype=torch.float32).to(device)
            noise_pred = model(noisy_data)
            test_loss = vlb_loss(noisy_data, noise_pred, noise)
            total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_loader.dataset)
    return avg_test_loss

# Main function to run the training with k-fold cross-validation
def main():
    initial_positions, final_momenta, scaler_positions, scaler_momenta = load_data(file_path)
    input_dim = final_momenta.shape[1]

    dataset = TensorDataset(torch.tensor(final_momenta, dtype=torch.float32).to(device),
                            torch.tensor(initial_positions, dtype=torch.float32).to(device))

    # Split dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Perform k-fold cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler)

        model = VDM(input_dim, hidden_dim).to(device)
        train_losses, val_losses = train_model(model, train_loader, val_loader, epochs, lr, beta, T, patience, delta, best_model_path)

        plot_learning_curve(train_losses, val_losses, fold)

        # Evaluate the model on the test set
        test_loss = evaluate_model(model, test_loader, beta, T)
        print(f'Test Loss for Fold {fold}: {test_loss:.6f}')

        fold_results.append({
            'fold': fold,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_loss': test_loss
        })

        torch.save(model.state_dict(), f'VDM_V2_Fold_{fold}.pth')

    # Print average test loss across all folds
    avg_test_loss = np.mean([result['test_loss'] for result in fold_results])
    print(f'Average Test Loss across all folds: {avg_test_loss:.6f}')

    # Save both scalers
    joblib.dump(scaler_positions, 'scaler_positions.joblib')
    joblib.dump(scaler_momenta, 'scaler_momenta.joblib')
    print("Scalers saved successfully.")


if __name__ == '__main__':
    main()