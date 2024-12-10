import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

# Load dataset
data = pd.read_csv('/home/g/ghanaatian/MYFILES/FALL24/Physics/10K.csv')

# Select input (momenta) and output (initial positions) columns
inputs = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values
outputs = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values

# Normalize the data
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

inputs = input_scaler.fit_transform(inputs)
outputs = output_scaler.fit_transform(outputs)

# Convert to torch tensors
inputs = torch.tensor(inputs, dtype=torch.float32)
outputs = torch.tensor(outputs, dtype=torch.float32)

# Dataset and DataLoader
dataset = TensorDataset(inputs, outputs)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Model definition
class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, activation_fn):
        super(EncoderDecoder, self).__init__()
        
        # Encoder definition
        encoder_layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            encoder_layers.append(nn.Linear(current_size, hidden_size))
            encoder_layers.append(activation_fn())
            current_size = hidden_size
        encoder_layers.append(nn.Linear(current_size, latent_size))
        
        # Decoder definition
        decoder_layers = []
        current_size = latent_size
        for hidden_size in reversed(hidden_sizes):
            decoder_layers.append(nn.Linear(current_size, hidden_size))
            decoder_layers.append(activation_fn())
            current_size = hidden_size
        decoder_layers.append(nn.Linear(current_size, input_size))
        
        # Build Sequential models
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

# Training function
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * x_batch.size(0)
    
    val_loss /= len(val_loader.dataset)
    return val_loss

# Optuna objective function
def objective(trial):
    # Define hyperparameters to optimize
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    hidden_sizes = [trial.suggest_int(f"hidden_size_{i}", 32, 256) for i in range(3)]
    latent_size = trial.suggest_int("latent_size", 16, 128)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    activation_name = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "ELU", "Tanh"])
    
    # Map activation name to function
    activation_functions = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "ELU": nn.ELU,
        "Tanh": nn.Tanh
    }
    activation_fn = activation_functions[activation_name]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    
    # Initialize model, loss, optimizer
    model = EncoderDecoder(input_size=9, hidden_sizes=hidden_sizes, latent_size=latent_size, activation_fn=activation_fn)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train and evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loss = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device)
    
    return val_loss

# Run Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Print best parameters and score
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Train final model with best parameters
best_params = study.best_params
batch_size = best_params["batch_size"]
hidden_sizes = [best_params[f"hidden_size_{i}"] for i in range(3)]
latent_size = best_params["latent_size"]
lr = best_params["lr"]
activation_fn = activation_functions[best_params["activation"]]

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

final_model = EncoderDecoder(input_size=9, hidden_sizes=hidden_sizes, latent_size=latent_size, activation_fn=activation_fn)
criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr=lr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_and_evaluate(final_model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100)

# Evaluate on test set
final_model.eval()
test_loss = 0.0
predictions = []
actuals = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred = final_model(x_batch)
        predictions.append(y_pred.cpu().numpy())
        actuals.append(y_batch.cpu().numpy())
        loss = criterion(y_pred, y_batch)
        test_loss += loss.item() * x_batch.size(0)

test_loss /= len(test_loader.dataset)
predictions = np.vstack(predictions)
actuals = np.vstack(actuals)

# Inverse transform to original scale
predictions = output_scaler.inverse_transform(predictions)
actuals = output_scaler.inverse_transform(actuals)

# Calculate evaluation metrics
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print(f'Test Loss: {test_loss:.4f}')
print(f'MSE: {mse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'R²: {r2:.4f}')