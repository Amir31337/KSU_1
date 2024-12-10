import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

batch_size = 2048
input_size = 9
hidden_sizes = [256, 128, 64]
latent_size = 64
activation_fn = nn.Tanh

# Load dataset
data = pd.read_csv('/home/g/ghanaatian/MYFILES/FALL24/Physics/1M.csv')

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

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# Model definition
class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, activation_fn=nn.ReLU):
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


# Initialize model, loss, optimizer
model = EncoderDecoder(input_size=input_size, hidden_sizes=hidden_sizes, latent_size=latent_size, activation_fn=activation_fn)
if torch.cuda.is_available():
    model.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-3):
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

# Training loop
early_stopping = EarlyStopping(patience=10, delta=1e-3)
best_val_loss = np.inf
best_model_path = 'best_model.pth'

for epoch in range(100):
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        if torch.cuda.is_available():
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x_batch.size(0)

    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            if torch.cuda.is_available():
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * x_batch.size(0)

    val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# Load the best model
model.load_state_dict(torch.load(best_model_path))

# Test the model
model.eval()
test_loss = 0.0
predictions = []
actuals = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        if torch.cuda.is_available():
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        
        y_pred = model(x_batch)
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

