import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Configurable parameters
csv_path = '/home/g/ghanaatian/MYFILES/FALL24/Physics/1M.csv'
input_columns = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']
output_columns = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
latent_dim = 64
batch_size = 2048
learning_rate = 0.001
n_epochs = 50
patience = 10
delta = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model architecture parameters
encoder_layers = [128, 256]
decoder_layers = [256, 256]

# Step 1: Data Preparation
data = pd.read_csv(csv_path)

X = data[input_columns].values
y = data[output_columns].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 2: Model Architecture
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, layers[0])])
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))
        self.layers.append(nn.Linear(layers[-1], latent_dim))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        z = self.layers[-1](x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(latent_dim, layers[0])])
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))
        self.layers.append(nn.Linear(layers[-1], output_dim))
    
    def forward(self, z):
        for layer in self.layers[:-1]:
            z = torch.relu(layer(z))
        out = self.layers[-1](z)
        return out

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, encoder_layers, decoder_layers):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, encoder_layers)
        self.decoder = Decoder(latent_dim, output_dim, decoder_layers)
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

input_dim = len(input_columns)
output_dim = len(output_columns)

model = Autoencoder(input_dim, latent_dim, output_dim, encoder_layers, decoder_layers).to(device)

# Step 3: Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 4: Training the Model with Early Stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, patience, delta):
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')
        
        if val_loss < best_loss - delta:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping")
            break

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, patience, delta)

# Step 5: Model Evaluation
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

test_loss = 0.0
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        test_loss += criterion(outputs, targets).item() * inputs.size(0)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

test_loss /= len(test_loader.dataset)
all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

all_preds = scaler_y.inverse_transform(all_preds)
all_targets = scaler_y.inverse_transform(all_targets)

mse = mean_squared_error(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)
r2 = r2_score(all_targets, all_preds)

print(f"Test MSE: {mse:.6f}")
print(f"Test MAE: {mae:.6f}")
print(f"Test R2: {r2:.6f}")