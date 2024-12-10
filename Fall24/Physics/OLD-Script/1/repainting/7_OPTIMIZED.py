import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# Changeable parameters
RANDOM_SEED = 42
DATA_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/1M.csv'
INPUT_COLUMNS = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']
OUTPUT_COLUMNS = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_DELTA = 1e-3
MODEL_SAVE_PATH = 'best_model.pth'

# Neural network architecture
ENCODER_LAYERS = [
    (9, 1024),
    (1024, 512),
    (512, 32)
]
DECODER_LAYERS = [
    (32, 256),
    (256, 128),
    (128, 9)
]

# Set device for GPU computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
data = pd.read_csv(DATA_PATH)

# Define inputs and outputs
inputs = data[INPUT_COLUMNS].values
outputs = data[OUTPUT_COLUMNS].values

# Normalize the data
scaler_in = StandardScaler()
scaler_out = StandardScaler()
inputs = scaler_in.fit_transform(inputs)
outputs = scaler_out.fit_transform(outputs)

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(inputs, outputs, test_size=TEST_SIZE, random_state=RANDOM_SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.float32).to(device)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# Define the Encoder-Decoder model
class EncoderDecoder(nn.Module):
    def __init__(self, encoder_layers, decoder_layers):
        super(EncoderDecoder, self).__init__()
        self.encoder = self._build_layers(encoder_layers)
        self.decoder = self._build_layers(decoder_layers)
    
    def _build_layers(self, layer_specs):
        layers = []
        for in_features, out_features in layer_specs:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers[:-1])  # Remove the last ReLU
    
    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

# Initialize the model, loss function, and optimizer
model = EncoderDecoder(ENCODER_LAYERS, DECODER_LAYERS).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-3):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss, model, model_path='best_model.pth'):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, model_path)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(model, model_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def save_checkpoint(self, model, model_path):
        torch.save(model.state_dict(), model_path)

# Train the model
early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, delta=EARLY_STOPPING_DELTA)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    
    val_loss /= len(val_loader.dataset)
    
    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')
    
    early_stopping(val_loss, model, model_path=MODEL_SAVE_PATH)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

# Load the best model
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Evaluate the model on the test set
model.eval()
test_loss = 0.0
all_preds = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())
        loss = criterion(outputs, y_batch)
        test_loss += loss.item() * X_batch.size(0)

test_loss /= len(test_loader.dataset)

all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# Inverse transform the scaled data
all_preds = scaler_out.inverse_transform(all_preds)
all_targets = scaler_out.inverse_transform(all_targets)

# Calculate metrics
mse = mean_squared_error(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)
r2 = r2_score(all_targets, all_preds)

print(f'Test Loss: {test_loss:.6f}')
print(f'Test MSE: {mse:.6f}')
print(f'Test MAE: {mae:.6f}')
print(f'Test R2 Score: {r2:.6f}')