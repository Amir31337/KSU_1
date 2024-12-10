import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import optuna

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set device for GPU computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Changeable parameters
DATA_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/10K.csv'
INPUT_COLUMNS = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']
OUTPUT_COLUMNS = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.5
BATCH_SIZE = 512
EPOCHS = 50
N_TRIALS = 100  # Number of Optuna trials

# Load and preprocess data
data = pd.read_csv(DATA_PATH)
inputs = data[INPUT_COLUMNS].values
outputs = data[OUTPUT_COLUMNS].values

scaler_in = StandardScaler()
scaler_out = StandardScaler()
inputs = scaler_in.fit_transform(inputs)
outputs = scaler_out.fit_transform(outputs)

X_train, X_temp, y_train, y_temp = train_test_split(inputs, outputs, test_size=TEST_SIZE, random_state=RANDOM_SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, dropout_rate):
        super(EncoderDecoder, self).__init__()
        self.encoder = self._build_layers(encoder_layers, dropout_rate)
        self.decoder = self._build_layers(decoder_layers, dropout_rate)
    
    def _build_layers(self, layer_specs, dropout_rate):
        layers = []
        for i, (in_features, out_features) in enumerate(layer_specs):
            layers.append(nn.Linear(in_features, out_features))
            if i < len(layer_specs) - 1:  # No activation or dropout after the last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

def train_and_evaluate(model, optimizer, criterion, train_loader, val_loader, epochs):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss

def objective(trial):
    # Hyperparameters to optimize
    n_layers = trial.suggest_int('n_layers', 1, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    
    # Define layer sizes
    layer_sizes = [9] + [trial.suggest_int(f'layer_{i}', 16, 256) for i in range(n_layers)] + [9]
    
    # Define encoder and decoder layers
    encoder_layers = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    decoder_layers = list(zip(layer_sizes[:0:-1], layer_sizes[-2::-1]))
    
    # Create model, optimizer, and loss function
    model = EncoderDecoder(encoder_layers, decoder_layers, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Train and evaluate the model
    val_loss = train_and_evaluate(model, optimizer, criterion, train_loader, val_loader, EPOCHS)
    
    return val_loss

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=N_TRIALS)

# Print results
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Create the best model
best_n_layers = trial.params['n_layers']
best_dropout_rate = trial.params['dropout_rate']
best_layer_sizes = [9] + [trial.params[f'layer_{i}'] for i in range(best_n_layers)] + [9]

best_encoder_layers = list(zip(best_layer_sizes[:-1], best_layer_sizes[1:]))
best_decoder_layers = list(zip(best_layer_sizes[:0:-1], best_layer_sizes[-2::-1]))

best_model = EncoderDecoder(best_encoder_layers, best_decoder_layers, best_dropout_rate).to(device)
best_optimizer = optim.Adam(best_model.parameters(), lr=trial.params['learning_rate'])
criterion = nn.MSELoss()

# Train the best model
train_and_evaluate(best_model, best_optimizer, criterion, train_loader, val_loader, EPOCHS)

# Evaluate on test set
best_model.eval()
test_preds = []
test_targets = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = best_model(X_batch)
        test_preds.append(outputs.cpu().numpy())
        test_targets.append(y_batch.cpu().numpy())

test_preds = np.vstack(test_preds)
test_targets = np.vstack(test_targets)

# Inverse transform the scaled data
test_preds = scaler_out.inverse_transform(test_preds)
test_targets = scaler_out.inverse_transform(test_targets)

# Calculate final MSE
final_mse = mean_squared_error(test_targets, test_preds)
print(f"Final Test MSE: {final_mse}")

# Save the best model
torch.save(best_model.state_dict(), 'best_optuna_model.pth')