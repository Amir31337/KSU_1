import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import matplotlib.pyplot as plt
from torchinfo import summary  # Updated to use torchinfo
import warnings

warnings.filterwarnings('ignore')

# Clear GPU memory
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define activation function helper
def get_activation(name):
    if name == 'ReLU':
        return nn.ReLU()
    elif name == 'ELU':
        return nn.ELU()
    elif name == 'LeakyReLU':
        return nn.LeakyReLU()
    elif name == 'Tanh':
        return nn.Tanh()
    elif name == 'Sigmoid':
        return nn.Sigmoid()
    elif name == 'Softplus':
        return nn.Softplus()
    elif name == 'Swish':
        return nn.SiLU()  # Swish is implemented as SiLU in PyTorch
    elif name == 'GELU':
        return nn.GELU()
    else:
        raise ValueError(f'Unknown activation function: {name}')

# Define the CVAE model
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_layers, activation_fn, dropout_rate):
        super(CVAE, self).__init__()
        
        # Encoder: Encodes positions X into latent mean and variance
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(activation_fn)
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[-1], latent_dim)
        
        # Decoder: Decodes latent Z conditioned on momenta Y
        decoder_layers = []
        in_dim = latent_dim + condition_dim
        hidden_layers_rev = hidden_layers[::-1]
        for h_dim in hidden_layers_rev:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(activation_fn)
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        combined = torch.cat((z, condition), dim=1)
        return self.decoder(combined)

    def forward(self, x, condition):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, condition)
        return recon_x, mu, logvar

# Define the CVAE loss function
def cvae_loss_fn(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_divergence

# Best hyperparameters
best_hyperparams = {
    'LATENT_DIM': 32,
    'BATCH_SIZE': 1024,
    'LEARNING_RATE': 7.112898087131894e-06,
    'PATIENCE': 25,
    'MIN_DELTA': 0.0002257112582342097,
    'ACTIVATION': 'ELU',
    'NUM_LAYERS': 3,
    'layer_0_size': 1024,
    'layer_1_size': 256,
    'layer_2_size': 1024,
    'position_norm': 'MaxAbsScaler',
    'momenta_norm': 'MinMaxScaler',
    'REGULARIZATION': 'L1',
    'L1_LAMBDA': 0.0032510808582310174,
    'DROPOUT_RATE': 0.11659430853375953
}

# Define model parameters based on best hyperparameters
INPUT_DIM = 9
OUTPUT_DIM = 9
LATENT_DIM = best_hyperparams['LATENT_DIM']
CONDITION_DIM = OUTPUT_DIM  # Assuming condition is momenta with same dimension

hidden_layers = [
    best_hyperparams['layer_0_size'],
    best_hyperparams['layer_1_size'],
    best_hyperparams['layer_2_size']
]
activation_fn = get_activation(best_hyperparams['ACTIVATION'])
dropout_rate = best_hyperparams['DROPOUT_RATE']

# Initialize the CVAE model
model = CVAE(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM,
    condition_dim=CONDITION_DIM,
    hidden_layers=hidden_layers,
    activation_fn=activation_fn,
    dropout_rate=dropout_rate
)

# Move model to device
model.to(device)

# Print model architecture using torchinfo
# torchinfo can handle multiple inputs
print("\n=== Full CVAE Model Architecture ===")
dummy_x = torch.randn(1, INPUT_DIM).to(device)
dummy_cond = torch.randn(1, CONDITION_DIM).to(device)
summary(model, input_data=(dummy_x, dummy_cond))

# Define the file path
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'

# Load and preprocess data
print("\nLoading and preprocessing data...")
data = pd.read_csv(FILEPATH)

# Extract position and momenta columns
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Normalize the position and momenta using selected methods
if best_hyperparams['position_norm'] == 'StandardScaler':
    position_scaler = StandardScaler()
elif best_hyperparams['position_norm'] == 'MinMaxScaler':
    position_scaler = MinMaxScaler()
elif best_hyperparams['position_norm'] == 'MaxAbsScaler':
    position_scaler = MaxAbsScaler()
elif best_hyperparams['position_norm'] == 'RobustScaler':
    position_scaler = RobustScaler()

if best_hyperparams['momenta_norm'] == 'StandardScaler':
    momenta_scaler = StandardScaler()
elif best_hyperparams['momenta_norm'] == 'MinMaxScaler':
    momenta_scaler = MinMaxScaler()
elif best_hyperparams['momenta_norm'] == 'MaxAbsScaler':
    momenta_scaler = MaxAbsScaler()
elif best_hyperparams['momenta_norm'] == 'RobustScaler':
    momenta_scaler = RobustScaler()

position_normalized = position_scaler.fit_transform(position)
momenta_normalized = momenta_scaler.fit_transform(momenta)

# Split data into train (70%), validation (15%), test (15%) sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    position_normalized, momenta_normalized, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42  # 0.1765 * 0.85 ≈ 0.15
)

# Convert to PyTorch tensors and move to device
train_position = torch.FloatTensor(X_train).to(device)
val_position = torch.FloatTensor(X_val).to(device)
test_position = torch.FloatTensor(X_test).to(device)
train_momenta = torch.FloatTensor(y_train).to(device)
val_momenta = torch.FloatTensor(y_val).to(device)
test_momenta = torch.FloatTensor(y_test).to(device)

# Define optimizer with regularization
if best_hyperparams['REGULARIZATION'] == 'L2':
    WEIGHT_DECAY = best_hyperparams.get('WEIGHT_DECAY', 1e-5)  # Default value if not provided
    optimizer = optim.Adam(model.parameters(), lr=best_hyperparams['LEARNING_RATE'], weight_decay=WEIGHT_DECAY)
else:
    optimizer = optim.Adam(model.parameters(), lr=best_hyperparams['LEARNING_RATE'])

# Define DataLoaders
train_dataset = TensorDataset(train_position, train_momenta)
train_loader = DataLoader(train_dataset, batch_size=best_hyperparams['BATCH_SIZE'], shuffle=True)
val_dataset = TensorDataset(val_position, val_momenta)
val_loader = DataLoader(val_dataset, batch_size=best_hyperparams['BATCH_SIZE'], shuffle=False)

# Training loop parameters
EPOCHS = 1000  # Maximum number of epochs
PATIENCE = best_hyperparams['PATIENCE']
MIN_DELTA = best_hyperparams['MIN_DELTA']
REGULARIZATION = best_hyperparams['REGULARIZATION']
L1_LAMBDA = best_hyperparams.get('L1_LAMBDA', 1e-5)  # Default if not provided

# Initialize tracking variables
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0
best_model_state = model.state_dict()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_train_loss = 0

    for batch_idx, (position_batch, momenta_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_position, mu, logvar = model(position_batch, momenta_batch)
        loss = cvae_loss_fn(recon_position, position_batch, mu, logvar)
        
        # Add L1 regularization if selected
        if REGULARIZATION == 'L1':
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += L1_LAMBDA * l1_norm
        
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    
    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    # Validation phase
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for position_batch, momenta_batch in val_loader:
            recon_position, mu, logvar = model(position_batch, momenta_batch)
            loss = cvae_loss_fn(recon_position, position_batch, mu, logvar)
            epoch_val_loss += loss.item()
    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch}: Train Loss = {epoch_train_loss:.6f}, Val Loss = {epoch_val_loss:.6f}")

    # Check for improvement
    if epoch_val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

# Load the best model state
model.load_state_dict(best_model_state)
model.eval()

# Save training and validation losses plot
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.tight_layout()
plt.savefig('cvae_training_losses.png')
plt.close()
print("Training and validation losses plot saved as 'cvae_training_losses.png'.")

# Evaluation on Test Set
print("\nEvaluating on test set...")
with torch.no_grad():
    # Encode training data to get mu and logvar
    train_mu, train_logvar = model.encode(train_position)
    train_mu = train_mu.cpu().numpy()
    train_logvar = train_logvar.cpu().numpy()
    
    # Compute mean and std from training latent space
    mu_train_mean = np.mean(train_mu, axis=0)
    std_train_mean = np.mean(np.exp(0.5 * train_logvar), axis=0)
    
    # Sample z for test set
    num_test_samples = test_momenta.size(0)
    z_test = np.random.normal(loc=mu_train_mean, scale=std_train_mean, size=(num_test_samples, LATENT_DIM))
    z_test = torch.FloatTensor(z_test).to(device)
    
    # Reconstruct positions conditioned on test momenta
    reconstructed_test_positions = model.decode(z_test, test_momenta)
    
    # Inverse transform the predicted and actual positions
    reconstructed_test_positions_np = reconstructed_test_positions.cpu().numpy()
    test_position_np = test_position.cpu().numpy()
    reconstructed_test_positions_inverse = position_scaler.inverse_transform(reconstructed_test_positions_np)
    test_position_inverse = position_scaler.inverse_transform(test_position_np)
    
    # Mean Relative Error (MRE)
    # Avoid division by zero
    test_position_inverse_safe = np.where(test_position_inverse == 0, 1e-8, test_position_inverse)
    mre_test = np.mean(np.abs((test_position_inverse - reconstructed_test_positions_inverse) / np.abs(test_position_inverse_safe)))
    print(f"Test Mean Relative Error (MRE): {mre_test:.6f}")

# Save the results
results = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'test_mre': mre_test
}
torch.save(results, 'cvae_results.pt')
print("Results saved to 'cvae_results.pt'.")

# Print full model architecture using torchinfo
print("\n=== Final CVAE Model Architecture ===")
summary(model, input_data=(dummy_x, dummy_cond))
