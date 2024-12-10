import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import Normalizer
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================== #
#         Hyperparameters        #
# ============================== #

# Data Configuration
DATA_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
INPUT_POSITION_COLUMNS = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
INPUT_MOMENTUM_COLUMNS = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']

# Training Parameters
BATCH_SIZE = 2048
NUM_EPOCHS = 100
LEARNING_RATE = 3.573351033584548e-05
WEIGHT_DECAY = 4.131359745650163e-08

# Learning Rate Scheduler Parameters
LR_STEP_SIZE = 15
LR_GAMMA = 0.5622582997294658

# Model Architecture Parameters
NUM_COUPLING_BLOCKS = 57
HIDDEN_DIM = 649
CLAMPING_VALUE = 2.1250677354716623

# Activation Function
ACTIVATION_FN = nn.ReLU

# Early Stopping Parameters
EARLY_STOPPING_PATIENCE = 10

# Gradient Clipping Value
GRAD_CLIP_VALUE = 1.0

# ============================== #
#         Device Configuration   #
# ============================== #

# Detect if CUDA is available and set device accordingly
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================== #
#         Data Loading           #
# ============================== #

# Load data
data = pd.read_csv(DATA_PATH)

# Extract positions (X) and momenta (Y)
positions = data[INPUT_POSITION_COLUMNS].values  # Shape: (N, D1)
momenta = data[INPUT_MOMENTUM_COLUMNS].values  # Shape: (N, D2)

# Compute L2 norms for positions before normalization
positions_norms = np.linalg.norm(positions, axis=1, keepdims=True)  # Shape: (N, 1)

# Handle cases where norm is zero to avoid division by zero
positions_norms[positions_norms == 0] = 1.0

# Initialize Normalizer
pos_normalizer = Normalizer()
mom_normalizer = Normalizer()

# Normalize the data
positions_norm = pos_normalizer.fit_transform(positions)  # Shape: (N, D1)
momenta_norm = mom_normalizer.fit_transform(momenta)      # Shape: (N, D2)

# Convert to PyTorch tensors
positions_norm = torch.tensor(positions_norm, dtype=torch.float32)
momenta_norm = torch.tensor(momenta_norm, dtype=torch.float32)
positions_norms = torch.tensor(positions_norms, dtype=torch.float32)  # Store norms for inversion

# ============================== #
#          Dataset Splitting      #
# ============================== #

# Define split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Ensure that the ratios sum to 1
assert train_ratio + val_ratio + test_ratio == 1.0, "Train, val, test ratios must sum to 1."

# Get total number of samples
total_samples = positions_norm.shape[0]

# Generate shuffled indices
indices = np.arange(total_samples)
np.random.seed(42)  # For reproducibility
np.random.shuffle(indices)

# Compute split sizes
train_size = int(train_ratio * total_samples)
val_size = int(val_ratio * total_samples)
test_size = total_samples - train_size - val_size

# Split indices
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Split the data accordingly
train_positions = positions_norm[train_indices]
train_momenta = momenta_norm[train_indices]

val_positions = positions_norm[val_indices]
val_momenta = momenta_norm[val_indices]

test_positions = positions_norm[test_indices]
test_momenta = momenta_norm[test_indices]
test_norms = positions_norms[test_indices]  # Store norms for inverse_transform

# Create TensorDatasets
train_dataset = TensorDataset(train_positions, train_momenta)
val_dataset = TensorDataset(val_positions, val_momenta)
test_dataset = TensorDataset(test_positions, test_momenta, test_norms)

# ============================== #
#         DataLoaders            #
# ============================== #

# Create data loaders with pin_memory=True if using CUDA for faster data transfer
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    pin_memory=True if DEVICE.type == 'cuda' else False
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    pin_memory=True if DEVICE.type == 'cuda' else False
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    pin_memory=True if DEVICE.type == 'cuda' else False
)

# ============================== #
#       Model Definition         #
# ============================== #

# Define the subnet constructor for the coupling layers
def subnet_fc(dims_in, dims_out):
    return nn.Sequential(
        nn.Linear(dims_in, HIDDEN_DIM),
        ACTIVATION_FN(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        ACTIVATION_FN(),
        nn.Linear(HIDDEN_DIM, dims_out)
    )

# Dimensions
input_dim = positions.shape[1]      # Number of position features
condition_dim = momenta.shape[1]    # Number of momentum features

# Build the invertible network
nodes = [Ff.InputNode(input_dim, name='input')]
cond = Ff.ConditionNode(condition_dim, name='condition')

for k in range(NUM_COUPLING_BLOCKS):
    nodes.append(Ff.Node(
        nodes[-1],
        Fm.GLOWCouplingBlock,
        {'subnet_constructor': subnet_fc, 'clamp': CLAMPING_VALUE},
        conditions=cond,
        name=f'coupling_{k}'
    ))

nodes.append(Ff.OutputNode(nodes[-1], name='output'))

# Create the model and move it to the selected device
cinn = Ff.ReversibleGraphNet(nodes + [cond], verbose=False).to(DEVICE)

# ============================== #
#          Loss Function         #
# ============================== #

def cinn_loss(z, log_jacob_det):
    """
    Computes the negative log-likelihood loss.
    """
    nll = 0.5 * torch.sum(z ** 2, dim=1) - log_jacob_det
    return torch.mean(nll)

# ============================== #
#          Optimizer             #
# ============================== #

optimizer = torch.optim.Adam(cinn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

# ============================== #
#          Training Loop         #
# ============================== #

train_losses = []
val_losses = []

best_val_loss = float('inf')
patience = EARLY_STOPPING_PATIENCE  # Early stopping patience
counter = 0

for epoch in range(NUM_EPOCHS):
    # Training Phase
    cinn.train()
    total_train_loss = 0.0

    for batch in train_loader:
        x_batch, y_batch = batch  # Unpack the batch

        # Move data to the selected device
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        z, log_jacob_det = cinn(x_batch, c=[y_batch])
        loss = cinn_loss(z, log_jacob_det)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(cinn.parameters(), GRAD_CLIP_VALUE)

        optimizer.step()
        total_train_loss += loss.item() * x_batch.size(0)

    avg_train_loss = total_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # Validation Phase
    cinn.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_batch = batch  # Unpack the batch

            # Move data to the selected device
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            z, log_jacob_det = cinn(x_batch, c=[y_batch])
            loss = cinn_loss(z, log_jacob_det)
            total_val_loss += loss.item() * x_batch.size(0)

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        # Save the best model
        torch.save(cinn.state_dict(), 'best_cinn_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

    # Step the scheduler
    scheduler.step()

    # Print epoch statistics
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

# ============================== #
#          Plotting              #
# ============================== #

epochs_range = range(1, len(train_losses) + 1)

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('Training_Validation_Loss.png')
plt.close()

# ============================== #
#         Evaluation             #
# ============================== #

# Load the best model
cinn.load_state_dict(torch.load('best_cinn_model.pth'))
cinn.eval()

with torch.no_grad():
    x_trues = []
    x_preds = []
    norms_list = []

    for batch in test_loader:
        x_batch, y_batch, norms_batch = batch  # Unpack the batch

        # Move data to the selected device
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        # Sample z from standard normal and move to device
        z_sample = torch.randn_like(x_batch).to(DEVICE, non_blocking=True)

        # Generate predictions by reversing the flow
        x_pred_batch, _ = cinn(z_sample, c=[y_batch], rev=True)

        # Append true and predicted normalized positions
        x_trues.append(x_batch.cpu())          # Move back to CPU for numpy operations
        x_preds.append(x_pred_batch.cpu())

        # Append norms for inversion
        norms_list.append(norms_batch.cpu())

    # Concatenate all batches
    x_true_norm = torch.cat(x_trues, dim=0).numpy()    # Shape: (N_test, D1)
    x_pred_norm = torch.cat(x_preds, dim=0).numpy()    # Shape: (N_test, D1)
    test_norms = torch.cat(norms_list, dim=0).numpy()  # Shape: (N_test, 1)

    # Perform inverse transformation by multiplying with stored norms
    x_true_inv = x_true_norm * test_norms              # Shape: (N_test, D1)
    x_pred_inv = x_pred_norm * test_norms              # Shape: (N_test, D1)

    # Compute Mean Squared Error on the original scale
    mse = mean_squared_error(x_true_inv, x_pred_inv)
    print(f'Mean Squared Error on Test Set: {mse:.6f}')

# ============================== #
#         Final Plotting         #
# ============================== #

# Plot True vs Predicted for a few samples
num_samples_to_plot = 5
for i in range(num_samples_to_plot):
    plt.figure(figsize=(8, 4))
    plt.plot(x_true_inv[i], label='True')
    plt.plot(x_pred_inv[i], label='Predicted')
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.title(f'Sample {i+1}: True vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Sample_{i+1}_True_vs_Predicted.png')
    plt.close()
