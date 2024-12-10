# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random

# Set random seeds for reproducibility
def set_seed(seed=90):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)    

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Path to your CSV data file
data_path = '/home/g/ghanaatian/MYFILES/FALL24/Physics/Fifth/cei_traning_orient_1.csv'

# Load the dataset
data = pd.read_csv(data_path)

# Extract initial positions (X) and final momenta (Y)
positions = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values  # Shape: (N, 9)
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values  # Shape: (N, 9)

'''# Print a few samples to verify the correspondence
print("Position samples:")
print(positions[:5])
print("Momenta samples:")
print(momenta[:5])'''

# Create an array of indices
indices = np.arange(len(positions))

# Split the data into training and testing
X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
    positions, momenta, indices, test_size=0.2, random_state=42, shuffle=True
)

# Normalize the data based on training data
scaler_position = MinMaxScaler()
scaler_momenta = MinMaxScaler()

X_train_norm = scaler_position.fit_transform(X_train)
Y_train_norm = scaler_momenta.fit_transform(Y_train)

X_test_norm = scaler_position.transform(X_test)
Y_test_norm = scaler_momenta.transform(Y_test)

'''# Print the scaled data to verify normalization
print("Scaled position samples:")
print(X_train_norm[:5])
print("Scaled momenta samples:")
print(Y_train_norm[:5])'''

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(Y_train_norm, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(Y_test_norm, dtype=torch.float32).to(device)

# Create DataLoader for batching
batch_size = 512  # Adjust batch size as needed
train_dataset = data_utils.TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = data_utils.TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the CINN model architecture
def create_cinn_model(input_dim=9, condition_dim=9, hidden_features=256, num_blocks=4):
    nodes = [Ff.InputNode(input_dim, name='input')]

    condition_node = Ff.ConditionNode(condition_dim, name='condition')

    for i in range(num_blocks):
        nodes.append(
            Ff.Node(
                nodes[-1],
                Fm.GLOWCouplingBlock,
                {
                    'subnet_constructor': lambda in_features, out_features: nn.Sequential(
                        nn.Linear(in_features, hidden_features),
                        nn.LeakyReLU(),
                        nn.Linear(hidden_features, hidden_features),
                        nn.LeakyReLU(),
                        nn.Linear(hidden_features, out_features)
                    ),
                    'clamp': 2.0,
                },
                conditions=condition_node,
                name=f'block_{i}'
            )
        )

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))

    return Ff.ReversibleGraphNet(nodes + [condition_node])

# Instantiate the CINN model
input_dim = 9      # Positions dimension
condition_dim = 9  # Momenta dimension

cinn_model = create_cinn_model(input_dim=input_dim, condition_dim=condition_dim, hidden_features=256, num_blocks=4)
cinn_model = cinn_model.to(device)

# Define loss computation function for training
def compute_loss(model, x, y):
    z, _ = model(x, c=y)
    x_pred, _ = model(z, c=y, rev=True)
    loss = nn.MSELoss()(x_pred, x)
    return loss

# Modify the validation function to prevent data leakage
def validate(model, val_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_y = batch_y.to(device)
            batch_size = batch_y.size(0)
            # Sample z from standard normal distribution or use zeros
            z = torch.zeros(batch_size, input_dim).to(device)
            x_pred, _ = model(z, c=batch_y, rev=True)
            # Move batch_x to device for loss computation
            batch_x = batch_x.to(device)
            loss = nn.MSELoss()(x_pred, batch_x)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Define the optimizer and scheduler
learning_rate = 5e-4  # Adjust learning rate as needed
weight_decay = 1e-5
optimizer = optim.AdamW(cinn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Initialize variables to track the best validation loss
best_val_loss = float('inf')
best_model_path = 'best_model.pth'

# Early stopping parameters
patience_counter = 0
patience_limit = 20  # Set your patience level

# Training parameters
num_epochs = 2000

# Training loop with progress bar
train_losses = []
val_losses = []
prev_lr = optimizer.param_groups[0]['lr']

for epoch in tqdm(range(1, num_epochs + 1), desc='Training'):
    cinn_model.train()
    epoch_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = compute_loss(cinn_model, batch_x, batch_y)  # x: positions, y: momenta
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Perform validation at the end of each epoch
    val_loss = validate(cinn_model, test_loader)
    val_losses.append(val_loss)

    # Scheduler step based on validation loss
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    if current_lr != prev_lr:
        print(f'Learning rate changed from {prev_lr:.2e} to {current_lr:.2e}')
        prev_lr = current_lr

    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}')

    # Check if the current model is the best so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  # Reset patience counter
        try:
            torch.save(cinn_model.state_dict(), best_model_path)
            print(f'Best model saved at epoch {epoch} with Val Loss: {val_loss:.6f}')
        except Exception as e:
            print(f'Error saving best model at epoch {epoch}: {e}')
    else:
        patience_counter += 1
        if patience_counter >= patience_limit:
            print("Early stopping triggered.")
            break

    # Save periodic checkpoints
    if epoch % 100 == 0:
        checkpoint_path = f'cinn_checkpoint_epoch_{epoch}.pth'
        torch.save(cinn_model.state_dict(), checkpoint_path)
        print(f'Checkpoint saved at epoch {epoch}')

# After training, load the best model
if os.path.exists(best_model_path):
    try:
        cinn_model.load_state_dict(torch.load(best_model_path, weights_only=True, map_location=device))
        print('Best model loaded successfully after training.')
    except Exception as e:
        print(f'Error loading best model: {e}')
else:
    print(f'Best model not found at {best_model_path}. Evaluation will use the last epoch model.')

# Plotting the training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()
plt.savefig('paper-final-V2-final2Xcd V2.png')

# Evaluate the model on the test set and calculate MSE on original scale data
cinn_model.eval()
all_predicted_X = []
all_true_X = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_y = batch_y.to(device)
        batch_size = batch_y.size(0)
        # Sample z from standard normal distribution or use zeros
        z = torch.zeros(batch_size, input_dim).to(device)
        x_pred, _ = cinn_model(z, c=batch_y, rev=True)  # Reverse mapping to get positions
        all_predicted_X.append(x_pred.cpu().numpy())
        all_true_X.append(batch_x.cpu().numpy())

# Concatenate all predicted and true positions
all_predicted_X = np.concatenate(all_predicted_X, axis=0)
all_true_X = np.concatenate(all_true_X, axis=0)

# De-normalize all predicted and true positions to get them in original scale
all_predicted_X_orig = scaler_position.inverse_transform(all_predicted_X)
all_true_X_orig = scaler_position.inverse_transform(all_true_X)

# Calculate MSE on original scale data
mse_original = mean_squared_error(all_true_X_orig, all_predicted_X_orig)
print(f'Test MSE on original scale data: {mse_original:.6f}')

# Define a function to compute the relative error
def compute_relative_error(y_true, y_pred):
    # Avoid division by zero by adding a small epsilon
    relative_errors = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)
    mean_relative_error = np.mean(relative_errors)
    return mean_relative_error

# Compute the relative error
relative_error = compute_relative_error(all_true_X_orig, all_predicted_X_orig)
print(f'Relative Error: {relative_error:.6f}')

# Randomly select 2 samples from the test set
num_samples = 2
random_indices = np.random.choice(len(X_test), num_samples, replace=False)

print("\nRandomly selected samples from the test set:")
for i, idx_in_test in enumerate(random_indices):
    idx = idx_test[idx_in_test]  # Index in the original data

    # Get the original scale momenta and positions
    real_momenta = momenta[idx]  # Original momenta (not normalized)
    real_position = positions[idx]  # Original position (not normalized)

    # Get the normalized momenta tensor
    batch_y = Y_test_tensor[idx_in_test].unsqueeze(0)  # Normalized momenta tensor

    # Predict position without using the true position
    z = torch.zeros(1, input_dim).to(device)
    with torch.no_grad():
        x_pred_norm, _ = cinn_model(z, c=batch_y, rev=True)
    x_pred_norm = x_pred_norm.cpu().numpy()

    # De-normalize the predicted positions
    x_pred = scaler_position.inverse_transform(x_pred_norm)

    '''# Print out the data
    print(f"\nSample {i+1}:")
    print("Momenta (original scale):")
    print(real_momenta)
    print("Real Position (original scale):")
    print(real_position)
    print("Predicted Position (original scale):")
    print(x_pred.flatten())'''
