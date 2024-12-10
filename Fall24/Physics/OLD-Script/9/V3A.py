import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Hyperparameters
data_path = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
batch_size = 2048
num_epochs = 50
learning_rate = 1e-5
weight_decay = 1e-7
lr_step_size = 10
lr_gamma = 0.7
num_coupling_blocks = 32
hidden_dim = 512
clamping_value = 2.6

# Load data
data = pd.read_csv(data_path)

# Extract positions (X) and momenta (Y)
positions = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Convert to PyTorch tensors
positions = torch.tensor(positions, dtype=torch.float32)
momenta = torch.tensor(momenta, dtype=torch.float32)

# Normalize the data
pos_scaler = MinMaxScaler()
mom_scaler = MinMaxScaler()
positions_norm = torch.tensor(pos_scaler.fit_transform(positions), dtype=torch.float32)
momenta_norm = torch.tensor(mom_scaler.fit_transform(momenta), dtype=torch.float32)

# Create dataset
dataset = TensorDataset(positions_norm, momenta_norm)

# Split into training, validation, and test sets
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the subnet constructor for the coupling layers
def subnet_fc(dims_in, dims_out):
    return nn.Sequential(
        nn.Linear(dims_in, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, dims_out)
    )

# Dimensions
input_dim = positions.shape[1]
condition_dim = momenta.shape[1]

# Build the invertible network
nodes = [Ff.InputNode(input_dim, name='input')]
cond = Ff.ConditionNode(condition_dim, name='condition')

for k in range(num_coupling_blocks):
    nodes.append(Ff.Node(
        nodes[-1],
        Fm.GLOWCouplingBlock,
        {'subnet_constructor': subnet_fc, 'clamp': clamping_value},
        conditions=cond,
        name=f'coupling_{k}'
    ))

nodes.append(Ff.OutputNode(nodes[-1], name='output'))

# Create the model
cinn = Ff.ReversibleGraphNet(nodes + [cond], verbose=False)

def cinn_loss(z, log_jacob_det):
    nll = 0.5 * torch.sum(z ** 2, dim=1) - log_jacob_det
    return torch.mean(nll)

# Optimizer
optimizer = torch.optim.Adam(cinn.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

# Training loop
train_losses = []
val_losses = []
test_losses = []

for epoch in range(num_epochs):
    cinn.train()
    total_loss = 0.0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        z, log_jacob_det = cinn(x_batch, c=[y_batch])
        loss = cinn_loss(z, log_jacob_det)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # Validation
    cinn.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            z, log_jacob_det = cinn(x_batch, c=[y_batch])
            loss = cinn_loss(z, log_jacob_det)
            val_loss += loss.item() * x_batch.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    # Test Loss
    test_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            z, log_jacob_det = cinn(x_batch, c=[y_batch])
            loss = cinn_loss(z, log_jacob_det)
            test_loss += loss.item() * x_batch.size(0)
    avg_test_loss = test_loss / len(test_loader.dataset)
    test_losses.append(avg_test_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    scheduler.step()

# Plotting loss curves
epochs = range(1, num_epochs + 1)

# First 10 epochs
first_10_epochs = epochs[:10]
plt.figure(figsize=(10, 6))
plt.plot(first_10_epochs, train_losses[:10], label='Training Loss')
plt.plot(first_10_epochs, val_losses[:10], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epochs (First 10 Epochs)')
plt.legend()
plt.savefig('V3-First.png')

# Remaining epochs
remaining_epochs = epochs[10:]

plt.figure(figsize=(10, 6))
plt.plot(remaining_epochs, train_losses[10:], label='Training Loss')
plt.plot(remaining_epochs, val_losses[10:], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epochs (Epochs 11 onwards)')
plt.legend()
plt.savefig('V3-Second.png')

# Evaluate on the test set
cinn.eval()
with torch.no_grad():
    x_trues = []
    x_preds = []
    for x_batch, y_batch in test_loader:
        z_sample = torch.randn_like(x_batch)
        x_pred_batch, _ = cinn(z_sample, c=[y_batch], rev=True)
        x_trues.append(x_batch)
        x_preds.append(x_pred_batch)

    x_true = torch.cat(x_trues, dim=0)
    x_pred = torch.cat(x_preds, dim=0)

    x_pred_np = x_pred.numpy()
    x_true_np = x_true.numpy()

    x_pred_inv = pos_scaler.inverse_transform(x_pred_np)
    x_true_inv = pos_scaler.inverse_transform(x_true_np)

    mse = mean_squared_error(x_true_inv, x_pred_inv)
    print(f'Mean Squared Error on Test Set: {mse:.4f}')