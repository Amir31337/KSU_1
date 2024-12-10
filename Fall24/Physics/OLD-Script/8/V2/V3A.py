import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# ============================== #
#       Hyperparameters          #
# ============================== #

# Data Configuration
DATA_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/Fifth/cei_traning_orient_1.csv'
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

# ============================== #
#         Data Loading           #
# ============================== #

# Load data
data = pd.read_csv(DATA_PATH)

# Extract positions (X) and momenta (Y)
positions = data[INPUT_POSITION_COLUMNS].values
momenta = data[INPUT_MOMENTUM_COLUMNS].values

# Convert to PyTorch tensors
positions = torch.tensor(positions, dtype=torch.float32)
momenta = torch.tensor(momenta, dtype=torch.float32)

# Normalize the data
pos_ = MinMaxScaler()
mom_ = MinMaxScaler()
positions_norm = torch.tensor(pos_.fit_transform(positions), dtype=torch.float32)
momenta_norm = torch.tensor(mom_.fit_transform(momenta), dtype=torch.float32)

# Create dataset
dataset = TensorDataset(positions_norm, momenta_norm)

# Split into training, validation, and test sets
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

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
input_dim = positions.shape[1]
condition_dim = momenta.shape[1]

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

# Create the model
cinn = Ff.ReversibleGraphNet(nodes + [cond], verbose=False)

# ============================== #
#          Loss Function         #
# ============================== #

def cinn_loss(z, log_jacob_det):
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
test_losses = []

for epoch in range(NUM_EPOCHS):
    # Training Phase
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

    # Validation Phase
    cinn.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            z, log_jacob_det = cinn(x_batch, c=[y_batch])
            loss = cinn_loss(z, log_jacob_det)
            val_loss += loss.item() * x_batch.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    # Test Phase
    test_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            z, log_jacob_det = cinn(x_batch, c=[y_batch])
            loss = cinn_loss(z, log_jacob_det)
            test_loss += loss.item() * x_batch.size(0)
    avg_test_loss = test_loss / len(test_loader.dataset)
    test_losses.append(avg_test_loss)

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Step the scheduler
    scheduler.step()

# ============================== #
#          Plotting              #
# ============================== #

epochs = range(1, NUM_EPOCHS + 1)

# First 10 epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs[:10], train_losses[:10], label='Training Loss')
plt.plot(epochs[:10], val_losses[:10], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epochs (First 10 Epochs)')
plt.legend()
plt.savefig('V3-First.png')
plt.close()

# Remaining epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs[10:], train_losses[10:], label='Training Loss')
plt.plot(epochs[10:], val_losses[10:], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epochs (Epochs 11 onwards)')
plt.legend()
plt.savefig('V3-Second.png')
plt.close()

# ============================== #
#         Evaluation             #
# ============================== #

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

    x_pred_inv = pos_.inverse_transform(x_pred_np)
    x_true_inv = pos_.inverse_transform(x_true_np)

    mse = mean_squared_error(x_true_inv, x_pred_inv)
    print(f'Mean Squared Error on Test Set: {mse:.4f}')
