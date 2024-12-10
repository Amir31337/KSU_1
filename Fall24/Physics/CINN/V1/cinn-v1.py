'''
Epoch 25/50, Train Loss: 1664.8460, Val Loss: 1263.9717
Epoch 50/50, Train Loss: 873.8783, Val Loss: 790.0565
Mean Squared Error on Test Set (Original Scale): 0.9483
Mean Relative Error on Test Set (Original Method): 0.8327
Mean Relative Error on Test Set (Alternative Method): 0.8327
Mean Relative Error for cx: 2.0488
Mean Relative Error for cy: 0.7639
Mean Relative Error for cz: 1.3277
Mean Relative Error for ox: 2.0992
Mean Relative Error for oy: 0.8671
Mean Relative Error for oz: 0.0000
Mean Relative Error for sx: 0.3875
Mean Relative Error for sy: 0.0000
Mean Relative Error for sz: 0.0000
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

########################################
# Configuration and Hyperparameters
########################################

# ----------------------------
# Random Seed for Reproducibility
# ----------------------------
RANDOM_SEED = 42

# ----------------------------
# Data Parameters
# ----------------------------
DATA_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
POSITION_COLUMNS = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
MOMENTA_COLUMNS = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']

# ----------------------------
# Normalization
# ----------------------------
NORMALIZE_POSITIONS = True
NORMALIZE_MOMENTA = True

# ----------------------------
# Dataset Split Ratios
# ----------------------------
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ----------------------------
# DataLoader Parameters
# ----------------------------
BATCH_SIZE = 2048
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False
SHUFFLE_TEST = False

# ----------------------------
# Model Architecture Parameters
# ----------------------------
INPUT_DIM = len(POSITION_COLUMNS)  # 9
CONDITION_DIM = len(MOMENTA_COLUMNS)  # 9
NUM_AFFINE_COUPLING_LAYERS = 20  # Number of Affine Coupling Layers
TOTAL_LAYERS = NUM_AFFINE_COUPLING_LAYERS * 2  # Total layers including Flip layers

# ----------------------------
# Scale and Translate Network Parameters
# ----------------------------
# Shared parameters for both scale and translate networks
SCALE_TRANSLATE_HIDDEN_LAYERS = [128]  # List defining hidden layer sizes
SCALE_TRANSLATE_ACTIVATIONS = ['ELU', 'Tanh']  # List of activation functions for hidden layers
SCALE_OUTPUT_ACTIVATION = 'Sigmoid'  # Activation function for scale_net output
TRANSLATE_OUTPUT_ACTIVATION = 'ELU'  # Activation function for translate_net output

# ----------------------------
# Training Parameters
# ----------------------------
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3

# ----------------------------
# Device Configuration
# ----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################
# Set Random Seed for Reproducibility
########################################
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

########################################
# Data Preparation
########################################

# Load the data
data = pd.read_csv(DATA_PATH)

# Extract positions (x) and momenta (y)
positions = data[POSITION_COLUMNS].values  # x
momenta = data[MOMENTA_COLUMNS].values    # y

# Normalize the data
pos_scaler = StandardScaler() if NORMALIZE_POSITIONS else None
mom_scaler = StandardScaler() if NORMALIZE_MOMENTA else None

positions_norm = pos_scaler.fit_transform(positions) if pos_scaler else positions
momenta_norm = mom_scaler.fit_transform(momenta) if mom_scaler else momenta

# Convert to tensors
positions_tensor = torch.tensor(positions_norm, dtype=torch.float32)
momenta_tensor = torch.tensor(momenta_norm, dtype=torch.float32)

# Create dataset
dataset = TensorDataset(positions_tensor, momenta_tensor)

# Calculate dataset sizes
train_size = int(TRAIN_RATIO * len(dataset))
val_size = int(VAL_RATIO * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_VAL)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TEST)

########################################
# Utility Functions for Building Networks
########################################

def get_activation(name):
    """
    Returns the activation function corresponding to the given name.
    """
    activations = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
        'ELU': nn.ELU(),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        None: nn.Identity()
    }
    return activations.get(name, nn.Identity())

def build_network(input_size, hidden_layers, output_size, activations, output_activation=None):
    """
    Builds a neural network based on the provided configuration.

    Parameters:
        input_size (int): Size of the input layer.
        hidden_layers (list of int): Sizes of hidden layers.
        output_size (int): Size of the output layer.
        activations (list of str): Activation functions for hidden layers.
        output_activation (str or None): Activation function for the output layer.

    Returns:
        nn.Sequential: The constructed neural network.
    """
    layers = []
    current_size = input_size
    for idx, hidden_size in enumerate(hidden_layers):
        layers.append(nn.Linear(current_size, hidden_size))
        activation = get_activation(activations[idx] if idx < len(activations) else None)
        layers.append(activation)
        current_size = hidden_size
    layers.append(nn.Linear(current_size, output_size))
    if output_activation:
        layers.append(get_activation(output_activation))
    return nn.Sequential(*layers)

########################################
# Define the cINN Model Architecture
########################################

class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer used in cINN.
    Conditions on the momenta (y).
    Handles odd input dimensions by splitting appropriately.
    """
    def __init__(self, input_dim, condition_dim,
                 scale_hidden_layers, scale_activations, scale_output_activation,
                 translate_hidden_layers, translate_activations, translate_output_activation):
        super(AffineCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.half_dim = (input_dim + 1) // 2  # Ceiling division to handle odd dimensions
        self.x2_dim = input_dim - self.half_dim  # Dimension of the second half

        # Build Scale Network
        self.scale_net = build_network(
            input_size=self.half_dim + condition_dim,
            hidden_layers=scale_hidden_layers,
            output_size=self.x2_dim,
            activations=scale_activations,
            output_activation=scale_output_activation
        )

        # Build Translate Network
        self.translate_net = build_network(
            input_size=self.half_dim + condition_dim,
            hidden_layers=translate_hidden_layers,
            output_size=self.x2_dim,
            activations=translate_activations,
            output_activation=translate_output_activation
        )

    def forward(self, x, y):
        x1 = x[:, :self.half_dim]  # First half
        x2 = x[:, self.half_dim:]  # Second half

        # Concatenate x1 with y for conditioning
        condition = torch.cat([x1, y], dim=1)

        # Compute scale and translate
        s = self.scale_net(condition)
        t = self.translate_net(condition)

        # Affine transformation
        z1 = x1
        z2 = x2 * torch.exp(s) + t
        z = torch.cat([z1, z2], dim=1)

        # Compute log determinant of the Jacobian
        log_det_jacobian = s.sum(dim=1)

        return z, log_det_jacobian

    def inverse(self, z, y):
        z1 = z[:, :self.half_dim]
        z2 = z[:, self.half_dim:]

        # Concatenate z1 with y for conditioning
        condition = torch.cat([z1, y], dim=1)

        # Compute scale and translate
        s = self.scale_net(condition)
        t = self.translate_net(condition)

        # Inverse affine transformation
        x1 = z1
        x2 = (z2 - t) * torch.exp(-s)
        x = torch.cat([x1, x2], dim=1)

        # Compute log determinant of the Jacobian
        log_det_jacobian = -s.sum(dim=1)

        return x, log_det_jacobian

class FlipLayer(nn.Module):
    """
    Custom layer to flip the input tensor along dimension 1.
    """
    def __init__(self):
        super(FlipLayer, self).__init__()

    def forward(self, x):
        return x.flip(dims=[1])

class cINN(nn.Module):
    """
    Conditional Invertible Neural Network.
    """
    def __init__(self, input_dim, condition_dim, num_affine_coupling_layers,
                 scale_hidden_layers, scale_activations, scale_output_activation,
                 translate_hidden_layers, translate_activations, translate_output_activation):
        super(cINN, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_affine_coupling_layers):
            self.layers.append(
                AffineCouplingLayer(
                    input_dim=input_dim,
                    condition_dim=condition_dim,
                    scale_hidden_layers=scale_hidden_layers,
                    scale_activations=scale_activations,
                    scale_output_activation=scale_output_activation,
                    translate_hidden_layers=translate_hidden_layers,
                    translate_activations=translate_activations,
                    translate_output_activation=translate_output_activation
                )
            )
            self.layers.append(FlipLayer())  # Alternates which half is transformed

    def forward(self, x, y):
        log_det_jacobian = 0
        for layer in self.layers:
            if isinstance(layer, AffineCouplingLayer):
                x, ldj = layer(x, y)
                log_det_jacobian += ldj
            else:
                x = layer(x)
        return x, log_det_jacobian

    def inverse(self, z, y):
        for layer in reversed(self.layers):
            if isinstance(layer, AffineCouplingLayer):
                z, _ = layer.inverse(z, y)
            else:
                z = layer(z)
        return z

########################################
# Training the cINN Model
########################################

# Initialize the model with configurable Scale and Translate networks
model = cINN(
    input_dim=INPUT_DIM,
    condition_dim=CONDITION_DIM,
    num_affine_coupling_layers=NUM_AFFINE_COUPLING_LAYERS,
    scale_hidden_layers=SCALE_TRANSLATE_HIDDEN_LAYERS,
    scale_activations=SCALE_TRANSLATE_ACTIVATIONS,
    scale_output_activation=SCALE_OUTPUT_ACTIVATION,
    translate_hidden_layers=SCALE_TRANSLATE_HIDDEN_LAYERS,
    translate_activations=SCALE_TRANSLATE_ACTIVATIONS,
    translate_output_activation=TRANSLATE_OUTPUT_ACTIVATION
)

# Move model to the specified device
model.to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Loss function as per Equation 7: L(z) = 0.5 * ||z||^2 - log|det(J_{x->z})|
def maximum_likelihood_loss(z, log_det_jacobian):
    return 0.5 * torch.sum(z ** 2, dim=1) - log_det_jacobian

# Training loop
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss_epoch = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        z, log_det_jacobian = model(x_batch, y_batch)
        loss = maximum_likelihood_loss(z, log_det_jacobian).mean()
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item() * x_batch.size(0)
    train_loss_epoch /= len(train_loader.dataset)
    train_losses.append(train_loss_epoch)

    # Validation phase
    model.eval()
    val_loss_epoch = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            z, log_det_jacobian = model(x_batch, y_batch)
            loss = maximum_likelihood_loss(z, log_det_jacobian).mean()
            val_loss_epoch += loss.item() * x_batch.size(0)
    val_loss_epoch /= len(val_loader.dataset)
    val_losses.append(val_loss_epoch)

    # Print progress
    if (epoch + 1) % 25 == 0:
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}')

########################################
# Plotting the Loss Curves and Saving the Figures
########################################

epochs = np.arange(1, NUM_EPOCHS + 1)

# Plot for the first 10 epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs[:10], train_losses[:10], label='Training Loss')
plt.plot(epochs[:10], val_losses[:10], label='Validation Loss')
plt.title('Training and Validation Loss (First 10 Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('cinn-v1_first10.png')  # Save the figure as "cinn-v1_first10.png"
plt.show()

# Plot for the remaining epochs
if NUM_EPOCHS > 10:
    plt.figure(figsize=(10, 5))
    plt.plot(epochs[10:], train_losses[10:], label='Training Loss')
    plt.plot(epochs[10:], val_losses[10:], label='Validation Loss')
    plt.title(f'Training and Validation Loss (Epochs 11 to {NUM_EPOCHS})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('cinn-v1_remaining.png')  # Save the figure as "cinn-v1_remaining.png"
    plt.show()

########################################
# Inference (Solving the Inverse Problem)
########################################

model.eval()
all_true_positions = []
all_predicted_positions = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        y_batch = y_batch.to(DEVICE)
        x_batch = x_batch.to(DEVICE)
        # Sample latent variables (z) from a standard Gaussian distribution
        z = torch.randn_like(x_batch).to(DEVICE)  # Ensure z matches the input dimension
        # Pass z and y through the inverse mapping to obtain predicted positions
        x_pred = model.inverse(z, y_batch)
        all_true_positions.append(x_batch.cpu())
        all_predicted_positions.append(x_pred.cpu())

# Concatenate all batches
all_true_positions = torch.cat(all_true_positions, dim=0).numpy()
all_predicted_positions = torch.cat(all_predicted_positions, dim=0).numpy()

########################################
# Post-processing and Evaluation
########################################

# Denormalize the data
if pos_scaler:
    true_positions_denorm = pos_scaler.inverse_transform(all_true_positions)
    predicted_positions_denorm = pos_scaler.inverse_transform(all_predicted_positions)
else:
    true_positions_denorm = all_true_positions
    predicted_positions_denorm = all_predicted_positions

# Calculate MSE on the original scale
mse = np.mean((true_positions_denorm - predicted_positions_denorm) ** 2)
print(f'Mean Squared Error on Test Set (Original Scale): {mse:.4f}')

# ----------------------------------------
# Relative Error Calculation
# ----------------------------------------

# To avoid division by zero, add a small epsilon to the denominator
epsilon = 1e-8
relative_error = np.abs(true_positions_denorm - predicted_positions_denorm) / (np.abs(true_positions_denorm) + epsilon)

# Original method: Mean over all elements
mean_relative_error = np.mean(relative_error)
print(f'Mean Relative Error on Test Set (Original Method): {mean_relative_error:.4f}')

# Alternative method: Mean over components, then over samples
per_sample_relative_error = np.mean(relative_error, axis=1)  # Mean over components for each sample
mean_relative_error_alt = np.mean(per_sample_relative_error)  # Mean over all samples
print(f'Mean Relative Error on Test Set (Alternative Method): {mean_relative_error_alt:.4f}')

# Verify that both methods give the same result (within numerical precision)
difference = np.abs(mean_relative_error - mean_relative_error_alt)
'''print(f'Difference between methods: {difference:.10f}')'''

# ----------------------------------------
# Additional Statistics
# ----------------------------------------

# Median and Standard Deviation of Relative Error
median_relative_error = np.median(relative_error)
std_relative_error = np.std(relative_error)

'''print(f'Median Relative Error on Test Set: {median_relative_error:.4f}')
print(f'Standard Deviation of Relative Error on Test Set: {std_relative_error:.4f}')
'''
# Relative Error per Dimension
relative_error_per_dimension = np.mean(relative_error, axis=0)
for idx, column in enumerate(POSITION_COLUMNS):
    print(f'Mean Relative Error for {column}: {relative_error_per_dimension[idx]:.4f}')

'''# ----------------------------------------
# Save Relative Errors to a CSV
# ----------------------------------------

# Create a DataFrame for relative errors
relative_error_df = pd.DataFrame(relative_error, columns=POSITION_COLUMNS)

# Save to CSV
relative_error_df.to_csv('relative_errors_test_set.csv', index=False)
print('Relative errors have been saved to "relative_errors_test_set.csv".')'''
