import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

########################################
# Configuration and Hyperparameters
########################################

# ----------------------------
# Random Seed for Reproducibility
# ----------------------------
RANDOM_SEED = 0

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
# DataLoader Parameters
# ----------------------------
BATCH_SIZE = 2048
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False

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
TRANSLATE_OUTPUT_ACTIVATION = 'ELU'  # No activation for translate_net output

# ----------------------------
# Training Parameters
# ----------------------------
NUM_EPOCHS = 75
LEARNING_RATE = 1e-3
K_FOLDS = 5  # Number of folds for cross-validation

# ----------------------------
# Plotting Parameters
# ----------------------------
PLOT_FIRST_EPOCHS = 20  # Number of epochs to plot in the first figure

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

# Convert to tensors
positions_tensor = torch.tensor(positions, dtype=torch.float32)
momenta_tensor = torch.tensor(momenta, dtype=torch.float32)

# Create dataset
dataset = TensorDataset(positions_tensor, momenta_tensor)

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
        return z  # Only z is returned

########################################
# Cross-Validation Setup
########################################

# Initialize KFold
kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# To store MSE and Relative Error for each fold
mse_folds = []
relative_error_folds = []

# Enumerate the splits
for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
    print(f'\nFold {fold + 1}/{K_FOLDS}')
    print('--------------------------------')

    # Sample elements according to the indices
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Extract train and validation data
    train_positions = train_subset[:][0].numpy()
    train_momenta = train_subset[:][1].numpy()

    val_positions = val_subset[:][0].numpy()
    val_momenta = val_subset[:][1].numpy()

    # ----------------------------
    # Normalization
    # ----------------------------
    pos_scaler = StandardScaler() if NORMALIZE_POSITIONS else None
    mom_scaler = StandardScaler() if NORMALIZE_MOMENTA else None

    # Fit scalers on training data
    positions_norm_train = pos_scaler.fit_transform(train_positions) if pos_scaler else train_positions
    momenta_norm_train = mom_scaler.fit_transform(train_momenta) if mom_scaler else train_momenta

    # Transform validation data using the same scalers
    positions_norm_val = pos_scaler.transform(val_positions) if pos_scaler else val_positions
    momenta_norm_val = mom_scaler.transform(val_momenta) if mom_scaler else val_momenta

    # Convert to tensors
    positions_tensor_train = torch.tensor(positions_norm_train, dtype=torch.float32)
    momenta_tensor_train = torch.tensor(momenta_norm_train, dtype=torch.float32)

    positions_tensor_val = torch.tensor(positions_norm_val, dtype=torch.float32)
    momenta_tensor_val = torch.tensor(momenta_norm_val, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset_fold = TensorDataset(positions_tensor_train, momenta_tensor_train)
    val_dataset_fold = TensorDataset(positions_tensor_val, momenta_tensor_val)

    # Create DataLoaders
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, shuffle=SHUFFLE_VAL)

    ########################################
    # Initialize the cINN Model
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
        for x_batch, y_batch in train_loader_fold:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            z, log_det_jacobian = model(x_batch, y_batch)
            loss = maximum_likelihood_loss(z, log_det_jacobian).mean()
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * x_batch.size(0)
        train_loss_epoch /= len(train_loader_fold.dataset)
        train_losses.append(train_loss_epoch)

        # Validation phase
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader_fold:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                z, log_det_jacobian = model(x_batch, y_batch)
                loss = maximum_likelihood_loss(z, log_det_jacobian).mean()
                val_loss_epoch += loss.item() * x_batch.size(0)
        val_loss_epoch /= len(val_loader_fold.dataset)
        val_losses.append(val_loss_epoch)

        # Print progress every 10 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss_epoch:.2f}, Val Loss: {val_loss_epoch:.2f}')

    ########################################
    # Evaluation on Validation Set
    ########################################

    model.eval()
    all_true_positions = []
    all_predicted_positions = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader_fold:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # Sample latent variables (z) from a standard Gaussian distribution
            z = torch.randn(x_batch.size(0), INPUT_DIM).to(DEVICE)  # Ensure z matches the input dimension

            # Pass z and y through the inverse mapping to obtain predicted positions
            x_pred = model.inverse(z, y_batch)  # Fixed: Removed unpacking

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

    # Calculate Relative Error
    epsilon = 1e-8  # To prevent division by zero
    relative_error = np.abs(true_positions_denorm - predicted_positions_denorm) / (np.abs(true_positions_denorm) + epsilon)
    mean_relative_error = np.mean(relative_error)  # Average over all samples and dimensions

    print(f'Fold {fold + 1} MSE: {mse:.2f}')
    print(f'Fold {fold + 1} Mean Relative Error: {mean_relative_error:.2f}')
    mse_folds.append(mse)
    relative_error_folds.append(mean_relative_error)

    ########################################
    # Optional: Resetting GPU Memory (if needed)
    ########################################
    del model
    torch.cuda.empty_cache()

########################################
# Cross-Validation Results
########################################

average_mse = np.mean(mse_folds)
std_mse = np.std(mse_folds)
average_relative_error = np.mean(relative_error_folds)
std_relative_error = np.std(relative_error_folds)

print('\n========================================')
print(f'Cross-Validation Results over {K_FOLDS} Folds:')
for i, (mse, rel_err) in enumerate(zip(mse_folds, relative_error_folds)):
    print(f'Fold {i+1}: MSE = {mse:.2f}, Mean Relative Error = {rel_err:.2f}')
print(f'Average MSE: {average_mse:.2f} ± {std_mse:.2f}')
print(f'Average Mean Relative Error: {average_relative_error:.2f} ± {std_relative_error:.2f}')
print('========================================\n')
