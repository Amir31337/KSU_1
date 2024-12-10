import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Configuration settings
file_path = '/home/g/ghanaatian/MYFILES/FALL24/Physics/Fifth/cei_traning_orient_1.csv'

# General settings
lambda_mse = 0.1
grad_clamp = 15
eval_test = 10  # Evaluate every 10 epochs
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training schedule
lr_init = 1.0e-3
batch_size = 500
n_epochs = 1000
pre_low_lr = 0
final_decay = 0.02
l2_weight_reg = 1e-5
adam_betas = (0.9, 0.95)

# Data dimensions
ndim_x = 9  # Initial positions (cx, cy, cz, ox, oy, oz, sx, sy, sz)
ndim_y = 9  # Final momenta (pcx, pcy, pcz, pox, poy, poz, psx, psy, psz)
ndim_z = 0  # Latent dimensions (if any, set to 0 if not used)
ndim_pad_zy = 0  # Padding dimensions (if any)

# Training flags
train_forward_mmd = False
train_backward_mmd = False
train_reconstruction = False
train_max_likelihood = True

# Lambda values
lambd_fit_forw = 0
lambd_mmd_forw = 0
lambd_reconstruct = 0
lambd_mmd_back = 0
lambd_max_likelihood = 1

# Noise parameters
add_y_noise = 5e-3
add_z_noise = 2e-3
add_pad_noise = 1e-3
zeros_noise_scale = 5e-3

# MMD parameters
y_uncertainty_sigma = 0.12 * 4
mmd_forw_kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]
mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
mmd_back_weighted = False

# Model parameters
init_scale = 0.10
N_blocks = 6
exponent_clamping = 2.0
hidden_layer_sizes = 256
use_permutation = True
verbose_construction = False

# Data loading function
def load_data(file_path=file_path, batch_size=batch_size, test_size=0.2, random_state=42):
    """
    Loads the dataset from a CSV file, splits it into training and testing sets,
    and creates DataLoader objects for each.
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Extract positions and momenta
    position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
    momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values
    
    # Split into training and testing sets
    x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
        position, momenta, test_size=test_size, random_state=random_state
    )
    
    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    x_test = torch.tensor(x_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)
    
    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size, shuffle=False, drop_last=True
    )
    
    return train_loader, test_loader

# Create data loaders
train_loader, test_loader = load_data()