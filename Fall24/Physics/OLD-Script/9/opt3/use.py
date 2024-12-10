import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    Normalizer,
    MaxAbsScaler,
    PowerTransformer,
    QuantileTransformer
)
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
import os
import numpy as np
import pickle  # For saving scalers and latent representations

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================== #
#         Configuration          #
# ============================== #

# Data Configuration
DATA_PATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
INPUT_POSITION_COLUMNS = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
INPUT_MOMENTUM_COLUMNS = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fixed Training Parameters
BATCH_SIZE = 4096  # Reduced to manage memory better
NUM_EPOCHS = 50

# Path to save the best model
BEST_MODEL_PATH = "best_model.pth"
BEST_SCALERS_PATH = "scalers.pkl"
TRAIN_LATENTS_PATH = "train_latents.pkl"  # New path to save training latent representations

# ============================== #
#          Training Function      #
# ============================== #

def train_and_save_model(best_params):
    # Extract hyperparameters
    learning_rate = best_params['LEARNING_RATE']
    weight_decay = best_params['WEIGHT_DECAY']
    lr_step_size = best_params['LR_STEP_SIZE']
    lr_gamma = best_params['LR_GAMMA']
    num_coupling_blocks = best_params['NUM_COUPLING_BLOCKS']
    hidden_dim = best_params['HIDDEN_DIM']
    clamping_value = best_params['CLAMPING_VALUE']
    activation_name = best_params['ACTIVATION_FN']
    scaler_choice = best_params['SCALER']

    # Handle activation functions with additional parameters
    if activation_name == 'LeakyReLU':
        negative_slope = best_params.get('LEAKYRELU_NEGATIVE_SLOPE', 0.01)
        activation_fn = nn.LeakyReLU(negative_slope=negative_slope)
    elif activation_name == 'PReLU':
        num_parameters = best_params.get('PReLU_NUM_PARAMETERS', 1)
        activation_fn = nn.PReLU(num_parameters=num_parameters)
    else:
        activation_fn = getattr(nn, activation_name)()

    # Handle scalers with additional parameters
    if scaler_choice == 'PowerTransformer':
        method = best_params.get('POWER_TRANSFORMER_METHOD', 'yeo-johnson')
        pos_scaler = PowerTransformer(method=method)
        mom_scaler = PowerTransformer(method=method)
    elif scaler_choice == 'QuantileTransformer':
        n_quantiles = best_params.get('QUANTILE_TRANSFORMER_N_QUANTILES', 100)
        output_distribution = best_params.get('QUANTILE_TRANSFORMER_OUTPUT_DIST', 'uniform')
        pos_scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution, random_state=42)
        mom_scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution, random_state=42)
    else:
        # Default scalers
        if scaler_choice == 'MinMaxScaler':
            pos_scaler = MinMaxScaler()
            mom_scaler = MinMaxScaler()
        elif scaler_choice == 'StandardScaler':
            pos_scaler = StandardScaler()
            mom_scaler = StandardScaler()
        elif scaler_choice == 'RobustScaler':
            pos_scaler = RobustScaler()
            mom_scaler = RobustScaler()
        elif scaler_choice == 'Normalizer':
            pos_scaler = Normalizer()
            mom_scaler = Normalizer()
        elif scaler_choice == 'MaxAbsScaler':
            pos_scaler = MaxAbsScaler()
            mom_scaler = MaxAbsScaler()
        else:
            raise ValueError("Unknown scaler choice.")

    # Load data
    data = pd.read_csv(DATA_PATH)

    # Extract positions (X) and momenta (Y)
    positions = data[INPUT_POSITION_COLUMNS].values
    momenta = data[INPUT_MOMENTUM_COLUMNS].values

    # Check for NaNs in the data
    if np.isnan(positions).any() or np.isnan(momenta).any():
        raise ValueError("Input data contains NaNs.")

    # Convert to PyTorch tensors
    positions = torch.tensor(positions, dtype=torch.float32)
    momenta = torch.tensor(momenta, dtype=torch.float32)

    # Normalize the data
    try:
        positions_norm = torch.tensor(pos_scaler.fit_transform(positions), dtype=torch.float32)
        momenta_norm = torch.tensor(mom_scaler.fit_transform(momenta), dtype=torch.float32)
    except ValueError as e:
        raise ValueError(f"Scaler failed: {e}")

    # Check for NaNs after normalization
    if torch.isnan(positions_norm).any() or torch.isnan(momenta_norm).any():
        raise ValueError("Normalized data contains NaNs.")

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
            nn.Linear(dims_in, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
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
    cinn = Ff.ReversibleGraphNet(nodes + [cond], verbose=False).to(DEVICE)

    # ============================== #
    #          Loss Function         #
    # ============================== #

    def cinn_loss(z, log_jacob_det):
        nll = 0.5 * torch.sum(z ** 2, dim=1) - log_jacob_det
        return torch.mean(nll)

    # ============================== #
    #          Optimizer             #
    # ============================== #

    optimizer = torch.optim.Adam(cinn.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    # ============================== #
    #          Training Loop         #
    # ============================== #

    best_val_loss = float('inf')
    patience = 10  # Early stopping patience
    counter = 0

    # Implement gradient clipping to prevent exploding gradients
    GRAD_CLIP_VALUE = 1.0

    # Initialize a list to store training latent representations
    train_latents = []

    for epoch in range(NUM_EPOCHS):
        # Training Phase
        cinn.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            z, log_jacob_det = cinn(x_batch, c=[y_batch])
            loss = cinn_loss(z, log_jacob_det)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(cinn.parameters(), GRAD_CLIP_VALUE)

            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)

            # Store the latent representations
            train_latents.append(z.detach().cpu())

        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation Phase
        cinn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                z, log_jacob_det = cinn(x_batch, c=[y_batch])
                loss = cinn_loss(z, log_jacob_det)
                val_loss += loss.item() * x_batch.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            # Save the best model
            torch.save({
                'model_state_dict': cinn.state_dict(),
                'pos_scaler': pos_scaler,
                'mom_scaler': mom_scaler,
                'hyperparameters': best_params
            }, BEST_MODEL_PATH)
            # Save scalers separately if needed
            with open(BEST_SCALERS_PATH, 'wb') as f:
                pickle.dump({
                    'pos_scaler': pos_scaler,
                    'mom_scaler': mom_scaler
                }, f)
            # Save the training latent representations
            with open(TRAIN_LATENTS_PATH, 'wb') as f:
                pickle.dump(torch.cat(train_latents, dim=0), f)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Step the scheduler
        scheduler.step()

        # Print training progress
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    print(f"\nBest Validation Loss: {best_val_loss}")
    print(f"Model saved to {BEST_MODEL_PATH}")
    print(f"Scalers saved to {BEST_SCALERS_PATH}")
    print(f"Training Latent Representations saved to {TRAIN_LATENTS_PATH}")

    # ============================== #
    #          Evaluation             #
    # ============================== #

    # Load the best model
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    cinn.load_state_dict(checkpoint['model_state_dict'])
    cinn.eval()

    # Load the training latent representations
    with open(TRAIN_LATENTS_PATH, 'rb') as f:
        train_latents = pickle.load(f)
    train_latents = train_latents.to(DEVICE)

    with torch.no_grad():
        x_trues = []
        x_preds = []
        for _, y_batch in test_loader:
            y_batch = y_batch.to(DEVICE)
            # Use the stored training latent representations instead of sampling new z's
            # To align the number of latent vectors with the test set, you can sample randomly or iterate accordingly
            # Here, we'll randomly select z's from the training latent representations
            indices = torch.randint(0, train_latents.size(0), (y_batch.size(0),))
            z_sample = train_latents[indices]

            x_pred_batch, _ = cinn(z_sample, c=[y_batch], rev=True)
            x_preds.append(x_pred_batch.cpu())

        # To compute MSE, you still need the true positions from the test set
        # Load test positions separately without using them in the model's inference
        test_positions = torch.stack([batch[0] for batch in test_loader], dim=0).view(-1, input_dim).numpy()
        x_pred = torch.cat(x_preds, dim=0).numpy()

        # Inverse transform
        try:
            if scaler_choice in ['Normalizer']:
                # Normalizer does not support inverse_transform
                x_pred_inv = x_pred
                x_true_inv = test_positions  # Assuming test_positions are already normalized with Normalizer
            else:
                x_pred_inv = pos_scaler.inverse_transform(x_pred)
                x_true_inv = pos_scaler.inverse_transform(test_positions)
        except Exception as e:
            raise ValueError(f"Inverse transform failed: {e}")

        # Check for NaNs in predictions or true values
        if np.isnan(x_pred_inv).any() or np.isnan(x_true_inv).any():
            raise ValueError("Inverse transformed data contains NaNs.")

        mse = mean_squared_error(x_true_inv, x_pred_inv)

    print(f"\nTest Mean Squared Error (MSE): {mse}")

# ============================== #
#          Main Script            #
# ============================== #

def main():
    # Define the best hyperparameters
    best_params = {
        'LEARNING_RATE': 8.603491278164522e-06,
        'WEIGHT_DECAY': 3.03265339804328e-08,
        'LR_STEP_SIZE': 17,
        'LR_GAMMA': 0.6558273506083766,
        'NUM_COUPLING_BLOCKS': 53,
        'HIDDEN_DIM': 1016,
        'CLAMPING_VALUE': 3.841677082940536,
        'ACTIVATION_FN': 'ReLU',
        'SCALER': 'Normalizer'
        # No additional parameters needed for ReLU and Normalizer
    }

    # Train the model with the best hyperparameters
    train_and_save_model(best_params)



if __name__ == "__main__":
    main()
