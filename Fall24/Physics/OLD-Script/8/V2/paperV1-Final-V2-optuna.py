# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from tqdm import tqdm
import os
import random
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import optuna

# Set random seeds for reproducibility
def set_seed(seed=42):
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

# Extract initial positions (X) and final momenta (Y) as DataFrames
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']]
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']]

# Normalize the data and keep them as DataFrames
scaler_position = StandardScaler()
scaler_momenta = StandardScaler()

normalized_position = pd.DataFrame(
    scaler_position.fit_transform(position),
    columns=position.columns,
    index=position.index
)
normalized_momenta = pd.DataFrame(
    scaler_momenta.fit_transform(momenta),
    columns=momenta.columns,
    index=momenta.index
)

# Split the data into train, validation, and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    normalized_position, normalized_momenta, test_size=0.3, random_state=42
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_test, Y_test, test_size=0.5, random_state=42
)

# Hyperparameters (initial values, will be optimized by Optuna)
HIDDEN_FEATURES = 256
NUM_BLOCKS = 6
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
LR_SCHEDULER_FACTOR = 0.5
TOP_K = 3
LOCALIZATION_LR = 1e-2
LOCALIZATION_STEPS = 100
N_JOBS = 4

# Define the CINN model architecture
def create_cinn_model(input_dim=9, condition_dim=9, hidden_features=256, num_blocks=6):
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
                        nn.ReLU(),
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

def compute_loss(model, x, y):
    z, log_jacobian = model(x, c=y)

    log_p_z = -0.5 * torch.sum(z ** 2, dim=1)
    log_p_x = log_p_z + log_jacobian
    loss = -torch.mean(log_p_x)
    return loss

# Define the validation function
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = compute_loss(model, batch_x, batch_y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def infer_initial_positions(model, y, scaler_position, scaler_momenta, num_samples=1):
    model.eval()
    with torch.no_grad():
        y = y.to(device)
        generated_X = []

        for _ in range(num_samples):
            z = torch.randn(y.size(0), 9, device=device)
            x_pred, _ = model(z, c=y, rev=True)
            x_pred = x_pred.cpu().numpy()
            # Convert x_pred to a DataFrame with the same columns
            x_pred_df = pd.DataFrame(x_pred, columns=position.columns)
            x_denorm = scaler_position.inverse_transform(x_pred_df)
            # x_denorm is already a NumPy array after inverse_transform
            generated_X.append(x_denorm)

        if num_samples > 1:
            generated_X = np.stack(generated_X, axis=1)
        else:
            generated_X = generated_X[0]

        return generated_X

# Define the down-selection function
def down_select_generated_positions(model, generated_X, original_Y, scaler_position, scaler_momenta, top_k=3):
    model.eval()
    with torch.no_grad():
        N, num_samples, _ = generated_X.shape
        selected_X = []

        for i in range(N):
            candidates = generated_X[i]
            # Convert candidates to DataFrame with the same columns
            candidates_df = pd.DataFrame(candidates, columns=position.columns)
            candidates_normalized = scaler_position.transform(candidates_df)
            candidates_tensor = torch.tensor(candidates_normalized, dtype=torch.float32).to(device)
            y_tensor = original_Y[i].unsqueeze(0).repeat(num_samples, 1).to(device)

            predicted_Y, _ = model(candidates_tensor, c=y_tensor)

            loss = torch.mean((predicted_Y - original_Y[i].to(device))**2, dim=1)
            topk_indices = torch.topk(-loss, top_k).indices
            selected = candidates[topk_indices.cpu().numpy()]
            selected_X.append(selected)

        selected_X = np.stack(selected_X, axis=0)
        return selected_X

def localize_single_sample(model, candidates, y, scaler_position, scaler_momenta, learning_rate=1e-2, num_steps=100):
    model.eval()
    top_k, _ = candidates.shape
    # Convert candidates to DataFrame with the same columns
    candidates_df = pd.DataFrame(candidates, columns=position.columns)
    candidates_normalized = scaler_position.transform(candidates_df)
    candidates_tensor = torch.tensor(candidates_normalized, dtype=torch.float32, requires_grad=True, device=device)
    y_tensor = y.unsqueeze(0).to(device)

    optimizer_loc = optim.Adam([candidates_tensor], lr=learning_rate)

    for step in range(num_steps):
        optimizer_loc.zero_grad()
        predicted_Y, _ = model(candidates_tensor, c=y_tensor.repeat(top_k, 1))
        loss = nn.MSELoss()(predicted_Y, y_tensor.repeat(top_k, 1))
        loss.backward()
        optimizer_loc.step()

    candidates_localized = candidates_tensor.detach().cpu().numpy()
    # Convert back to DataFrame to inverse transform
    candidates_localized_df = pd.DataFrame(candidates_localized, columns=position.columns)
    candidates_localized_denorm = scaler_position.inverse_transform(candidates_localized_df)
    return candidates_localized_denorm

def parallel_localize(model, generated_X, original_Y, scaler_position, scaler_momenta, learning_rate=1e-2, num_steps=100, n_jobs=4):
    localized_X = Parallel(n_jobs=n_jobs)(
        delayed(localize_single_sample)(
            model,
            generated_X[i],
            original_Y[i],
            scaler_position,
            scaler_momenta,
            learning_rate,
            num_steps
        ) for i in range(generated_X.shape[0])
    )
    return np.stack(localized_X, axis=0)

def train_and_evaluate(hidden_features, num_blocks, batch_size, learning_rate, weight_decay,
                       lr_scheduler_factor, top_k, localization_lr, localization_steps, n_jobs):
    # Convert DataFrames to NumPy arrays and then to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val.values, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).to(device)

    # Create DataLoader for batching
    train_dataset = data_utils.TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create validation DataLoader
    val_dataset = data_utils.TensorDataset(X_val_tensor, Y_val_tensor)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the CINN model
    input_dim = 9
    condition_dim = 9
    cinn_model = create_cinn_model(input_dim, condition_dim, hidden_features=hidden_features, num_blocks=num_blocks)
    cinn_model = cinn_model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(cinn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_scheduler_factor, patience=10)

    # Training parameters
    num_epochs = 300

    # Define the checkpoint directory
    checkpoint_dir = './checkpoints_optuna'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize the best validation loss
    best_val_loss = float('inf')

    # Training loop with progress bar
    train_losses = []
    val_losses = []
    prev_lr = optimizer.param_groups[0]['lr']
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    for epoch in range(1, num_epochs + 1):
        cinn_model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = compute_loss(cinn_model, batch_x, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Perform validation at the end of each epoch
        val_loss = validate(cinn_model, val_loader, device)
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
            try:
                torch.save(cinn_model.state_dict(), best_model_path)
                print(f'Best model saved at epoch {epoch} with Val Loss: {val_loss:.6f}')
            except Exception as e:
                print(f'Error saving best model at epoch {epoch}: {e}')

        # Save periodic checkpoints
        if epoch % 100 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'cinn_checkpoint_epoch_{epoch}.pth')
            torch.save(cinn_model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch}')

    # After training, plot the first 20 epochs in one image
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 21), train_losses[:20], label='Training Loss')
    plt.plot(range(1, 21), val_losses[:20], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Epochs 1-20)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot_first_20_epochs.png')
    plt.close()
    print('Training and validation loss plot saved as loss_plot_first_20_epochs.png')

    # Plot the remaining epochs (21 to num_epochs) in another image
    plt.figure(figsize=(8, 5))
    plt.plot(range(21, num_epochs + 1), train_losses[20:], label='Training Loss')
    plt.plot(range(21, num_epochs + 1), val_losses[20:], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (Epochs 21-{num_epochs})')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot_remaining_epochs.png')
    plt.close()
    print('Training and validation loss plot saved as loss_plot_remaining_epochs.png')

    # After training, ensure the best model is loaded
    if os.path.exists(best_model_path):
        try:
            cinn_model.load_state_dict(torch.load(best_model_path))
            print('Best model loaded successfully after training.')
        except Exception as e:
            print(f'Error loading best model: {e}')
    else:
        print(f'Best model not found at {best_model_path}. Evaluation will use the last epoch model.')

    # Generate predictions on the test set
    num_samples = 50  # Generate 50 samples per test instance for diversity
    predicted_positions = infer_initial_positions(
        model=cinn_model,
        y=Y_test_tensor,
        scaler_position=scaler_position,
        scaler_momenta=scaler_momenta,
        num_samples=num_samples
    )

    # Down-selection: Select top_k samples that best match the desired momenta
    down_selected_positions = down_select_generated_positions(
        model=cinn_model,
        generated_X=predicted_positions,
        original_Y=Y_test_tensor,
        scaler_position=scaler_position,
        scaler_momenta=scaler_momenta,
        top_k=top_k
    )

    num_test_samples = 2000

    localized_positions = parallel_localize(
        model=cinn_model,
        generated_X=down_selected_positions[:num_test_samples],
        original_Y=Y_test_tensor[:num_test_samples],
        scaler_position=scaler_position,
        scaler_momenta=scaler_momenta,
        learning_rate=localization_lr,
        num_steps=localization_steps,
        n_jobs=n_jobs  # Number of parallel jobs; adjust based on your CPU cores
    )

    # Calculate the mean of localized positions for each test sample
    predicted_X_single = np.mean(localized_positions, axis=1)

    # Evaluate the model on the test set in the original scale
    X_test_original = position.loc[X_test.index].values[:num_test_samples]
    predicted_X_single_original = predicted_X_single
    mse = mean_squared_error(X_test_original, predicted_X_single_original)
    print(f"Test MSE: {mse:.6f}")

    return mse

def objective(trial):
    # Sample hyperparameters
    hidden_features = trial.suggest_categorical('hidden_features', [64, 128, 256, 512])
    num_blocks = trial.suggest_int('num_blocks', 2, 8, step=2)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
    lr_scheduler_factor = trial.suggest_uniform('lr_scheduler_factor', 0.1, 0.9)
    top_k = trial.suggest_categorical('top_k', [1, 3, 5])
    localization_lr = trial.suggest_loguniform('localization_lr', 1e-3, 1e-1)
    localization_steps = trial.suggest_int('localization_steps', 10, 100, step=10)
    n_jobs = trial.suggest_categorical('n_jobs', [1, 2, 4, 8])

    # Now, call train_and_evaluate with these hyperparameters
    mse = train_and_evaluate(
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_factor=lr_scheduler_factor,
        top_k=top_k,
        localization_lr=localization_lr,
        localization_steps=localization_steps,
        n_jobs=n_jobs
    )

    return mse

# Train and evaluate the model using Optuna
if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print(f"  MSE: {trial.value}")
    print("  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
