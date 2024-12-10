import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import json
import torchsummary
import torchinfo

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/cei_traning_orient_1.csv'
data = pd.read_csv(FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Define K-Fold Cross Validator
k = 5
seed = 42
kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

# Initialize list to store results
results = []

# Define hyperparameters (unchanged from original code)
INPUT_DIM = position.shape[1]  # 9
OUTPUT_DIM = momenta.shape[1]  # 9

LATENT_DIM = 1414
EPOCHS = 192
BATCH_SIZE = 256
LEARNING_RATE = 6.4358632430186e-06
PATIENCE = 17
MIN_DELTA = 2.091387832095796e-05
activation_name = 'Sigmoid'
position_norm_method = 'StandardScaler'
momenta_norm_method = 'StandardScaler'
use_l1 = True
L1_LAMBDA = 0.00014087065777225403
num_hidden_layers = 2
hidden_layer_size = 307

# Activation function
if activation_name == 'LeakyReLU':
    activation_function = nn.LeakyReLU()
else:
    activation_function = getattr(nn, activation_name)()

# Define the CVAE model with the specified hyperparameters
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_layers, activation_function):
        super(CVAE, self).__init__()

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_layers[0]))
        encoder_layers.append(activation_function)
        for i in range(len(hidden_layers) - 1):
            encoder_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            encoder_layers.append(activation_function)
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(hidden_layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[-1], latent_dim)

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim + condition_dim, hidden_layers[-1]))
        decoder_layers.append(activation_function)
        for i in reversed(range(len(hidden_layers) - 1)):
            decoder_layers.append(nn.Linear(hidden_layers[i+1], hidden_layers[i]))
            decoder_layers.append(activation_function)
        decoder_layers.append(nn.Linear(hidden_layers[0], input_dim))
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

# Define the loss function with L1 regularization if applicable
def loss_fn(recon_x, x, mu, logvar, model):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_divergence /= x.size(0) * x.size(1)
    loss = recon_loss + kl_divergence
    if use_l1:
        l1_loss = 0.0
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        loss += L1_LAMBDA * l1_loss
    return loss

# Iterate over each fold
for fold, (train_indices, test_indices) in enumerate(kfold.split(position), 1):
    print(f"\n=== Fold {fold} ===")

    # Split data into train and test
    train_position, test_position = position[train_indices], position[test_indices]
    train_momenta, test_momenta = momenta[train_indices], momenta[test_indices]

    # Further split train into train and validation (approximately 87.5% train, 12.5% val)
    # Since the original split was 70% train, 15% val, 15% test out of 100%
    # Now, we have 80% train (per fold) and 20% test
    # To maintain the same ratio within the training set:
    # Train: 70/80 = 87.5%, Val: 15/80 = 18.75%
    train_size = 0.875
    val_size = 0.125

    train_pos, val_pos, train_mom, val_mom = train_test_split(
        train_position, train_momenta, train_size=train_size, test_size=val_size,
        random_state=seed, shuffle=True
    )

    # Convert to PyTorch tensors
    train_position_tensor = torch.FloatTensor(train_pos).to(device)
    val_position_tensor = torch.FloatTensor(val_pos).to(device)
    test_position_tensor = torch.FloatTensor(test_position).to(device)
    train_momenta_tensor = torch.FloatTensor(train_mom).to(device)
    val_momenta_tensor = torch.FloatTensor(val_mom).to(device)
    test_momenta_tensor = torch.FloatTensor(test_momenta).to(device)  # Corrected variable name

    # Normalization
    if position_norm_method == 'StandardScaler':
        position_scaler = StandardScaler()
    elif position_norm_method == 'MinMaxScaler':
        position_scaler = MinMaxScaler()
    else:
        position_scaler = None

    if momenta_norm_method == 'StandardScaler':
        momenta_scaler = StandardScaler()
    elif momenta_norm_method == 'MinMaxScaler':
        momenta_scaler = MinMaxScaler()
    else:
        momenta_scaler = None

    # Fit scaler on training data and transform
    if position_scaler is not None:
        train_position_norm = torch.FloatTensor(position_scaler.fit_transform(train_position_tensor.cpu())).to(device)
        val_position_norm = torch.FloatTensor(position_scaler.transform(val_position_tensor.cpu())).to(device)
        test_position_norm = torch.FloatTensor(position_scaler.transform(test_position_tensor.cpu())).to(device)
    else:
        train_position_norm = train_position_tensor
        val_position_norm = val_position_tensor
        test_position_norm = test_position_tensor

    if momenta_scaler is not None:
        train_momenta_norm = torch.FloatTensor(momenta_scaler.fit_transform(train_momenta_tensor.cpu())).to(device)
        val_momenta_norm = torch.FloatTensor(momenta_scaler.transform(val_momenta_tensor.cpu())).to(device)
        test_momenta_norm = torch.FloatTensor(momenta_scaler.transform(test_momenta_tensor.cpu())).to(device)
    else:
        train_momenta_norm = train_momenta_tensor
        val_momenta_norm = val_momenta_tensor
        test_momenta_norm = test_momenta_tensor

    # Hidden layers configuration
    hidden_layers = [hidden_layer_size // (2 ** i) for i in range(num_hidden_layers)]

    # Initialize the model
    model = CVAE(INPUT_DIM, LATENT_DIM, OUTPUT_DIM, hidden_layers, activation_function).to(device)

    # Define the optimizer with no weight decay since use_l2 is False
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

    # Create DataLoaders with the specified BATCH_SIZE
    train_loader = DataLoader(TensorDataset(train_position_norm, train_momenta_norm), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_position_norm, val_momenta_norm), batch_size=BATCH_SIZE, shuffle=False)

    # Initialize lists to store training and validation losses
    train_losses = []
    val_losses = []

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = f'best_model_fold{fold}.pth'
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_x, batch_cond in train_loader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x, batch_cond)
            loss = loss_fn(recon_x, batch_x, mu, logvar, model)
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_cond in val_loader:
                recon_x, mu, logvar = model(batch_x, batch_cond)
                loss = loss_fn(recon_x, batch_x, mu, logvar, model)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the model
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Load the best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print(f"Best model not found for fold {fold}. Using current model.")

    # Compute MRE on the test set
    model.eval()
    test_predictions = []
    with torch.no_grad():
        # Compute latent stats from training set
        mu_list = []
        logvar_list = []
        z_train_list = []
        for batch_x, batch_cond in train_loader:
            mu, logvar = model.encode(batch_x)
            z = model.reparameterize(mu, logvar)
            mu_list.append(mu)
            logvar_list.append(logvar)
            z_train_list.append(z)
        mu_train = torch.cat(mu_list, dim=0)
        logvar_train = torch.cat(logvar_list, dim=0)
        z_train = torch.cat(z_train_list, dim=0)
        mu_train_mean = mu_train.mean(dim=0)
        mu_train_std = mu_train.std(dim=0)

        # Save latent stats
        latent_stats_path = f'latent_stats_fold{fold}.pt'
        torch.save({'mu_train_mean': mu_train_mean.cpu(), 'mu_train_std': mu_train_std.cpu()}, latent_stats_path)

        # Sample z and decode
        z_sample = torch.randn(len(test_momenta_norm), LATENT_DIM).to(device)
        z_sample = z_sample * mu_train_std.unsqueeze(0) + mu_train_mean.unsqueeze(0)
        for i in range(len(test_momenta_norm)):
            cond = test_momenta_norm[i].unsqueeze(0)
            pred = model.decode(z_sample[i].unsqueeze(0), cond)
            test_predictions.append(pred)
    test_predictions = torch.cat(test_predictions, dim=0)

    # Inverse transform
    if position_norm_method:
        test_predictions_inv = position_scaler.inverse_transform(test_predictions.cpu().numpy())
        test_position_inv = position_scaler.inverse_transform(test_position_tensor.cpu().numpy())
    else:
        test_predictions_inv = test_predictions.cpu().numpy()
        test_position_inv = test_position_tensor.cpu().numpy()

    # Calculate MRE
    relative_errors = np.abs(test_predictions_inv - test_position_inv) / (np.abs(test_position_inv) + 1e-8)
    mre = np.mean(relative_errors)

    print(f"Test MRE for Fold {fold}: {mre:.6f}")

    # Append the result with 'mre' converted to float
    results.append({
        'fold': fold,
        'mre': float(mre)  # Conversion to native Python float
    })

    # Save the MRE to a JSON file for this fold
    run_results = {
        'fold': fold,
        'mre': float(mre)
    }
    run_results_path = f'results_fold{fold}.json'
    with open(run_results_path, 'w') as f:
        json.dump(run_results, f, indent=4)

    # Plotting Learning Curves
    # Ensure that matplotlib does not try to open any window
    plt.switch_backend('Agg')

    # Plot first 10 epochs
    plt.figure(figsize=(10, 6))
    epochs_first = range(1, min(11, len(train_losses)+1))
    plt.plot(epochs_first, train_losses[:10], label='Train Loss')
    plt.plot(epochs_first, val_losses[:10], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train & Val Loss - First 10 Epochs (Fold {fold})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'first10_fold{fold}.png')
    plt.close()

    # Plot remaining epochs
    if len(train_losses) > 10:
        plt.figure(figsize=(10, 6))
        epochs_rest = range(11, len(train_losses)+1)
        plt.plot(epochs_rest, train_losses[10:], label='Train Loss')
        plt.plot(epochs_rest, val_losses[10:], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Train & Val Loss - Remaining Epochs (Fold {fold})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'rest_epochs_fold{fold}.png')
        plt.close()

    # Model Summaries
    print(f"\n--- Model Summary using torchsummary for Fold {fold} ---")
    try:
        torchsummary.summary(model, input_size=[(INPUT_DIM,), (OUTPUT_DIM,)])
    except Exception as e:
        print(f"torchsummary failed: {e}")

    print(f"\n--- Model Summary using torchinfo for Fold {fold} ---")
    try:
        x_dummy = torch.randn(1, INPUT_DIM).to(device)
        cond_dummy = torch.randn(1, OUTPUT_DIM).to(device)
        torchinfo.summary(model, input_data=(x_dummy, cond_dummy), device=device)
    except Exception as e:
        print(f"torchinfo failed: {e}")

# Save all results to a single JSON file
with open('all_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n=== All Folds Completed ===")
print(f"Total MREs computed: {len(results)}")
for res in results:
    print(f"Fold: {res['fold']} | MRE: {res['mre']:.6f}")

# Optionally, save results as a CSV
results_df = pd.DataFrame(results)
results_df.to_csv('all_results.csv', index=False)
print("\nAll results have been saved to 'all_results.json' and 'all_results.csv'.")
