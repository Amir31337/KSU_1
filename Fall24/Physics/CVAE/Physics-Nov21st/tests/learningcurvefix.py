import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import json
import warnings

# Suppress specific FutureWarnings related to torch.load for internal trusted files
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Set the data file path and base save directory at the beginning
FILEPATH = '/home/g/ghanaatian/MYFILES/FALL24/Physics/Data/sim_million_orient.csv'  
SAVE_DIR_BASE = '/home/g/ghanaatian/MYFILES/FALL24/Physics/CVAE/Physics-Nov21st/new/after'

# Define the list of experiments
experiments = [
    {
        'name': 'EXPERIMENT1',
        'use_l1': True,
        'l1_lambda': 0.0001,
        'use_l2': True,
        'l2_lambda': 0.01,
        'use_beta': True,
        'beta': 0.1,
        'LATENT_DIM': 256,
        'EPOCHS': 100,
        'BATCH_SIZE': 4096,
        'LEARNING_RATE': 5e-05,
        'PATIENCE': 20,
        'MIN_DELTA': 1e-04,
        'hidden_layer_size': 1024,
        'num_hidden_layers': 4,
        'activation_name': 'LeakyReLU',
        'position_norm_method': 'MinMaxScaler',
        'momenta_norm_method': 'MinMaxScaler',
    }
]

# Update experiments to include SAVE_DIR using SAVE_DIR_BASE
for i, experiment in enumerate(experiments, start=1):
    experiment['SAVE_DIR'] = os.path.join(SAVE_DIR_BASE, f'EXPERIMENT{i}')

# Load data
data = pd.read_csv(FILEPATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Split data into train, validation, and test sets (70%, 15%, 15%)
train_position_raw, temp_position_raw, train_momenta_raw, temp_momenta_raw = train_test_split(
    position, momenta, test_size=0.3, random_state=42
)

val_position_raw, test_position_raw, val_momenta_raw, test_momenta_raw = train_test_split(
    temp_position_raw, temp_momenta_raw, test_size=0.5, random_state=42
)

def run_experiment(config):
    # Unpack configuration
    use_l1 = config['use_l1']
    l1_lambda = config['l1_lambda']
    use_l2 = config['use_l2']
    l2_lambda = config['l2_lambda']
    use_beta = config['use_beta']
    beta = config['beta']
    LATENT_DIM = config['LATENT_DIM']
    EPOCHS = config['EPOCHS']
    BATCH_SIZE = config['BATCH_SIZE']
    LEARNING_RATE = config['LEARNING_RATE']
    PATIENCE = config['PATIENCE']
    MIN_DELTA = config['MIN_DELTA']
    hidden_layer_size = config['hidden_layer_size']
    num_hidden_layers = config['num_hidden_layers']
    activation_name = config['activation_name']
    position_norm_method = config['position_norm_method']
    momenta_norm_method = config['momenta_norm_method']
    SAVE_DIR = config['SAVE_DIR']
    activation_function = getattr(nn, activation_name)()
    hidden_layers = [hidden_layer_size * (2 ** i) for i in range(num_hidden_layers)]

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Position normalization
    if position_norm_method == 'StandardScaler':
        position_scaler = StandardScaler()
    elif position_norm_method == 'MinMaxScaler':
        position_scaler = MinMaxScaler()
    else:
        position_scaler = None

    if position_scaler is not None:
        train_position = position_scaler.fit_transform(train_position_raw)
        val_position = position_scaler.transform(val_position_raw)
        test_position = position_scaler.transform(test_position_raw)
    else:
        train_position = train_position_raw
        val_position = val_position_raw
        test_position = test_position_raw

    # Momenta normalization
    if momenta_norm_method == 'StandardScaler':
        momenta_scaler = StandardScaler()
    elif momenta_norm_method == 'MinMaxScaler':
        momenta_scaler = MinMaxScaler()
    else:
        momenta_scaler = None

    if momenta_scaler is not None:
        train_momenta = momenta_scaler.fit_transform(train_momenta_raw)
        val_momenta = momenta_scaler.transform(val_momenta_raw)
        test_momenta = momenta_scaler.transform(test_momenta_raw)
    else:
        train_momenta = momenta_raw
        val_momenta = momenta_raw
        test_momenta = momenta_raw

    # Convert to PyTorch tensors
    train_position = torch.FloatTensor(train_position)
    val_position = torch.FloatTensor(val_position)
    test_position = torch.FloatTensor(test_position)
    train_momenta = torch.FloatTensor(train_momenta)
    val_momenta = torch.FloatTensor(val_momenta)
    test_momenta = torch.FloatTensor(test_momenta)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    INPUT_DIM = train_position.shape[1]  # Should be 9
    OUTPUT_DIM = train_momenta.shape[1]  # Should be 9

    # Define the CVAE model
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

    # Loss function with beta scaling; regularization added only during training
    def cvae_loss_fn(recon_x, x, mu, logvar, model, training=True):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_divergence /= x.size(0) * x.size(1)  # Normalize by batch size and input dimensions

        if use_beta:
            kl_divergence *= beta

        primary_loss = recon_loss + kl_divergence
        reg_loss = torch.tensor(0., device=device)

        if training:
            # L1 regularization
            if use_l1:
                for param in model.parameters():
                    reg_loss += torch.norm(param, 1)
                reg_loss *= l1_lambda

        if torch.isnan(primary_loss + reg_loss):
            print("NaN detected in loss computation.")

        return primary_loss, reg_loss

    # Build the model
    cvae = CVAE(INPUT_DIM, LATENT_DIM, OUTPUT_DIM, hidden_layers, activation_function).to(device)

    # Optimizer with weight_decay for L2 regularization
    if use_l2:
        cvae_optimizer = optim.Adam(cvae.parameters(), lr=LEARNING_RATE, weight_decay=l2_lambda)
    else:
        cvae_optimizer = optim.Adam(cvae.parameters(), lr=LEARNING_RATE)

    # Determine if pin_memory should be enabled
    pin_memory = True if device.type == 'cuda' else False

    # Create DataLoaders
    train_dataset = TensorDataset(train_position, train_momenta)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=pin_memory)

    val_dataset = TensorDataset(val_position, val_momenta)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin_memory)

    test_dataset = TensorDataset(test_position, test_momenta)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin_memory)

    # Training loop with early stopping and mixed precision
    train_primary_losses = []
    train_reg_losses = []
    val_primary_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    model_saved = False  # Flag to check if model was saved

    # Initialize GradScaler from torch.amp
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

    for epoch in range(EPOCHS):
        cvae.train()
        train_primary_loss_epoch = 0
        train_reg_loss_epoch = 0
        batches_processed = 0  # To handle cases where all batches are skipped

        for batch_idx, (position_batch, momenta_batch) in enumerate(train_loader):
            position_batch = position_batch.to(device, non_blocking=True)
            momenta_batch = momenta_batch.to(device, non_blocking=True)

            cvae_optimizer.zero_grad()

            # Update autocast based on device type and scaler availability
            if device.type == 'cuda' and scaler is not None:
                autocast_context = torch.amp.autocast(device_type='cuda', enabled=True)
            else:
                autocast_context = torch.amp.autocast(enabled=False)

            with autocast_context:
                recon_position, mu, logvar = cvae(position_batch, momenta_batch)
                primary_loss, reg_loss = cvae_loss_fn(recon_position, position_batch, mu, logvar, cvae, training=True)
                total_loss = primary_loss + reg_loss

            if torch.isnan(total_loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx+1}. Skipping this batch.")
                continue  # Skip this batch

            # Backward pass and optimization
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(cvae_optimizer)
                nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=5.0)
                scaler.step(cvae_optimizer)
                scaler.update()
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=5.0)
                cvae_optimizer.step()

            train_primary_loss_epoch += primary_loss.item()
            train_reg_loss_epoch += reg_loss.item()
            batches_processed += 1

        # If no batches were processed, break the loop
        if batches_processed == 0:
            print(f"No valid batches in epoch {epoch+1}. Stopping training.")
            break

        train_primary_loss_epoch /= batches_processed
        train_reg_loss_epoch /= batches_processed
        train_primary_losses.append(train_primary_loss_epoch)
        train_reg_losses.append(train_reg_loss_epoch)

        # Validation
        cvae.eval()
        val_primary_loss_epoch = 0
        val_batches_processed = 0
        with torch.no_grad():
            for position_batch, momenta_batch in val_loader:
                position_batch = position_batch.to(device, non_blocking=True)
                momenta_batch = momenta_batch.to(device, non_blocking=True)

                # Update autocast based on device type and scaler availability
                if device.type == 'cuda' and scaler is not None:
                    autocast_context = torch.amp.autocast(device_type='cuda', enabled=True)
                else:
                    autocast_context = torch.amp.autocast(enabled=False)

                with autocast_context:
                    recon_position, mu, logvar = cvae(position_batch, momenta_batch)
                    primary_loss, _ = cvae_loss_fn(recon_position, position_batch, mu, logvar, cvae, training=False)
                    loss = primary_loss

                if torch.isnan(loss):
                    print(f"NaN loss detected during validation at epoch {epoch+1}. Skipping this batch.")
                    continue  # Skip this batch

                val_primary_loss_epoch += loss.item()
                val_batches_processed += 1

        if val_batches_processed > 0:
            val_primary_loss_epoch /= val_batches_processed
            val_primary_losses.append(val_primary_loss_epoch)
        else:
            val_primary_loss_epoch = float('inf')
            val_primary_losses.append(val_primary_loss_epoch)

        print(f'Epoch {epoch+1}/{EPOCHS}, Training Primary Loss: {train_primary_loss_epoch:.6f}, '
              f'Training Reg Loss: {train_reg_loss_epoch:.6f}, Validation Loss: {val_primary_loss_epoch:.6f}')

        # Early stopping and model saving
        if not np.isnan(val_primary_loss_epoch) and val_primary_loss_epoch < best_val_loss - MIN_DELTA:
            best_val_loss = val_primary_loss_epoch
            patience_counter = 0
            # Save the model
            torch.save(cvae.state_dict(), os.path.join(SAVE_DIR, 'best_cvae.pth'))
            model_saved = True
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Check if a model was saved
    if not model_saved:
        print("No valid model was saved during training.")
    else:
        # Load the best model
        cvae.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_cvae.pth'), map_location=device))

        # Compute and save Mean and Std of latent variables on training set
        cvae.eval()
        with torch.no_grad():
            mu_list = []
            logvar_list = []
            for position_batch, momenta_batch in train_loader:
                position_batch = position_batch.to(device, non_blocking=True)
                # Note: We are encoding the training positions to get the latent variables
                mu, logvar = cvae.encode(position_batch)
                mu_list.append(mu.cpu())
                logvar_list.append(logvar.cpu())

            mu_train = torch.cat(mu_list, dim=0)
            logvar_train = torch.cat(logvar_list, dim=0)

            # Compute mean and std of latent variables
            mu_train_mean = mu_train.mean(dim=0)
            mu_train_std = mu_train.std(dim=0)

            # Save mu_train_mean and mu_train_std
            torch.save({'mu_train_mean': mu_train_mean, 'mu_train_std': mu_train_std}, os.path.join(SAVE_DIR, 'latent_stats.pt'))

        # Plot losses
        # Plot Training Primary Loss and Validation Loss
        plt.figure()
        plt.plot(range(1, len(train_primary_losses) + 1), train_primary_losses, label='Training Primary Loss')
        plt.plot(range(1, len(val_primary_losses) + 1), val_primary_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Primary Loss')
        plt.legend()
        plt.title('Primary Loss Curves')
        plt.savefig(os.path.join(SAVE_DIR, 'primary_loss_curves.png'))
        plt.close()

        # Plot Training Regularization Loss
        plt.figure()
        plt.plot(range(1, len(train_reg_losses) + 1), train_reg_losses, label='Training Regularization Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Regularization Loss')
        plt.legend()
        plt.title('Regularization Loss Curve')
        plt.savefig(os.path.join(SAVE_DIR, 'reg_loss_curve.png'))
        plt.close()

        # For test set, sample z from training distribution and decode
        cvae.eval()
        test_predictions = []
        with torch.no_grad():
            # Load latent stats
            latent_stats = torch.load(os.path.join(SAVE_DIR, 'latent_stats.pt'), map_location=device)
            mu_train_mean = latent_stats['mu_train_mean'].to(device)
            mu_train_std = latent_stats['mu_train_std'].to(device)

            # Create a DataLoader for test_momenta with appropriate pin_memory
            test_momenta_loader = DataLoader(test_momenta, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin_memory)

            for momenta_batch in test_momenta_loader:
                momenta_batch = momenta_batch.to(device, non_blocking=True)
                batch_size = momenta_batch.size(0)
                # Sample z from training distribution
                z_sample = torch.randn(batch_size, LATENT_DIM, device=device)
                z_sample = z_sample * mu_train_std.unsqueeze(0) + mu_train_mean.unsqueeze(0)
                
                # Update autocast based on device type and scaler availability
                if device.type == 'cuda' and scaler is not None:
                    autocast_context = torch.amp.autocast(device_type='cuda', enabled=True)
                else:
                    autocast_context = torch.amp.autocast(enabled=False)
                
                with autocast_context:
                    predicted_position = cvae.decode(z_sample, momenta_batch)
                test_predictions.append(predicted_position.cpu())

        test_predictions = torch.cat(test_predictions, dim=0)

        # Inverse transform the predicted and actual positions
        if position_scaler is not None:
            test_predictions_inverse = position_scaler.inverse_transform(test_predictions.numpy())
            test_position_inverse = test_position_raw  # Original positions without scaling
        else:
            test_predictions_inverse = test_predictions.numpy()
            test_position_inverse = test_position.numpy()

        # Calculate MSE and MRE on test set using original values
        mse = np.mean((test_predictions_inverse - test_position_inverse) ** 2)
        relative_errors = np.abs(test_predictions_inverse - test_position_inverse) / (np.abs(test_position_inverse) + 1e-8)
        mre = np.mean(relative_errors)

        # Print and save results
        print(f"Test MSE: {mse}")
        print(f"Test MRE: {mre}")

        results = {
            'mse': float(mse),
            'mre': float(mre),
            'hyperparameters': {
                'LATENT_DIM': LATENT_DIM,
                'EPOCHS': EPOCHS,
                'BATCH_SIZE': BATCH_SIZE,
                'LEARNING_RATE': LEARNING_RATE,
                'PATIENCE': PATIENCE,
                'MIN_DELTA': MIN_DELTA,
                'hidden_layer_size': hidden_layer_size,
                'num_hidden_layers': num_hidden_layers,
                'activation': activation_name,
                'position_norm_method': position_norm_method,
                'momenta_norm_method': momenta_norm_method,
                'use_l1': use_l1,
                'l1_lambda': l1_lambda,
                'use_l2': use_l2,
                'l2_lambda': l2_lambda,
                'use_beta': use_beta,
                'beta': beta
            },
            'losses': {
                'training_primary_loss': train_primary_losses,
                'training_reg_loss': train_reg_losses,
                'validation_loss': val_primary_losses
            }
        }
        with open(os.path.join(SAVE_DIR, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)

# Run experiments
for config in experiments:
    print(f"Starting {config['name']}...")
    run_experiment(config)
    print(f"Finished {config['name']}.")
