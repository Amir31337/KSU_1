import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------------
# 1. Configuration and Setup
# ---------------------------

# Define the path to your data file
DATA_PATH = '/content/drive/MyDrive/PhysicsProject-KSU/sim_million_orient.csv'

# Define the directory where all output files are saved
SAVE_DIR = '/content/drive/MyDrive/PhysicsProject-KSU/CVAE/OLD/v3'

# Define file names
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'best_model.pth')
LATENT_STATS_PATH = os.path.join(SAVE_DIR, 'latent_stats.pt')
MOMENTA_SCALER_PATH = os.path.join(SAVE_DIR, 'momenta_scaler.pkl')
POSITION_SCALER_PATH = os.path.join(SAVE_DIR, 'position_scaler.pkl')
DETAILED_RESULTS_PATH = os.path.join(SAVE_DIR, 'detailed_results.json')

# Output directories for visualizations
VISUALIZATION_DIR = os.path.join(SAVE_DIR, 'visualizations')
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# 2. Model Definition
# ---------------------------

# Define the CVAE model architecture (must match the training script)
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

# ---------------------------
# 3. Data Loading and Preprocessing
# ---------------------------

# Load test data
data = pd.read_csv(DATA_PATH)
position = data[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].values
momenta = data[['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']].values

# Split data into train, validation, and test sets (70%, 15%, 15%)
train_position, temp_position, train_momenta, temp_momenta = train_test_split(
    position, momenta, test_size=0.3, random_state=42, shuffle=True
)

val_position, test_position, val_momenta, test_momenta = train_test_split(
    temp_position, temp_momenta, test_size=0.5, random_state=42, shuffle=True
)

# Convert to PyTorch tensors
test_position_tensor = torch.FloatTensor(test_position)
test_momenta_tensor = torch.FloatTensor(test_momenta)

# Load scalers
position_scaler = joblib.load(POSITION_SCALER_PATH)
momenta_scaler = joblib.load(MOMENTA_SCALER_PATH)

# Normalize test data
test_position_norm = torch.FloatTensor(position_scaler.transform(test_position_tensor.numpy()))
test_momenta_norm = torch.FloatTensor(momenta_scaler.transform(test_momenta_tensor.numpy()))

# Create DataLoader for test data
BATCH_SIZE = 4096
test_dataset = TensorDataset(test_position_norm, test_momenta_norm)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,  # Adjust based on your system
    pin_memory=True,
    persistent_workers=False
)

# ---------------------------
# 4. Model Loading
# ---------------------------

# Initialize the model with the same hyperparameters as during training
# Update these hyperparameters if they were different during training
INPUT_DIM = 9
LATENT_DIM = 256
CONDITION_DIM = 9
HIDDEN_LAYERS = [64, 128]  # Adjust based on num_hidden_layers and hidden_layer_size
ACTIVATION_FUNCTION = nn.Sigmoid()

model = CVAE(
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM,
    condition_dim=CONDITION_DIM,
    hidden_layers=HIDDEN_LAYERS,
    activation_function=ACTIVATION_FUNCTION
).to(device)

# Load the trained model weights
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()
print("Trained model loaded successfully.")

# ---------------------------
# 5. Latent Space Visualization
# ---------------------------

# Function to extract latent variables (mu) from the test set
def extract_latent_mu(model, data_loader, device):
    mu_list = []
    with torch.no_grad():
        for batch_x, batch_cond in data_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_cond = batch_cond.to(device, non_blocking=True)
            mu, _ = model.encode(batch_x)
            mu_list.append(mu.cpu().numpy())
    mu_all = np.concatenate(mu_list, axis=0)
    return mu_all

# Extract mu from the test set
mu_test = extract_latent_mu(model, test_loader, device)
print(f"Extracted mu shape: {mu_test.shape}")

# For visualization, it's helpful to have labels. We'll use the momenta as labels.
# Depending on the dimensionality, we might need to select specific components or use clustering.

# Option 1: Use one of the momenta components as the label (e.g., pcx)
# Option 2: Cluster the momenta and use cluster labels

# Here, we'll proceed with Option 2: Clustering the momenta to obtain categorical labels

from sklearn.cluster import KMeans

# Number of clusters
NUM_CLUSTERS = 5

# Fit KMeans on momenta to obtain cluster labels
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
cluster_labels = kmeans.fit_predict(momenta_scaler.transform(test_momenta_tensor.numpy()))

# ---------------------------
# 5A. PCA Visualization
# ---------------------------

# Apply PCA to reduce latent dimensions to 2
pca = PCA(n_components=2)
mu_test_pca = pca.fit_transform(mu_test)

# Plot PCA scatter plot
plt.figure(figsize=(10, 8))
palette = sns.color_palette("hsv", NUM_CLUSTERS)
sns.scatterplot(
    x=mu_test_pca[:,0],
    y=mu_test_pca[:,1],
    hue=cluster_labels,
    palette=palette,
    legend='full',
    alpha=0.6
)
plt.title('Latent Space Visualization using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Momentum Cluster')
plt.grid(True)
plt.savefig(os.path.join(VISUALIZATION_DIR, 'latent_space_pca.png'))
plt.close()
print("PCA latent space visualization saved.")

# ---------------------------
# 5B. t-SNE Visualization
# ---------------------------

# Apply t-SNE to reduce latent dimensions to 2
tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
mu_test_tsne = tsne.fit_transform(mu_test)

# Plot t-SNE scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=mu_test_tsne[:,0],
    y=mu_test_tsne[:,1],
    hue=cluster_labels,
    palette=palette,
    legend='full',
    alpha=0.6
)
plt.title('Latent Space Visualization using t-SNE')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Momentum Cluster')
plt.grid(True)
plt.savefig(os.path.join(VISUALIZATION_DIR, 'latent_space_tsne.png'))
plt.close()
print("t-SNE latent space visualization saved.")

# ---------------------------
# 6. Reconstruction Quality Assessment
# ---------------------------

# Function to reconstruct positions with varying momenta
def reconstruct_with_varying_momenta(model, original_positions, original_momenta, scaler_pos, scaler_mom, device, num_samples=100):
    """
    Reconstruct positions by varying momenta conditions.
    Selects a subset of the data, perturbs their momenta, and generates new positions.
    
    Args:
        model: Trained CVAE model.
        original_positions: Original position data (numpy array).
        original_momenta: Original momenta data (numpy array).
        scaler_pos: Position scaler.
        scaler_mom: Momenta scaler.
        device: Torch device.
        num_samples: Number of samples to visualize.
    
    Returns:
        recon_positions_inv: Reconstructed positions after inverse scaling.
        original_positions_inv: Original positions after inverse scaling.
    """
    # Select random samples
    indices = np.random.choice(len(original_positions), size=num_samples, replace=False)
    selected_positions = original_positions[indices]
    selected_momenta = original_momenta[indices]
    
    # Apply some perturbation to the momenta
    # For simplicity, let's add small random noise
    noise = np.random.normal(0, 0.1, selected_momenta.shape)
    perturbed_momenta = selected_momenta + noise
    
    # Normalize perturbed momenta
    perturbed_momenta_norm = scaler_mom.transform(perturbed_momenta)
    perturbed_momenta_norm = torch.FloatTensor(perturbed_momenta_norm).to(device)
    
    # Encode the original positions to get mu and logvar
    with torch.no_grad():
        selected_positions_norm = scaler_pos.transform(selected_positions)
        selected_positions_norm = torch.FloatTensor(selected_positions_norm).to(device)
        mu, logvar = model.encode(selected_positions_norm)
        z = model.reparameterize(mu, logvar)
    
        # Decode with perturbed momenta
        recon_positions = model.decode(z, perturbed_momenta_norm).cpu().numpy()
        recon_positions_inv = scaler_pos.inverse_transform(recon_positions)
    
    # Inverse transform original positions for comparison
    original_positions_inv = scaler_pos.inverse_transform(selected_positions_norm.cpu().numpy())
    
    return recon_positions_inv, original_positions_inv, perturbed_momenta

# Perform reconstruction with varying momenta
recon_positions_inv, original_positions_inv, perturbed_momenta = reconstruct_with_varying_momenta(
    model,
    test_position,
    test_momenta,
    position_scaler,
    momenta_scaler,
    device,
    num_samples=100  # Adjust as needed
)

# ---------------------------
# 6A. Visual Comparison of Reconstructions
# ---------------------------

# Plot original vs reconstructed positions for a subset of components
components = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
num_components = len(components)

# Create a grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
fig.suptitle('Original vs Reconstructed Positions with Varying Momenta', fontsize=20)

for i, ax in enumerate(axes.flatten()):
    if i < num_components:
        # Scatter plot for original positions
        ax.scatter(
            original_positions_inv[:, i],
            recon_positions_inv[:, i],
            alpha=0.6,
            label='Reconstructed',
            color='r'
        )
        # Plot a diagonal line for reference
        min_val = min(original_positions_inv[:, i].min(), recon_positions_inv[:, i].min())
        max_val = max(original_positions_inv[:, i].max(), recon_positions_inv[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax.set_xlabel(f'Original {components[i]}')
        ax.set_ylabel(f'Reconstructed {components[i]}')
        ax.set_title(f'Original vs Reconstructed {components[i]}')
        ax.legend()
        ax.grid(True)
    else:
        ax.axis('off')  # Hide any unused subplots

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(VISUALIZATION_DIR, 'reconstruction_quality_scatter.png'))
plt.close()
print("Reconstruction quality scatter plots saved.")

# ---------------------------
# 6B. Trajectory Visualization
# ---------------------------

# For a more dynamic comparison, visualize how reconstructed positions change with varying momenta
# For example, plot the trajectory of a single sample as its momenta vary

def plot_trajectory(original, reconstructed, perturbed_momenta, component_pairs=[('cx','cz'), ('ox','oz')], save_path='trajectory.png'):
    """
    Plot trajectories of original and reconstructed positions for specified component pairs.
    
    Args:
        original: Original positions (num_samples, num_components).
        reconstructed: Reconstructed positions (num_samples, num_components).
        perturbed_momenta: Perturbed momenta used for reconstruction (num_samples, num_components).
        component_pairs: List of tuples specifying which components to plot against each other.
        save_path: Path to save the plot.
    """
    num_pairs = len(component_pairs)
    fig, axes = plt.subplots(1, num_pairs, figsize=(10*num_pairs, 8))
    
    if num_pairs == 1:
        axes = [axes]  # Make it iterable
    
    for ax, (comp_x, comp_y) in zip(axes, component_pairs):
        idx_x = components.index(comp_x)
        idx_y = components.index(comp_y)
        
        # Original positions
        ax.scatter(
            original[:, idx_x],
            original[:, idx_y],
            alpha=0.6,
            label='Original',
            color='b'
        )
        
        # Reconstructed positions
        ax.scatter(
            reconstructed[:, idx_x],
            reconstructed[:, idx_y],
            alpha=0.6,
            label='Reconstructed',
            color='r'
        )
        
        ax.set_xlabel(f'Original {comp_x}')
        ax.set_ylabel(f'Original {comp_y}')
        ax.set_title(f'{comp_x} vs {comp_y}')
        ax.legend()
        ax.grid(True)
    
    plt.suptitle('Trajectory Comparison: Original vs Reconstructed Positions', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Trajectory comparison plot saved to {save_path}.")

# Example usage for two component pairs
plot_trajectory(
    original_positions_inv,
    recon_positions_inv,
    perturbed_momenta,
    component_pairs=[('cx', 'cz'), ('ox', 'oz')],
    save_path=os.path.join(VISUALIZATION_DIR, 'trajectory_comparison.png')
)

# ---------------------------
# 7. Latent Space Statistics (Optional)
# ---------------------------

# If you wish to explore the latent space statistics further, you can load and visualize mu_train_mean and mu_train_std
# which were saved during training.

if os.path.exists(LATENT_STATS_PATH):
    latent_stats = torch.load(LATENT_STATS_PATH)
    mu_train_mean = latent_stats['mu_train_mean'].numpy()
    mu_train_std = latent_stats['mu_train_std'].numpy()
    print("Latent space statistics loaded successfully.")
else:
    print(f"Latent stats file not found at {LATENT_STATS_PATH}")

# ---------------------------
# 8. Additional Visualizations (Optional)
# ---------------------------

# For a more comprehensive analysis, consider plotting histograms of the latent variables or correlations with momenta.

# Example: Plotting histograms of the first 5 latent dimensions
num_histograms = 5
plt.figure(figsize=(20, 15))
for i in range(num_histograms):
    plt.subplot(3, 2, i+1)
    sns.histplot(mu_test[:, i], bins=50, kde=True, color='g')
    plt.title(f'Latent Dimension {i+1} Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_DIR, 'latent_dim_histograms.png'))
plt.close()
print("Latent dimension histograms saved.")
