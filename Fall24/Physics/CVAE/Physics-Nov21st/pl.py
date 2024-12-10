import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Read the CSV file
data_path = "/Users/khorzu/Downloads/test_with_predictions.csv"
save_path = "/Users/khorzu/Downloads/grid_plots.png"

def create_grid_plots(data_file):
    # Read the data
    df = pd.read_csv(data_file)
    
    # Define the columns we want to plot
    columns = ['pred_cx', 'pred_cy', 'pred_cz',
              'pred_ox', 'pred_oy', 'pred_oz',
              'pred_sx', 'pred_sy', 'pred_sz']
    
    # Create readable labels
    labels = ['Predicted Carbon X', 'Predicted Carbon Y', 'Predicted Carbon Z',
             'Predicted Oxygen X', 'Predicted Oxygen Y', 'Predicted Oxygen Z',
             'Predicted Sulfur X', 'Predicted Sulfur Y', 'Predicted Sulfur Z']
    
    # Create a figure with 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Distribution of Predicted Coordinates',
                fontsize=16, y=0.95)
    
    # Flatten the 2D array of axes for easier iteration
    axes = axes.ravel()
    
    for i, (column, label) in enumerate(zip(columns, labels)):
        var = df[column].values
        ax = axes[i]
        
        # Anderson-Darling test
        result = stats.anderson(var, dist='norm')
        
        # Histogram with fitted normal
        mean, std = stats.norm.fit(var)
        x = np.linspace(min(var), max(var), 100)
        pdf = stats.norm.pdf(x, mean, std)
        
        # Plot on the corresponding subplot
        ax.hist(var, bins=50, density=True, alpha=0.7, label='Data')
        ax.plot(x, pdf, 'r-', label=f'μ={mean:.2f}\nσ={std:.2f}')
        ax.legend(fontsize='small')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # Add Anderson-Darling test statistic
        ax.text(0.05, 0.95, f'A²={result.statistic:.3f}',
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=8)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Add space for the main title
    plt.subplots_adjust(top=0.92)
    
    return fig

# Create the plots
fig = create_grid_plots(data_path)

# Save the figure
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Plot saved as: {save_path}")
