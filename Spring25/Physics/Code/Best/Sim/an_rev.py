import pandas as pd
import numpy as np

# File path to the CSV file
file_path = '/home/g/ghanaatian/MYFILES/FALL24/Physics/CVAE/Energy/Best/Sim2/test_predictions.csv'

# Define the column lists
real_cols = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
predicted_cols = ['pred_cx', 'pred_cy', 'pred_cz', 'pred_ox', 'pred_oy', 'pred_oz', 'pred_sx', 'pred_sy', 'pred_sz']
momenta_cols = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']

# Load the data from the CSV file
data = pd.read_csv(file_path)

# Columns to set to zero
columns_to_zero = ['cz', 'pred_cz', 'sz', 'pred_sz', 'sy', 'pred_sy']

# Set the specified columns to zero
data[columns_to_zero] = 0

# Define columns to be used for MRE calculation (6 columns)
mre_real_cols = ['cx', 'cy', 'ox', 'oy', 'oz', 'sx']
mre_pred_cols = ['pred_cx', 'pred_cy', 'pred_ox', 'pred_oy', 'pred_oz', 'pred_sx']

# Ensure that the real and predicted columns match in length
assert len(mre_real_cols) == len(mre_pred_cols), "Mismatch between real and predicted columns for MRE calculation."

# Calculate the Mean Relative Error (MRE)
epsilon = 1e-100  # Small constant to avoid division by zero

# Compute absolute differences between real and predicted values
abs_diff = np.abs(data[mre_real_cols].values - data[mre_pred_cols].values)

# Compute absolute real values and add epsilon to avoid division by zero
abs_real = np.abs(data[mre_real_cols].values) + epsilon

# Calculate MRE values in percentage for the 6 columns
mre_values = (abs_diff / abs_real) * 100  # Shape: (n_samples, 6)

# Sum the relative errors for each sample
mre_sum = mre_values.sum(axis=1)  # Shape: (n_samples,)

# Divide the summed relative errors by 9
mre_normalized = mre_sum / 9  # Shape: (n_samples,)

# Compute the final Mean Relative Error by averaging over all samples
mre = mre_normalized.mean()

# Extract momenta for energy calculations
pc = data[['pcx', 'pcy', 'pcz']]
po = data[['pox', 'poy', 'poz']]
ps = data[['psx', 'psy', 'psz']]

# Define masses
masses = {'mc': 21894.71361, 'mo': 29164.39289, 'ms': 58441.80487}

# Calculate kinetic energy for each component
ke_c = (pc**2).sum(axis=1) / (2 * masses['mc'])
ke_o = (po**2).sum(axis=1) / (2 * masses['mo'])
ke_s = (ps**2).sum(axis=1) / (2 * masses['ms'])

# Total kinetic energy
ke_total = ke_c + ke_o + ke_s

# Calculate distances between particles with epsilon to avoid division by zero
r_co = np.sqrt((data['cx'] - data['ox'])**2 + (data['cy'] - data['oy'])**2 + (data['cz'] - data['oz'])**2) + epsilon
r_cs = np.sqrt((data['cx'] - data['sx'])**2 + (data['cy'] - data['sy'])**2 + (data['cz'] - data['sz'])**2) + epsilon
r_os = np.sqrt((data['ox'] - data['sx'])**2 + (data['oy'] - data['sy'])**2 + (data['oz'] - data['sz'])**2) + epsilon

# Calculate potential energy
pe_total = 4 / r_co + 4 / r_cs + 4 / r_os

# Calculate energy difference loss
energy_diff_loss = np.abs(ke_total - pe_total) / np.abs(ke_total)

# Output the results
print(f"Mean Relative Error (MRE): {mre:.2f}%")
print(f"Average Energy Difference Loss: {energy_diff_loss.mean():.20f}")

'''
Mean Relative Error (MRE): 82.66%
Average Energy Difference Loss: 0.00002028507780736689
'''