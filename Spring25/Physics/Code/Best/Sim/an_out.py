import pandas as pd
import numpy as np

# Load the provided CSV file
file_path = '/home/g/ghanaatian/MYFILES/FALL24/Physics/CVAE/Energy/Best/Sim2/test_predictions.csv'
data = pd.read_csv(file_path)

# Define columns
real_cols = ['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']
predicted_cols = ['pred_cx', 'pred_cy', 'pred_cz', 'pred_ox', 'pred_oy', 'pred_oz', 'pred_sx', 'pred_sy', 'pred_sz']
momenta_cols = ['pcx', 'pcy', 'pcz', 'pox', 'poy', 'poz', 'psx', 'psy', 'psz']
masses = {'mc': 21894.71361, 'mo': 29164.39289, 'ms': 58441.80487}

# Calculate the Mean Relative Error (MRE) for real vs predicted positions
epsilon = 1e-10  # Small constant to avoid division by zero

# Compute absolute differences
abs_diff = abs(data[real_cols].values - data[predicted_cols].values)

# Compute absolute real values and handle zeros by adding epsilon
abs_real = abs(data[real_cols].values) + epsilon

# Calculate MRE values
mre_values = (abs_diff / abs_real) * 100

# Compute mean MRE
mre = mre_values.mean()

# Extract momenta for energy calculations
pc = data[['pcx', 'pcy', 'pcz']]
po = data[['pox', 'poy', 'poz']]
ps = data[['psx', 'psy', 'psz']]

# Calculate kinetic energy
ke_c = (pc**2).sum(axis=1) / (2 * masses['mc'])
ke_o = (po**2).sum(axis=1) / (2 * masses['mo'])
ke_s = (ps**2).sum(axis=1) / (2 * masses['ms'])
ke_total = ke_c + ke_o + ke_s

# Calculate distances
r_co = np.sqrt((data['cx'] - data['ox'])**2 + (data['cy'] - data['oy'])**2 + (data['cz'] - data['oz'])**2) + epsilon
r_cs = np.sqrt((data['cx'] - data['sx'])**2 + (data['cy'] - data['sy'])**2 + (data['cz'] - data['sz'])**2) + epsilon
r_os = np.sqrt((data['ox'] - data['sx'])**2 + (data['oy'] - data['sy'])**2 + (data['oz'] - data['sz'])**2) + epsilon

# Calculate potential energy
pe_total = 4 / r_co + 4 / r_cs + 4 / r_os

# Calculate energy difference loss
energy_diff_loss = abs(ke_total - pe_total) / abs(ke_total)

# Output the results
print(f"Mean Relative Error (MRE): {mre:.2f}%")
print(f"Average Energy Difference Loss: {energy_diff_loss.mean():.20f}")


'''
Mean Relative Error (MRE): 82.67%
'''