import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import List, Tuple  # Add this line

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 9
ENCODER_HIDDEN_DIMS = [256, 128, 64]
DECODER_HIDDEN_DIMS = [64, 128, 256]

class CVAE(nn.Module):
    def __init__(self, input_size, latent_size, output_size, condition_size):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.condition_size = condition_size

        # encode
        self.fc1 = nn.Linear(input_size + condition_size, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + condition_size, 400)
        self.fc4 = nn.Linear(400, output_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h3 = self.elu(self.fc3(inputs))
        return self.fc4(h3)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


# Load the test data
#df_test = pd.read_csv('/home/g/ghanaatian/MYFILES/FALL24/Physics/19Aug/CVAE/generated_cos3d_check.csv')
df_test = pd.read_csv('/home/g/ghanaatian/MYFILES/FALL24/Physics/19Aug/CINN/test.csv')
df_test[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']] = df_test[['cx', 'cy', 'cz', 'ox', 'oy', 'oz', 'sx', 'sy', 'sz']].fillna(1)

# Fit scalers on the entire dataset
scaler_momenta = MinMaxScaler().fit(df_test.iloc[:, 10:19])

# Load the saved model
input_size = 9  # 9 momentum values
condition_dim = 9  # 9 momentum values
output_size = 9  # 9 position values
condition_size = 9  # 9 position values
model = CVAE(input_size, LATENT_DIM, output_size, condition_size).to(device)
model.load_state_dict(torch.load('/home/g/ghanaatian/MYFILES/FALL24/Physics/19Aug/CVAE/V3.pth', map_location=device, weights_only=True))
model.eval()


# Define a function to align numbers with fixed width
def format_list(lst):
    return ', '.join(f"{x:>7.3f}" for x in lst)


# Function to process a single row
def process_row(row_number):
    row = df_test.iloc[row_number]
    momenta_real = pd.DataFrame(row[10:]).T
    positions_real = pd.DataFrame(row[1:10]).T
    momenta_normalized = torch.tensor(scaler_momenta.transform(momenta_real), dtype=torch.float32)
    positions_real = torch.tensor(positions_real.values, dtype=torch.float32)

    with torch.no_grad():
        # Pass momenta_normalized twice, as both input and condition
        predicted_positions, _, _ = model(momenta_normalized.to(device), positions_real.to(device))

    return positions_real.numpy().flatten(), predicted_positions.cpu().numpy().flatten()


# Generate 10 unique random row numbers
random_rows = np.random.choice(len(df_test), 1, replace=False)

# Print real and predicted values for 10 random rows
for row in random_rows:
    real, predicted = process_row(row)
    print(f"\nRow number {row}")
    positions_str = format_list(real)
    predicted_positions_str = format_list(predicted)
    print(f"Real positions     : [{positions_str}]")
    print(f"Predicted positions: [{predicted_positions_str}]")

# Process all rows for loss calculation
all_real_values = []
all_predicted_values = []

for i in range(len(df_test)):
    real, predicted = process_row(i)
    all_real_values.extend(real)
    all_predicted_values.extend(predicted)

# Calculate the Mean Squared Error, R-squared, and Mean Absolute Error for all rows
mse_all_rows = mean_squared_error(all_real_values, all_predicted_values)
r2_all_rows = r2_score(all_real_values, all_predicted_values)
mae_all_rows = mean_absolute_error(all_real_values, all_predicted_values)

print(f"\nLosses for all 10K rows:")
print(f"Mean Squared Error: {mse_all_rows:.4f}")
print(f"R-squared         : {r2_all_rows:.4f}")
print(f"Mean Absolute Error: {mae_all_rows:.4f}")