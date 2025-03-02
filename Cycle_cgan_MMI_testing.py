import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import pandas as pd
import time
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Check for available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load dataset
def load_data(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        inputs = hf['input'][:]
        outputs = hf['output'][:]
    outputs = np.round(outputs, 4)
    inputs = inputs * 1000000  # Scaling input values
    inputs = np.round(inputs, 4)
    outputs = np.round(outputs, 4)
    outputs_sum = outputs.sum(axis=1)
    valid_indices = np.where(outputs_sum > 0.60)[0]
    inputs = inputs[valid_indices]
    outputs = outputs[valid_indices]
    inputs = inputs.astype(np.float32)
    outputs = outputs.astype(np.float32)
    return inputs, outputs

# Load trained model weights
forward_model_path = 'path_to_trained_forward_model.pth'
inverse_model_path = 'path_to_trained_inverse_model.pth'
inputs_v1, outputs_v1 = load_data('path_to_h5_file_v1.h5')
inputs_v2, outputs_v2 = load_data('path_to_h5_file_v2.h5')
inputs_v3, outputs_v3 = load_data('path_to_h5_file_v3.h5')
# Combine datasets
inputs_big = np.concatenate((inputs_v1, inputs_v2, inputs_v3), axis=0)
outputs_big = np.concatenate((outputs_v1, outputs_v2, outputs_v3), axis=0)
print(inputs_big.shape, outputs_big.shape)

# Convert data to tensors
inputs_big = torch.tensor(inputs_big, dtype=torch.float32)
outputs_big = torch.tensor(outputs_big, dtype=torch.float32)

# Normalize data
inputs_mean = inputs_big.mean(0, keepdim=True)
inputs_std = inputs_big.std(0, keepdim=True)
inputs_big_norm = (inputs_big - inputs_mean) / inputs_std

outputs_mean = outputs_big.mean(0, keepdim=True)
outputs_std = outputs_big.std(0, keepdim=True)
outputs_big_norm = (outputs_big - outputs_mean) / outputs_std

# Move mean and std to GPU if available
inputs_mean = inputs_mean.to(device)
inputs_std = inputs_std.to(device)
outputs_mean = outputs_mean.to(device)
outputs_std = outputs_std.to(device)

# Split dataset into training and testing sets
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
    inputs_big_norm, outputs_big_norm, test_size=0.2, random_state=42)

# Create DataLoader objects
train_dataset = TensorDataset(inputs_train, outputs_train)
test_dataset = TensorDataset(inputs_test, outputs_test)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Get input and output dimensions
input_dim = inputs_big.shape[1]   # Input parameter dimension
output_dim = outputs_big.shape[1]  # Output parameter dimension
latent_dim = 4  # Latent noise dimension
# Define the forward model
class ForwardModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ForwardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

# Define the inverse model (Generator)
class InverseModel(nn.Module):
    def __init__(self, output_dim, input_dim, latent_dim):
        super(InverseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(output_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    def forward(self, y, noise):
        x = torch.cat([y, noise], dim=1)
        x = self.model(x)
        return x

# Initialize models
input_dim = 6  # Adjust based on actual input dimensions
output_dim = 2  # Adjust based on actual output dimensions
latent_dim = 4

forward_model = ForwardModel(input_dim, output_dim).to(device)
inverse_model = InverseModel(output_dim, input_dim, latent_dim).to(device)

forward_model.load_state_dict(torch.load(forward_model_path))
inverse_model.load_state_dict(torch.load(inverse_model_path))

# Set models to evaluation mode
forward_model.eval()
inverse_model.eval()

# Define desired power splitting ratios (PSRs) and excess loss range
psrs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
initial_excess_loss = 0.01  # Starting excess loss in dB
max_excess_loss = 1.0  # Maximum allowed excess loss in dB
loss_increment = 0.005  # Increment for excess loss iteration
acceptable_mse_threshold = 0.005  # Target MSE threshold

total_power = 1.0  # Normalized total power

# Function to compute power values based on PSR and excess loss
def compute_power_values(psr, excess_loss):
    power_factor = 10 ** (-excess_loss / 10)
    P_outB = total_power * power_factor * psr
    P_outC = total_power * power_factor * (1 - psr)
    return P_outB, P_outC

# Store results
results = []

# Iterate over PSRs
for psr in psrs:
    excess_loss = initial_excess_loss

    while excess_loss <= max_excess_loss:
        # Compute target power values
        P_outB, P_outC = compute_power_values(psr, excess_loss)
        target_power = torch.tensor([[P_outB, P_outC]], dtype=torch.float32).to(device)

        # Normalize target power
        target_power_norm = (target_power - outputs_mean) / outputs_std

        # Generate multiple structural parameters for each excess loss
        generated_structures = []
        predicted_powers = []
        mses = []

        for _ in range(20):  # Generate 20 candidates per iteration
            with torch.no_grad():
                noise = torch.randn(1, latent_dim, device=device)
                generated_params_norm = inverse_model(target_power_norm, noise)
                
                # Denormalize generated parameters
                generated_params = generated_params_norm * inputs_std + inputs_mean
                generated_params = generated_params.cpu().numpy().flatten()

                # Evaluate with forward model
                predicted_power_norm = forward_model(
                    torch.tensor(generated_params_norm, dtype=torch.float32).unsqueeze(0).to(device))
                predicted_power = predicted_power_norm * outputs_std + outputs_mean
                predicted_power = predicted_power.cpu().numpy().flatten()

                # Compute MSE
                mse = np.mean(abs(predicted_power - target_power.cpu().numpy().flatten()))
                
                # Store results
                generated_structures.append(generated_params)
                predicted_powers.append(predicted_power)
                mses.append(mse)

        # Select the best structure based on the smallest MSE
        best_idx = np.argmin(mses)
        best_mse = mses[best_idx]
        best_structure = generated_structures[best_idx]
        best_predicted_power = predicted_powers[best_idx]

        # Check if the best MSE meets the threshold
        if best_mse <= acceptable_mse_threshold:
            # Store results
            results.append({
                'PSR': psr,
                'Excess Loss (dB)': excess_loss,
                'Best Generated Params': best_structure.tolist(),
                'Target Power': [P_outB, P_outC],
                'Best Predicted Power': best_predicted_power.tolist(),
                'Best MSE': best_mse
            })
            break  # Stop iteration for this PSR
        
        # Increment excess loss
        excess_loss += loss_increment

# Convert results to DataFrame and save to Excel
results_df = pd.DataFrame(results)
output_path = 'design_mode_results.xlsx'
results_df.to_excel(output_path, index=False)

print(f"Design mode results saved to {output_path}.")
