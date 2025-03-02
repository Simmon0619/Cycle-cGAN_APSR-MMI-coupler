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

# Load dataset
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

# Define the discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + output_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.model(x)
        return x

# Initialize models
forward_model = ForwardModel(input_dim, output_dim).to(device)
inverse_model = InverseModel(output_dim, input_dim, latent_dim).to(device)
discriminator = Discriminator(input_dim, output_dim).to(device)

# Define loss functions and optimizers
criterion_mse = nn.MSELoss()
criterion_bce = nn.BCELoss()

optimizer_F = optim.Adam(forward_model.parameters(), lr=1e-3)
optimizer_G = optim.Adam(inverse_model.parameters(), lr=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=4e-4)

# Learning rate scheduler
scheduler_F = ReduceLROnPlateau(optimizer_F, mode='min', factor=0.5, patience=20, verbose=True)
scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=20, verbose=True)
scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=20, verbose=True)

print("Models and training setup ready.")

