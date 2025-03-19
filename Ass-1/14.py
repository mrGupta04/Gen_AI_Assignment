import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create a simple synthetic dataset (binary classification problem)
X = np.random.randn(1000, 2)  # 1000 samples, 2 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Label is 1 if sum of features > 0, else 0
X = Variable(torch.tensor(X, dtype=torch.float32))
y = Variable(torch.tensor(y, dtype=torch.float32))

# Define a simple Neural Network class with different activation functions
class SimpleNN(nn.Module):
    def __init__(self, activation_fn, init_type='xavier'):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 1)

        # Apply weight initialization
        if init_type == 'xavier':
            # Xavier Initialization
            nn.init.xavier_uniform_(self.layer1.weight)
            nn.init.xavier_uniform_(self.layer2.weight)
        elif init_type == 'he':
            # He Initialization
            nn.init.kaiming_uniform_(self.layer1.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.layer2.weight, nonlinearity='relu')

        # Choose the activation function
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.layer1(x))
        x = self.layer2(x)
        return x

# Train the network with different activation functions
def train_model(activation_fn, init_type):
    model = SimpleNN(activation_fn=activation_fn, init_type=init_type)
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    num_epochs = 100
    weight_distributions = []

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs.squeeze(), y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store the weight distribution of the first layer
        weight_distributions.append(model.layer1.weight.data.cpu().numpy().flatten())

    return weight_distributions

# Activation functions
activation_functions = {
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'relu': torch.relu
}

# Training with different activation functions and weight initializations
weight_dists_sigmoid_xavier = train_model(activation_functions['sigmoid'], init_type='xavier')
weight_dists_tanh_xavier = train_model(activation_functions['tanh'], init_type='xavier')
weight_dists_relu_he = train_model(activation_functions['relu'], init_type='he')

# Plotting the weight distributions over training for each case
plt.figure(figsize=(12, 6))

# Sigmoid + Xavier
plt.subplot(1, 3, 1)
plt.hist(weight_dists_sigmoid_xavier[-1], bins=30, color='blue', alpha=0.7)
plt.title('Sigmoid + Xavier Initialization')

# Tanh + Xavier
plt.subplot(1, 3, 2)
plt.hist(weight_dists_tanh_xavier[-1], bins=30, color='green', alpha=0.7)
plt.title('Tanh + Xavier Initialization')

# ReLU + He
plt.subplot(1, 3, 3)
plt.hist(weight_dists_relu_he[-1], bins=30, color='red', alpha=0.7)
plt.title('ReLU + He Initialization')

# Display the plots
plt.tight_layout()
plt.show()
