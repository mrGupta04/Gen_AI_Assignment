import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Toy dataset: XOR problem (a simple binary classification problem)
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define a simple 2-layer neural network with ReLU activation
class SimpleNN(nn.Module):
    def __init__(self, activation_fn):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 2 input features, 2 neurons in hidden layer
        self.fc2 = nn.Linear(2, 1)  # 2 neurons in hidden layer, 1 output neuron
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a training function
def train_model(activation_fn, learning_rate=0.1, num_epochs=1000):
    model = SimpleNN(activation_fn)
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    loss_list = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return loss_list

# Train with ReLU activation
relu_loss = train_model(torch.relu, learning_rate=0.1)

# Train with Sigmoid activation
sigmoid_loss = train_model(torch.sigmoid, learning_rate=0.1)

# Plot the loss curves
plt.plot(relu_loss, label='ReLU Activation')
plt.plot(sigmoid_loss, label='Sigmoid Activation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves for ReLU and Sigmoid Activation')
plt.grid(True)
plt.show()
