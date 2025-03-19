import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)

# Define a simple logistic regression model (1-layer neural network)
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(28*28, 10)  # 28x28 input, 10 output classes

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the 28x28 images to a 1D vector
        return self.fc(x)

# Function to train the model with a given learning rate
def train_model(learning_rate, num_epochs=10):
    model = LogisticRegression()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    loss_curve = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        loss_curve.append(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return loss_curve

# Set different learning rates to experiment with
learning_rates = [0.0001, 0.01, 0.1, 1.0]

# Train the model with each learning rate and store the loss curves
loss_curves = {}
for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    loss_curves[lr] = train_model(lr)

# Plot the loss curves for comparison
plt.figure(figsize=(10, 6))

for lr, loss_curve in loss_curves.items():
    plt.plot(range(1, len(loss_curve) + 1), loss_curve, label=f'LR={lr}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves for Different Learning Rates')
plt.legend()
plt.grid(True)
plt.show()
