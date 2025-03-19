import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Small e value to prevent numerical issues (Only for Training)
e = 1e-2

# Define the XOR dataset (Epsilon applied only in Training)
X_train = torch.tensor([[0+e, 0-e], [0+e, 1-e], [1-e, 0+e], [1+e, 1+e]], dtype=torch.float32)
y_train = torch.tensor([[0 + e], [1 - e], [1 - e], [0 + e]], dtype=torch.float32)  

# Define the Neural Network with one hidden layer
class XOR_Network(nn.Module):
    def __init__(self):
        super(XOR_Network, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # Small hidden layer (4 neurons)
        self.fc2 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply ReLU activation in hidden layer
        x = self.sigmoid(self.fc2(x))  # Apply Sigmoid in output layer
        return x

# Initialize the neural network
model = XOR_Network()

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy for classification
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Lower learning rate

# Train the model
num_epochs = 10000
loss_list = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)  
    loss = criterion(outputs, y_train)  # Loss computed using modified training labels
    loss.backward()
    optimizer.step()
    
    # Store loss every 100 epochs
    if epoch % 100 == 0:
        loss_list.append(loss.item())

    # Print progress every 2000 epochs
    if (epoch + 1) % 2000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}', flush=True)

# Training completed, test the model
print("\nTraining completed. Running test cases...\n", flush=True)

# Test the model after training (No e applied in testing)
with torch.no_grad():
    test_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    test_outputs = model(test_inputs)  # No clamping applied

    for i in range(len(test_inputs)):
        input_values = [int(x) for x in test_inputs[i].tolist()]
        predicted_value = test_outputs[i].item()
        predicted = 1 if predicted_value > 0.5 else 0  # Thresholding
        print(f"Input: {input_values} -> Predicted: {predicted} (Raw Output: {predicted_value:.6f})", flush=True)

# Plot the loss curve
plt.plot(loss_list)
plt.yscale("log")  # Log scale for better visualization
plt.xlabel('Epoch (every 100)')
plt.ylabel('Loss')
plt.title('Loss Curve for XOR Neural Network')
plt.grid(True)
plt.show()
