import numpy as np
import matplotlib.pyplot as plt

# Sigmoid Activation Function and its Derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Tanh Activation Function and its Derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Binary Cross-Entropy Loss Function
def binary_cross_entropy(y_true, y_pred):
    m = len(y_true)
    epsilon = 1e-15  # To prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid log(0)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m

# Gradient Descent Update
def gradient_descent(X, y, activation_fn, activation_fn_derivative, learning_rate=0.1, epochs=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    loss_history = []

    for epoch in range(epochs):
        # Forward pass
        z = np.dot(X, w) + b
        y_pred = activation_fn(z)

        # Compute loss
        loss = binary_cross_entropy(y, y_pred)
        loss_history.append(loss)

        # Compute gradients
        dz = y_pred - y
        dw = (1 / m) * np.dot(X.T, dz)
        db = (1 / m) * np.sum(dz)

        # Update weights and bias
        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    return w, b, loss_history

# Generate Some Synthetic Data for Logistic Regression
np.random.seed(42)

# Create a simple dataset
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Class 1 if sum of features > 0, else class 0

# Train Logistic Regression with Sigmoid Activation
print("Training with Sigmoid activation:")
w_sigmoid, b_sigmoid, loss_sigmoid = gradient_descent(X, y, sigmoid, sigmoid_derivative)

# Train Logistic Regression with Tanh Activation
print("\nTraining with Tanh activation:")
w_tanh, b_tanh, loss_tanh = gradient_descent(X, y, tanh, tanh_derivative)

# Plotting the loss convergence for both cases
plt.figure(figsize=(12, 6))

# Plot the loss for Sigmoid
plt.plot(loss_sigmoid, label="Sigmoid", color="blue")

# Plot the loss for Tanh
plt.plot(loss_tanh, label="Tanh", color="green")

# Labels and title
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Convergence for Sigmoid vs Tanh")
plt.legend()

# Show the plot
plt.show()
