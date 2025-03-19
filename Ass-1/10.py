import numpy as np
import matplotlib.pyplot as plt

# 1. Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# 2. Plotting the Activation Functions and Their Derivatives
def plot_activation_functions():
    # Create a range of values for x
    x = np.linspace(-10, 10, 400)

    # Set up a figure for plotting
    plt.figure(figsize=(12, 8))

    # Sigmoid
    plt.subplot(2, 2, 1)
    plt.plot(x, sigmoid(x), label="Sigmoid", color="blue")
    plt.title("Sigmoid Activation Function")
    plt.grid(True)

    # Tanh
    plt.subplot(2, 2, 2)
    plt.plot(x, tanh(x), label="Tanh", color="green")
    plt.title("Tanh Activation Function")
    plt.grid(True)

    # ReLU
    plt.subplot(2, 2, 3)
    plt.plot(x, relu(x), label="ReLU", color="red")
    plt.title("ReLU Activation Function")
    plt.grid(True)

    # Leaky ReLU
    plt.subplot(2, 2, 4)
    plt.plot(x, leaky_relu(x), label="Leaky ReLU", color="purple")
    plt.title("Leaky ReLU Activation Function")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_activation_derivatives():
    # Create a range of values for x
    x = np.linspace(-10, 10, 400)

    # Set up a figure for plotting the derivatives
    plt.figure(figsize=(12, 8))

    # Sigmoid Derivative
    plt.subplot(2, 2, 1)
    plt.plot(x, sigmoid_derivative(x), label="Sigmoid Derivative", color="blue")
    plt.title("Sigmoid Derivative")
    plt.grid(True)

    # Tanh Derivative
    plt.subplot(2, 2, 2)
    plt.plot(x, tanh_derivative(x), label="Tanh Derivative", color="green")
    plt.title("Tanh Derivative")
    plt.grid(True)

    # ReLU Derivative
    plt.subplot(2, 2, 3)
    plt.plot(x, relu_derivative(x), label="ReLU Derivative", color="red")
    plt.title("ReLU Derivative")
    plt.grid(True)

    # Leaky ReLU Derivative
    plt.subplot(2, 2, 4)
    plt.plot(x, leaky_relu_derivative(x), label="Leaky ReLU Derivative", color="purple")
    plt.title("Leaky ReLU Derivative")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 3. Main Code to Generate Plots
def main():
    # Plot Activation Functions
    plot_activation_functions()
    
    # Plot Derivatives of Activation Functions
    plot_activation_derivatives()

# Run the main function
if __name__ == "__main__":
    main()
