import numpy as np
import matplotlib.pyplot as plt

# Define the 1000-dimensional function and its gradient
def f(x):
    return np.sum(x ** 2)

def grad_f(x):
    return 2 * x  # Gradient of f(x) = sum(x_i^2)

# Implementing Gradient Descent
def gradient_descent(learning_rate, epochs, dimensions=1000):
    x = np.random.randn(dimensions)  # Initialize the parameters randomly
    loss_history = []

    for _ in range(epochs):
        grad = grad_f(x)
        x -= learning_rate * grad  # Update rule for gradient descent
        loss_history.append(f(x))  # Track the loss

    return loss_history

# Implementing Adam Optimizer
def adam_optimizer(learning_rate, beta1, beta2, epsilon, epochs, dimensions=1000):
    x = np.random.randn(dimensions)  # Initialize the parameters randomly
    m = np.zeros(dimensions)  # First moment estimate (for momentum)
    v = np.zeros(dimensions)  # Second moment estimate (for adaptive learning rate)
    loss_history = []

    for t in range(1, epochs + 1):
        grad = grad_f(x)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        m_hat = m / (1 - beta1 ** t)  # Bias correction
        v_hat = v / (1 - beta2 ** t)  # Bias correction
        
        x -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)  # Update rule for Adam
        
        loss_history.append(f(x))  # Track the loss

    return loss_history

# Hyperparameters
epochs = 1000
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Perform optimization with Gradient Descent and Adam
gd_loss = gradient_descent(learning_rate, epochs)
adam_loss = adam_optimizer(learning_rate, beta1, beta2, epsilon, epochs)

# Plotting the loss curves
plt.figure(figsize=(10, 6))
plt.plot(gd_loss, label='Gradient Descent', color='b')
plt.plot(adam_loss, label='Adam', color='g')

plt.title('Loss Comparison on 1000-dimensional Quadratic Function')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
