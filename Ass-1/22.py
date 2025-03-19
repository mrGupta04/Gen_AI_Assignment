import numpy as np
import matplotlib.pyplot as plt

# Function: f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# Gradient of f(x, y) = [2x, 2y]
def gradient(x, y):
    return np.array([2 * x, 2 * y])

# Gradient Descent Variants:

# Standard Gradient Descent
def gradient_descent(learning_rate, epochs):
    x, y = np.random.randn(), np.random.randn()  # Random initialization
    loss_history = []
    
    for _ in range(epochs):
        grad = gradient(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
        loss_history.append(f(x, y))  # Track the loss
    
    return loss_history

# Stochastic Gradient Descent (SGD) - For simplicity, we will use the same loss function but update in a "stochastic" way.
def sgd(learning_rate, epochs):
    x, y = np.random.randn(), np.random.randn()  # Random initialization
    loss_history = []
    
    for _ in range(epochs):
        grad = gradient(x, y)
        # SGD usually uses one sample update, so here it's treated like full-batch update, for simplicity
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
        loss_history.append(f(x, y))  # Track the loss
    
    return loss_history

# Momentum-based Gradient Descent
def momentum_gradient_descent(learning_rate, momentum, epochs):
    x, y = np.random.randn(), np.random.randn()  # Random initialization
    v_x, v_y = 0, 0  # Initialize momentum terms
    loss_history = []
    
    for _ in range(epochs):
        grad = gradient(x, y)
        v_x = momentum * v_x + learning_rate * grad[0]
        v_y = momentum * v_y + learning_rate * grad[1]
        
        x -= v_x
        y -= v_y
        loss_history.append(f(x, y))  # Track the loss
    
    return loss_history

# Adam Optimizer
def adam_optimizer(learning_rate, beta1, beta2, epsilon, epochs):
    x, y = np.random.randn(), np.random.randn()  # Random initialization
    m_x, m_y = 0, 0  # Initialize first moment estimates
    v_x, v_y = 0, 0  # Initialize second moment estimates
    loss_history = []
    
    for t in range(1, epochs + 1):
        grad = gradient(x, y)
        
        m_x = beta1 * m_x + (1 - beta1) * grad[0]
        m_y = beta1 * m_y + (1 - beta1) * grad[1]
        v_x = beta2 * v_x + (1 - beta2) * grad[0]**2
        v_y = beta2 * v_y + (1 - beta2) * grad[1]**2
        
        m_x_hat = m_x / (1 - beta1**t)  # Bias correction
        m_y_hat = m_y / (1 - beta1**t)
        v_x_hat = v_x / (1 - beta2**t)  # Bias correction
        v_y_hat = v_y / (1 - beta2**t)
        
        x -= learning_rate * m_x_hat / (np.sqrt(v_x_hat) + epsilon)
        y -= learning_rate * m_y_hat / (np.sqrt(v_y_hat) + epsilon)
        
        loss_history.append(f(x, y))  # Track the loss
    
    return loss_history

# Hyperparameters
epochs = 100
learning_rate = 0.1
momentum = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Train the models
loss_gd = gradient_descent(learning_rate, epochs)
loss_sgd = sgd(learning_rate, epochs)
loss_momentum = momentum_gradient_descent(learning_rate, momentum, epochs)
loss_adam = adam_optimizer(learning_rate, beta1, beta2, epsilon, epochs)

# Plotting the loss curves
plt.figure(figsize=(10, 6))
plt.plot(loss_gd, label='Gradient Descent', color='b')
plt.plot(loss_sgd, label='Stochastic Gradient Descent', color='g')
plt.plot(loss_momentum, label='Momentum-based Gradient Descent', color='r')
plt.plot(loss_adam, label='Adam Optimizer', color='m')

plt.title('Loss Comparison of Different Optimizers on f(x, y) = x^2 + y^2')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
