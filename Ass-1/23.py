import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock function and its gradient
def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_grad(x, y, a=1, b=100):
    dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    dy = 2 * b * (y - x**2)
    return np.array([dx, dy])

# Adam Optimizer Implementation
def adam_optimizer(learning_rate, beta1, beta2, epsilon, epochs):
    x, y = np.random.randn(), np.random.randn()  # Initialize parameters
    m_x, m_y = 0, 0  # First moment estimates
    v_x, v_y = 0, 0  # Second moment estimates
    loss_history = []
    
    for t in range(1, epochs + 1):
        grad = rosenbrock_grad(x, y)
        
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
        
        loss_history.append(rosenbrock(x, y))  # Track the loss
    
    return loss_history

# RMSprop Optimizer Implementation
def rmsprop_optimizer(learning_rate, beta, epsilon, epochs):
    x, y = np.random.randn(), np.random.randn()  # Initialize parameters
    v_x, v_y = 0, 0  # Initialize squared gradient averages
    loss_history = []
    
    for _ in range(epochs):
        grad = rosenbrock_grad(x, y)
        
        v_x = beta * v_x + (1 - beta) * grad[0]**2
        v_y = beta * v_y + (1 - beta) * grad[1]**2
        
        x -= learning_rate * grad[0] / (np.sqrt(v_x) + epsilon)
        y -= learning_rate * grad[1] / (np.sqrt(v_y) + epsilon)
        
        loss_history.append(rosenbrock(x, y))  # Track the loss
    
    return loss_history

# Stochastic Gradient Descent (SGD) Implementation
def sgd_optimizer(learning_rate, epochs):
    x, y = np.random.randn(), np.random.randn()  # Initialize parameters
    loss_history = []
    
    for _ in range(epochs):
        grad = rosenbrock_grad(x, y)
        
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
        
        loss_history.append(rosenbrock(x, y))  # Track the loss
    
    return loss_history

# Hyperparameters
epochs = 1000
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
beta_rms = 0.9
epsilon = 1e-8

# Train with Adam, RMSprop, and SGD
loss_adam = adam_optimizer(learning_rate, beta1, beta2, epsilon, epochs)
loss_rmsprop = rmsprop_optimizer(learning_rate, beta_rms, epsilon, epochs)
loss_sgd = sgd_optimizer(learning_rate, epochs)

# Plotting the loss curves
plt.figure(figsize=(10, 6))
plt.plot(loss_adam, label='Adam', color='b')
plt.plot(loss_rmsprop, label='RMSprop', color='g')
plt.plot(loss_sgd, label='SGD', color='r')

plt.title('Loss Comparison on Rosenbrock Function')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
