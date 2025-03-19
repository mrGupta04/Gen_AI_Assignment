import numpy as np

class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights
        self.Wx = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output

        # Biases
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias
        self.by = np.zeros((output_size, 1))  # Output bias

    def forward(self, inputs):
       
        h = np.zeros((self.hidden_size, 1))  # Initialize hidden state
        self.last_inputs = inputs
        self.last_hs = { -1: h }

        outputs = []
        
        for t, x in enumerate(inputs):
            x = x.reshape(-1, 1)  # Ensure column vector
            h = np.tanh(self.Wx @ x + self.Wh @ self.last_hs[t - 1] + self.bh)
            y = self.Wy @ h + self.by
            outputs.append(y)
            self.last_hs[t] = h  # Store hidden state
        
        return outputs, self.last_hs

    def backward(self, doutputs):
      
        dWx, dWh, dWy = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.Wy)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        
        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(self.last_inputs))):
            dy = doutputs[t]  # Gradient of loss w.r.t. output
            dWy += dy @ self.last_hs[t].T
            dby += dy

            dh = self.Wy.T @ dy + dh_next  # Backprop into hidden state
            dh_raw = (1 - self.last_hs[t] ** 2) * dh  # Derivative of tanh
            dbh += dh_raw
            dWx += dh_raw @ self.last_inputs[t].reshape(1, -1)
            dWh += dh_raw @ self.last_hs[t - 1].T
            dh_next = self.Wh.T @ dh_raw  # Pass gradient back to previous timestep
        
        # Gradient descent update
        for param, dparam in zip([self.Wx, self.Wh, self.Wy, self.bh, self.by], 
                                 [dWx, dWh, dWy, dbh, dby]):
            param -= self.learning_rate * dparam

    def train(self, inputs, targets):
       
        outputs, _ = self.forward(inputs)
        
        # Compute loss gradients (Mean Squared Error Loss)
        doutputs = [2 * (out - target.reshape(-1, 1)) for out, target in zip(outputs, targets)]
        
        self.backward(doutputs)
        
        loss = sum(np.mean((out - target.reshape(-1, 1))**2) for out, target in zip(outputs, targets))
        return loss

# Example usage
np.random.seed(0)
input_size = 3
hidden_size = 5
output_size = 2
rnn = VanillaRNN(input_size, hidden_size, output_size, learning_rate=0.01)

# Example data (sequence length = 4)
inputs = [np.random.randn(input_size) for _ in range(4)]
targets = [np.random.randn(output_size) for _ in range(4)]

# Train for 100 epochs
for epoch in range(100):
    loss = rnn.train(inputs, targets)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
