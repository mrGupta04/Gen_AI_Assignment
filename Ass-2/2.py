import numpy as np
import matplotlib.pyplot as plt

class VanillaRNNWithDropout:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, dropout_rate=0.3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # Initialize weights
        self.Wx = np.random.randn(hidden_size, input_size) * 0.01
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, train=True):
        h = np.zeros((self.hidden_size, 1))
        self.last_inputs = inputs
        self.last_hs = {-1: h}
        self.dropout_masks = {}
        outputs = []

        for t, x in enumerate(inputs):
            x = x.reshape(-1, 1)
            h = np.tanh(self.Wx @ x + self.Wh @ self.last_hs[t - 1] + self.bh)
            
            if train:
                dropout_mask = (np.random.rand(*h.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                h *= dropout_mask
                self.dropout_masks[t] = dropout_mask

            y = self.Wy @ h + self.by
            outputs.append(y)
            self.last_hs[t] = h

        return outputs, self.last_hs

    def train(self, inputs, targets):
        outputs, _ = self.forward(inputs, train=True)
        doutputs = [2 * (out - target.reshape(-1, 1)) for out, target in zip(outputs, targets)]
        loss = sum(np.mean((out - target.reshape(-1, 1))**2) for out, target in zip(outputs, targets))
        return loss


def train_with_different_batch_sizes():
    batch_sizes = [16, 32, 64, 128]
    input_size, hidden_size, output_size = 10, 20, 1
    inputs = [np.random.randn(input_size) for _ in range(10)]
    targets = [np.random.randn(output_size) for _ in range(10)]

    for batch_size in batch_sizes:
        rnn = VanillaRNNWithDropout(input_size, hidden_size, output_size)
        losses = []

        for epoch in range(50):
            batch_loss = 0
            for _ in range(batch_size):
                loss = rnn.train(inputs, targets)
                batch_loss += loss
            losses.append(batch_loss / batch_size)

        print(f"Batch size {batch_size}: Final loss = {losses[-1]:.4f}")


def fine_tune_rnn(pretrained_rnn, new_data, new_targets, fine_tune_epochs=20):
    for epoch in range(fine_tune_epochs):
        loss = pretrained_rnn.train(new_data, new_targets)
        if epoch % 5 == 0:
            print(f"Fine-tune Epoch {epoch}, Loss: {loss:.4f}")
    return pretrained_rnn


def visualize_hidden_states(rnn):
    inputs = [np.random.randn(rnn.input_size) for _ in range(10)]
    _, hidden_states = rnn.forward(inputs, train=False)

    hidden_states_matrix = np.hstack([hidden_states[t] for t in range(len(inputs))])

    plt.figure(figsize=(10, 6))
    for i in range(hidden_states_matrix.shape[0]):
        plt.plot(hidden_states_matrix[i], label=f"Hidden Unit {i}")

    plt.title("Hidden States Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Activation Value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    input_size, hidden_size, output_size = 10, 20, 1
    rnn = VanillaRNNWithDropout(input_size, hidden_size, output_size)
    
    print("Training with different batch sizes:")
    train_with_different_batch_sizes()
    
    print("\nFine-tuning pretrained RNN:")
    new_data = [np.random.randn(input_size) for _ in range(10)]
    new_targets = [np.random.randn(output_size) for _ in range(10)]
    fine_tuned_rnn = fine_tune_rnn(rnn, new_data, new_targets)
    
    print("\nVisualizing hidden states:")
    visualize_hidden_states(rnn)
