import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F


# Load a sample text dataset
text = "hello world! this is an example of text generation using RNNs."

# Create character mappings
chars = sorted(list(set(text)))  # Unique characters
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for ch, i in char_to_ix.items()}
vocab_size = len(chars)

# Convert text into integer sequences
def text_to_tensor(text):
    return torch.tensor([char_to_ix[ch] for ch in text], dtype=torch.long)

text_tensor = text_to_tensor(text)

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Convert char indices to embeddings
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)  # Output layer
        
    def forward(self, x, hidden):
        x = self.embedding(x)  # Convert indices to embeddings
        out, hidden = self.rnn(x, hidden)  # Pass through RNN
        out = self.fc(out)  # Fully connected layer
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)  # Initialize hidden state

def train(model, text_tensor, epochs=100, lr=0.01, teacher_forcing_ratio=0.5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        hidden = model.init_hidden(1)  # Initialize hidden state
        loss = 0
        
        for i in range(len(text_tensor) - 1):
            x = text_tensor[i].unsqueeze(0).unsqueeze(0)  # Input char
            target = text_tensor[i+1].unsqueeze(0)  # Target char
            
            if random.random() < teacher_forcing_ratio:
                out, hidden = model(x, hidden)
            else:
                with torch.no_grad():
                    out, hidden = model(x, hidden)
                out = out.argmax(dim=-1)  # Greedy prediction
                out, hidden = model(out, hidden)
            
            loss += criterion(out.squeeze(0), target)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Initialize model
embed_size = 16
hidden_size = 128
num_layers = 2

model = CharRNN(vocab_size, embed_size, hidden_size, num_layers)
train(model, text_tensor, epochs=200)



def generate_text(model, start_str, length=100, temperature=1.0):
    model.eval()
    hidden = model.init_hidden(1)
    
    # Convert start string to tensor
    input_seq = text_to_tensor(start_str)
    input_seq = input_seq.unsqueeze(0)  # Add batch dimension
    
    output_text = start_str
    
    for _ in range(length):
        out, hidden = model(input_seq[:, -1:], hidden)  # Get last character's output
        out = out.squeeze(0)  # Remove batch dimension
        
        # Apply temperature sampling
        out = out / temperature
        probabilities = F.softmax(out, dim=-1).detach().numpy()
        next_char_idx = np.random.choice(range(vocab_size), p=probabilities.ravel())
        
        output_text += ix_to_char[next_char_idx]
        input_seq = torch.cat((input_seq, torch.tensor([[next_char_idx]])), dim=1)
    
    return output_text

# Generate text using trained model
print(generate_text(model, "hello", length=100, temperature=0.8))


text = "hello world this is an example of word level text generation using RNN."
words = text.split()
word_to_ix = {word: i for i, word in enumerate(set(words))}
ix_to_word = {i: word for word, i in word_to_ix.items()}
vocab_size = len(word_to_ix)

def words_to_tensor(words):
    return torch.tensor([word_to_ix[word] for word in words], dtype=torch.long)

word_tensor = words_to_tensor(words)

class WordRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(WordRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)  # Initialize hidden state


# Train word-level model
word_model = WordRNN(vocab_size, embed_size=16, hidden_size=128, num_layers=2)
train(word_model, word_tensor, epochs=200)

# Generate text at word level
def generate_words(model, start_word, length=20, temperature=1.0):
    model.eval()
    hidden = model.init_hidden(1)
    
    input_seq = torch.tensor([[word_to_ix[start_word]]])
    output_text = start_word
    
    for _ in range(length):
        out, hidden = model(input_seq, hidden)
        out = out.squeeze(0) / temperature
        probabilities = F.softmax(out, dim=-1).detach().numpy()
        next_word_idx = np.random.choice(range(vocab_size), p=probabilities.ravel())
        
        output_text += " " + ix_to_word[next_word_idx]
        input_seq = torch.tensor([[next_word_idx]])
    
    return output_text

print(generate_words(word_model, "hello", length=20, temperature=0.8))
