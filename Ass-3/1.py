import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader



# ====================== 1. Scaled Dot-Product Attention in NumPy ======================


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention implementation using NumPy
    
    Args:
        Q: Queries matrix, shape (..., seq_len_q, d_k)
        K: Keys matrix, shape (..., seq_len_k, d_k)
        V: Values matrix, shape (..., seq_len_k, d_v)
        mask: Float mask (0 or -inf), shape (..., seq_len_q, seq_len_k)
    
    Returns:
        output: Attention-weighted values, shape (..., seq_len_q, d_v)
        attention_weights: Softmax scores, shape (..., seq_len_q, seq_len_k)
    """
    # Calculate dot products between queries and keys
    matmul_qk = np.matmul(Q, np.swapaxes(K, -2, -1))  # (..., seq_len_q, seq_len_k)
    
    # Scale by square root of key dimension
    d_k = Q.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scaled_attention_logits += mask
    
    # Softmax to get attention weights
    attention_weights = np.exp(scaled_attention_logits - 
                              np.max(scaled_attention_logits, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    # Multiply by values to get output
    output = np.matmul(attention_weights, V)  # (..., seq_len_q, d_v)
    
    return output, attention_weights

# ====================== 2. Transformer Encoder in PyTorch ======================


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_logits = matmul_qk / math.sqrt(self.depth)
        
        if mask is not None:
            scaled_attention_logits += mask
        
        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        return self.dense(output), attention_weights

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFFN, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size,
                 maximum_position_encoding=10000, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, d_model)
        
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
    def positional_encoding(self, max_len, d_model):
      position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
      div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * 
        (-math.log(10000.0) / d_model)
      )
    
      pe = torch.zeros(max_len, d_model)
      pe[:, 0::2] = torch.sin(position * div_term)
      pe[:, 1::2] = torch.cos(position * div_term)
    
      return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)
            
        return x
def train_transformer_encoder():
    # Hyperparameters
    num_layers = 2
    d_model = 128
    num_heads = 8
    d_ff = 512
    dropout = 0.1
    vocab_size = 32  # Small vocabulary for example
    batch_size = 32
    seq_len = 10
    num_batches = 100
    num_epochs = 10
    
    # Create model
    model = TransformerEncoder(num_layers, d_model, num_heads, d_ff, vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for _ in range(num_batches):
            # Generate random sequences
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = inputs.clone()  # Simple copy task
            
            # Forward pass
            outputs = model(inputs)  # Shape: [batch_size, seq_len, d_model]
            
            # Project outputs to vocabulary size
            logits = nn.Linear(d_model, vocab_size)(outputs)  # Shape: [batch_size, seq_len, vocab_size]
            
            # Reshape for loss calculation
            logits = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
            targets = targets.view(-1)  # [batch_size * seq_len]
            
            # Calculate loss and update
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/num_batches:.4f}")


# ====================== 3. Adaptive Attention Span ======================


class AdaptiveMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_span, adapt_span_enabled=True):
        super(AdaptiveMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.max_span = max_span
        self.adapt_span_enabled = adapt_span_enabled
        
        # Projection layers
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        
        # Adaptive span parameters
        if adapt_span_enabled:
            self.span_weights = nn.Parameter(torch.randn(num_heads))
            self.span_bias = nn.Parameter(torch.zeros(num_heads))
            self.softplus = nn.Softplus()
    
    def get_adaptive_span_mask(self, seq_len):
        """Create mask that limits attention to learned spans for each head"""
        if not self.adapt_span_enabled or not self.training:
            return None
        
        # Calculate effective span for each head
        effective_span = self.softplus(self.span_weights + self.span_bias) * self.max_span
        effective_span = effective_span.unsqueeze(-1).unsqueeze(-1)  # (num_heads, 1, 1)
        
        # Create relative positions (seq_len, seq_len)
        positions = torch.arange(seq_len).view(1, 1, seq_len) - torch.arange(seq_len).view(1, seq_len, 1)
        positions = positions.float().abs().unsqueeze(0)  # (1, seq_len, seq_len)
        
        # Create mask (num_heads, seq_len, seq_len)
        mask = (positions > effective_span).float() * -1e9
        return mask
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len = q.size(0), q.size(1)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Get adaptive span mask if enabled
        adapt_mask = self.get_adaptive_span_mask(seq_len)
        if adapt_mask is not None:
            if mask is None:
                mask = adapt_mask
            else:
                mask = mask + adapt_mask
        
        # Scaled dot-product attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_logits = matmul_qk / math.sqrt(self.depth)
        
        if mask is not None:
            scaled_attention_logits += mask
        
        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        return self.dense(output), attention_weights


# ====================== 4. Time-Series Forecasting ======================


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, num_heads=4, num_layers=3, d_ff=256, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, 1)
    
    def forward(self, src):
        # src shape: (batch_size, seq_len, input_size)
        src = self.input_proj(src)  # (batch_size, seq_len, d_model)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model) for Transformer
        
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        
        # Take last time step's output
        output = output[-1, :, :]  # (batch_size, d_model)
        output = self.decoder(output)  # (batch_size, 1)
        
        return output

class TimeSeriesRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super(TimeSeriesRNN, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        output, _ = self.rnn(x)  # (batch_size, seq_len, hidden_size)
        output = output[:, -1, :]  # Take last time step
        output = self.decoder(output)
        return output

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_time_series_data(n_samples=1000, seq_len=50):
    time = np.arange(n_samples * seq_len, dtype=np.float32) * 0.01
    data = np.sin(time) + np.sin(0.5 * time) + np.random.normal(0, 0.1, len(time))
    data = data.reshape(n_samples, seq_len, 1)
    
    # Split into train/test
    train_size = int(0.8 * n_samples)
    X_train, y_train = data[:train_size, :-1], data[:train_size, -1]
    X_test, y_test = data[train_size:, :-1], data[train_size:, -1]
    
    # Normalize
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    y_train = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_test = scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    
    return X_train, y_train, X_test, y_test

def train_and_compare():
    # Generate data
    X_train, y_train, X_test, y_test = generate_time_series_data()
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize models
    transformer_model = TimeSeriesTransformer()
    rnn_model = TimeSeriesRNN()
    
    # Training function
    def train_model(model, train_loader, test_loader, epochs=20):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch)
                    test_loss += criterion(outputs, y_batch).item()
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Test Loss: {test_loss/len(test_loader):.4f}")
    
    print("Training Transformer...")
    train_model(transformer_model, train_loader, test_loader)
    
    print("\nTraining RNN...")
    train_model(rnn_model, train_loader, test_loader)

# ====================== Main Execution ======================


if __name__ == "__main__":
    print("1. Testing Scaled Dot-Product Attention with NumPy")
    Q = np.random.randn(2, 4, 3)
    K = np.random.randn(2, 4, 3)
    V = np.random.randn(2, 4, 5)
    output, attn_weights = scaled_dot_product_attention(Q, K, V)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape)
    
    print("\n2. Training Transformer Encoder on Copy Task")
    train_transformer_encoder()
    
    print("\n4. Comparing Transformer and RNN on Time-Series Forecasting")
    train_and_compare()
