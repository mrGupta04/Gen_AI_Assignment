import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os

# Ensure the GloVe file exists
glove_path = r"C:\Users\adiap\OneDrive\Documents\Gen Ai\Ass-2\glove.6B.50d.txt"
if not os.path.exists(glove_path):
    raise FileNotFoundError(f"GloVe file not found: {glove_path}. Place it in the correct directory.")

# Load pretrained GloVe embeddings
def load_glove_embeddings(glove_path, word_index, embedding_dim=50):
    embeddings = np.random.randn(len(word_index), embedding_dim) * 0.01  # Small random initialization
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype=np.float32)
            if word in word_index:
                embeddings[word_index[word]] = vector
    return torch.tensor(embeddings, dtype=torch.float32)

# Dataset class for Named Entity Recognition
class NERDataset(Dataset):
    def __init__(self, sentences, labels, word2idx, tag2idx):
        self.sentences = [[word2idx.get(w, 1) for w in sent] for sent in sentences]  # 1 for <UNK>
        self.labels = [[tag2idx[t] for t in label] for label in labels]
        self.lengths = [len(sent) for sent in self.sentences]
        
        self.sentences = [torch.tensor(s, dtype=torch.long) for s in self.sentences]
        self.labels = [torch.tensor(l, dtype=torch.long) for l in self.labels]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx], self.lengths[idx]

# Collate function for variable-length sequences
def collate_fn(batch):
    sentences, labels, lengths = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)  # -1 for masked loss
    return sentences_padded, labels_padded, torch.tensor(lengths, dtype=torch.long)

# BiLSTM Model for Named Entity Recognition
class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, pretrained_embeddings):
        super(BiLSTM_NER, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentences, lengths):
        embeds = self.embedding(sentences)
        packed_embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(lstm_out)
        return logits

# Training function
def train_model(model, train_loader, optimizer, loss_fn, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for sentences, labels, lengths in train_loader:
            sentences, labels, lengths = sentences.to(device), labels.to(device), lengths.to(device)
            optimizer.zero_grad()
            logits = model(sentences, lengths)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

# Evaluation function
@torch.no_grad()
def evaluate_model(model, val_loader, idx2tag, device):
    model.eval()
    all_preds, all_labels = [], []
    
    for sentences, labels, lengths in val_loader:
        sentences, labels, lengths = sentences.to(device), labels.to(device), lengths.to(device)
        logits = model(sentences, lengths)
        preds = torch.argmax(logits, dim=-1)
        
        for i in range(len(lengths)):
            all_preds.extend(preds[i][:lengths[i]].cpu().numpy())
            all_labels.extend(labels[i][:lengths[i]].cpu().numpy())

    # Fix UndefinedMetricWarning by setting zero_division=0
    print(classification_report(
        all_labels, all_preds, 
        target_names=list(idx2tag.values()), 
        labels=list(idx2tag.keys()), 
        zero_division=0  # This ensures no undefined metric warnings
    ))

# Main script
if __name__ == "__main__":
    # Sample training data
    train_sentences = [['John', 'lives', 'in', 'New', 'York']]
    train_labels = [['B-PER', 'O', 'O', 'B-LOC', 'I-LOC']]

    word2idx = {'<PAD>': 0, '<UNK>': 1, 'John': 2, 'lives': 3, 'in': 4, 'New': 5, 'York': 6}
    tag2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4}
    idx2tag = {v: k for k, v in tag2idx.items()}  # Reverse mapping for evaluation

    # Create dataset and data loader
    train_dataset = NERDataset(train_sentences, train_labels, word2idx, tag2idx)
    train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)

    # Load GloVe embeddings
    pretrained_embeddings = load_glove_embeddings(glove_path, word2idx)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM_NER(len(word2idx), 50, 128, len(tag2idx), pretrained_embeddings).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # Train and evaluate
    train_model(model, train_loader, optimizer, loss_fn, device, epochs=10)
    evaluate_model(model, train_loader, idx2tag, device)
