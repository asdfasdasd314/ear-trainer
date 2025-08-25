import torch
import torch.nn as nn
from pathlib import Path
import os

PROJECT_ROOT = Path(os.getcwd())
features_base = PROJECT_ROOT / 'data' / 'features'
labels_base = PROJECT_ROOT / 'data' / 'labels'
models_base = PROJECT_ROOT / 'models'

max_len = 8 # Max chords in a sequence
num_layers = 4 # Number of encoder layers
d_model = 128 # Embedding dimension, think 3Blue1Brown
input_dim = 13 # 12 chroma + 1 time
n_heads = 4 # Number of attention heads for transformer
num_classes = 12 # 12 chroma


# Create embedding for midi + duration
class ChordEmbedding(nn.Module):
    def __init__(self, input_dim=input_dim, d_model=d_model):
        super().__init__()

        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        return self.proj(x)  # (batch, seq_len, d_model)


# Create positional encoding for chords
class PositionalEncoding(nn.Module):

    def __init__(self, d_model=d_model, max_len=max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class ChordTransformer(nn.Module):

    def __init__(self, d_model=128, n_heads=8, num_layers=8, input_dim=13, num_classes=12):
        super().__init__()

        # Map chord features to transformer dimension
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final linear layer to map hidden states to chord root probabilities
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x, src_key_padding_mask=None):
        """
        x: (seq_len, d_model)
        src_key_padding_mask: (seq_len), True=padding positions
        """
        x = self.embedding(x)  # (seq_len, d_model)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        out = self.fc_out(x)  # (seq_len, num_classes)
        return out