"""
Gravitational Attention Model Implementation

This file implements the gravitational attention mechanism for comparison with
conventional and multi-timescale attention variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding used in Transformer models."""
    
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GravitationalAttention(nn.Module):
    """
    Gravitational attention mechanism where tokens interact based on their
    "masses" and distances, similar to gravitational fields.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # Mass projection - generates a scalar "mass" for each token
        self.mass_q_projection = nn.Linear(d_model, n_heads)
        self.mass_k_projection = nn.Linear(d_model, n_heads)
        
        # Gravitational constant (learnable parameter per head)
        self.G = nn.Parameter(torch.ones(n_heads, 1, 1))
        
        # Minimum distance to prevent numerical issues
        self.epsilon = 1e-8
        
        # Distance power factor (default: 2 for inverse square law)
        self.distance_power = 2.0
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Query tensor [batch_size, seq_length, d_model]
            key: Key tensor [batch_size, seq_length, d_model]
            value: Value tensor [batch_size, seq_length, d_model]
            mask: Optional mask tensor [batch_size, 1, 1, seq_length]
        Returns:
            Output tensor after attention [batch_size, seq_length, d_model]
            Attention weights for visualization [batch_size, n_heads, seq_length, seq_length]
        """
        batch_size = query.shape[0]
        q_len, k_len = query.shape[1], key.shape[1]
        
        # Linear projections and reshape for multi-head attention
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L_q, D/H]
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)    # [B, H, L_k, D/H]
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L_k, D/H]
        
        # Calculate "masses" for each token (positive values)
        # [B, L_q, H] -> [B, H, L_q, 1]
        query_mass = F.softplus(self.mass_q_projection(query)).permute(0, 2, 1).unsqueeze(-1)
        # [B, L_k, H] -> [B, H, 1, L_k]
        key_mass = F.softplus(self.mass_k_projection(key)).permute(0, 2, 1).unsqueeze(2)
        
        # Calculate distances between query and key vectors
        # Euclidean distance in the embedding space
        # [B, H, L_q, D/H] - [B, H, 1, D/H] -> [B, H, L_q, L_k, D/H]
        Q_expanded = Q.unsqueeze(3)  # [B, H, L_q, 1, D/H]
        K_expanded = K.unsqueeze(2)  # [B, H, 1, L_k, D/H]
        
        # Calculate squared distances
        distances = torch.sum((Q_expanded - K_expanded) ** 2, dim=-1)  # [B, H, L_q, L_k]
        
        # Add small epsilon to prevent division by zero
        distances = distances + self.epsilon
        
        # Apply inverse square law (or other power) for gravitational attention
        # G * (m_q * m_k) / r^2
        gravity = self.G * query_mass * key_mass / (distances ** (self.distance_power / 2))  # [B, H, L_q, L_k]
        
        # Apply mask if provided
        if mask is not None:
            gravity = gravity.masked_fill(mask == 0, 0)
        
        # Normalize gravity scores (row-wise normalization)
        gravity_sum = gravity.sum(dim=-1, keepdim=True)
        attention = gravity / (gravity_sum + self.epsilon)  # [B, H, L_q, L_k]
        
        # Apply dropout
        attention = self.dropout(attention)
        
        # Apply attention weights to values
        x = torch.matmul(attention, V)  # [B, H, L_q, D/H]
        
        # Reshape and apply output projection
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)  # [B, L_q, D]
        output = self.out_linear(x)  # [B, L_q, D]
        
        return output, attention


class GravitationalTransformerLayer(nn.Module):
    """A single layer of the Transformer encoder with gravitational attention."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Gravitational multi-head attention
        self.attention = GravitationalAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            mask: Optional mask tensor [batch_size, 1, 1, seq_length]
        Returns:
            Output tensor after one encoder layer [batch_size, seq_length, d_model]
            Attention weights for visualization [batch_size, n_heads, seq_length, seq_length]
        """
        # Gravitational multi-head attention with residual connection and layer norm
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network with residual connection and layer norm
        ff_output = self.ff_network(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class GravitationalTransformerEncoder(nn.Module):
    """Transformer encoder with gravitational attention for sequence classification."""
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length, 
                 num_classes, dropout=0.1, pad_idx=0):
        super().__init__()
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder layers with gravitational attention
        self.layers = nn.ModuleList([
            GravitationalTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of token indices [batch_size, seq_length]
            mask: Optional mask tensor [batch_size, 1, 1, seq_length]
        Returns:
            Output logits for classification [batch_size, num_classes]
            List of attention weights from each layer for visualization
        """
        # Create embedding
        x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Store attention weights from each layer for visualization
        attention_weights = []
        
        # Apply transformer layers
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        # Global average pooling for classification
        x = torch.mean(x, dim=1)
        
        # Classification head
        logits = self.classifier(x)
        
        return logits, attention_weights


def create_gravitational_model(config):
    """
    Factory function to create a gravitational transformer model based on config.
    
    Args:
        config: Dictionary containing model configuration
    Returns:
        Initialized model
    """
    return GravitationalTransformerEncoder(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_length=config['max_seq_length'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        pad_idx=config['pad_idx']
    )
