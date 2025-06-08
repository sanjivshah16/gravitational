"""
Gravitational Multi-Timescale Attention Model Implementation

This file implements the combined gravitational + multi-timescale attention mechanism
for comparison with conventional, gravitational-only, and multi-timescale-only variants.
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


class GravitationalMultiTimescaleAttention(nn.Module):
    """
    Combined gravitational and multi-timescale attention mechanism where tokens interact
    based on their "masses" at multiple timescales, similar to gravitational fields
    with different temporal horizons.
    """
    
    def __init__(self, d_model, n_heads, n_timescales=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_timescales = n_timescales
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_linear = nn.Linear(d_model * n_timescales, d_model)
        
        # Mass projections - generates a vector of "masses" for each token at different timescales
        self.mass_q_projection = nn.Linear(d_model, n_heads * n_timescales)
        self.mass_k_projection = nn.Linear(d_model, n_heads * n_timescales)
        
        # Gravitational constant (learnable parameter per head and timescale)
        self.G = nn.Parameter(torch.ones(n_timescales, n_heads, 1, 1))
        
        # Learned discount factors for each timescale (from short to long range)
        # Initialize with different values to encourage diversity
        init_gammas = torch.tensor([0.6, 0.9, 0.99])[:n_timescales]
        self.gammas = nn.Parameter(init_gammas.unsqueeze(0).unsqueeze(0).unsqueeze(-1))  # [1, 1, n_timescales, 1]
        
        # Laplace-like decoder (inverse mapping)
        self.decoder = nn.Sequential(
            nn.Linear(d_model * n_timescales, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
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
            Attention weights for visualization [batch_size, n_timescales, n_heads, seq_length, seq_length]
        """
        batch_size = query.shape[0]
        q_len, k_len = query.shape[1], key.shape[1]
        
        # Linear projections
        Q = self.q_linear(query)  # [B, L_q, D]
        K = self.k_linear(key)    # [B, L_k, D]
        V = self.v_linear(value)  # [B, L_k, D]
        
        # Calculate "masses" for each token at each timescale (positive values)
        # [B, L_q, H*T] -> [B, L_q, T, H]
        query_mass = F.softplus(self.mass_q_projection(query)).view(
            batch_size, q_len, self.n_timescales, self.n_heads
        )
        # [B, L_k, H*T] -> [B, L_k, T, H]
        key_mass = F.softplus(self.mass_k_projection(key)).view(
            batch_size, k_len, self.n_timescales, self.n_heads
        )
        
        # Store outputs from each timescale
        timescale_outputs = []
        all_attention_weights = []
        
        # Process each timescale separately
        for t in range(self.n_timescales):
            # Reshape Q, K for this timescale's processing
            Q_t = Q.view(batch_size, q_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L_q, D/H]
            K_t = K.view(batch_size, k_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L_k, D/H]
            V_t = V.view(batch_size, k_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L_k, D/H]
            
            # Extract masses for this timescale
            q_mass_t = query_mass[:, :, t, :].permute(0, 2, 1).unsqueeze(-1)  # [B, H, L_q, 1]
            k_mass_t = key_mass[:, :, t, :].permute(0, 2, 1).unsqueeze(2)     # [B, H, 1, L_k]
            
            # Calculate distances between query and key vectors
            # Euclidean distance in the embedding space
            Q_expanded = Q_t.unsqueeze(3)  # [B, H, L_q, 1, D/H]
            K_expanded = K_t.unsqueeze(2)  # [B, H, 1, L_k, D/H]
            
            # Calculate squared distances
            distances = torch.sum((Q_expanded - K_expanded) ** 2, dim=-1)  # [B, H, L_q, L_k]
            
            # Add small epsilon to prevent division by zero
            distances = distances + self.epsilon
            
            # Create position difference matrix for temporal discounting
            pos_q = torch.arange(q_len, device=query.device).unsqueeze(1)  # [L_q, 1]
            pos_k = torch.arange(k_len, device=query.device).unsqueeze(0)  # [1, L_k]
            pos_diff = torch.abs(pos_q - pos_k)  # [L_q, L_k]
            
            # Apply timescale-specific discount to distances
            # gamma^|i-j| where gamma is the discount factor for this timescale
            temporal_discount = self.gammas[:, :, t, :] ** pos_diff.unsqueeze(0).unsqueeze(0)  # [1, 1, L_q, L_k]
            
            # Apply inverse square law with gravitational constant and masses
            # G * (m_q * m_k) / (r^2 * temporal_discount)
            gravity = self.G[t] * q_mass_t * k_mass_t / (distances ** (self.distance_power / 2))  # [B, H, L_q, L_k]
            
            # Apply temporal discount
            gravity = gravity * temporal_discount
            
            # Apply mask if provided
            if mask is not None:
                gravity = gravity.masked_fill(mask == 0, 0)
            
            # Normalize gravity scores (row-wise normalization)
            gravity_sum = gravity.sum(dim=-1, keepdim=True)
            attention = gravity / (gravity_sum + self.epsilon)  # [B, H, L_q, L_k]
            
            # Apply dropout
            attention = self.dropout(attention)
            all_attention_weights.append(attention)
            
            # Apply attention weights to values
            x = torch.matmul(attention, V_t)  # [B, H, L_q, D/H]
            
            # Reshape
            x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)  # [B, L_q, D]
            
            # Store output for this timescale
            timescale_outputs.append(x)
        
        # Concatenate outputs from all timescales
        multi_timescale_output = torch.cat(timescale_outputs, dim=-1)  # [B, L_q, D*n_timescales]
        
        # Apply Laplace-like decoder (inverse mapping)
        output = self.decoder(multi_timescale_output)  # [B, L_q, D]
        
        # Stack attention weights for visualization
        stacked_attention = torch.stack(all_attention_weights, dim=1)  # [B, n_timescales, H, L_q, L_k]
        
        return output, stacked_attention


class GravitationalMultiTimescaleTransformerLayer(nn.Module):
    """A single layer of the Transformer encoder with gravitational multi-timescale attention."""
    
    def __init__(self, d_model, n_heads, d_ff, n_timescales=3, dropout=0.1):
        super().__init__()
        
        # Gravitational multi-timescale attention
        self.attention = GravitationalMultiTimescaleAttention(d_model, n_heads, n_timescales, dropout)
        
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
            Attention weights for visualization [batch_size, n_timescales, n_heads, seq_length, seq_length]
        """
        # Gravitational multi-timescale attention with residual connection and layer norm
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network with residual connection and layer norm
        ff_output = self.ff_network(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class GravitationalMultiTimescaleTransformerEncoder(nn.Module):
    """Transformer encoder with gravitational multi-timescale attention for sequence classification."""
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length, 
                 num_classes, n_timescales=3, dropout=0.1, pad_idx=0):
        super().__init__()
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder layers with gravitational multi-timescale attention
        self.layers = nn.ModuleList([
            GravitationalMultiTimescaleTransformerLayer(d_model, n_heads, d_ff, n_timescales, dropout)
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


def create_gravitational_multi_timescale_model(config):
    """
    Factory function to create a gravitational multi-timescale transformer model based on config.
    
    Args:
        config: Dictionary containing model configuration
    Returns:
        Initialized model
    """
    return GravitationalMultiTimescaleTransformerEncoder(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_length=config['max_seq_length'],
        num_classes=config['num_classes'],
        n_timescales=config.get('n_timescales', 3),
        dropout=config['dropout'],
        pad_idx=config['pad_idx']
    )
