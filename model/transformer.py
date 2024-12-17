import logging
import math
import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.init as init
from sklearn.metrics import mean_absolute_error, mean_squared_error
from embeddings import SpatiotemporalEmbedding


class PositionalEncoding(nn.Module):
    """
    Adds positional encodings to the input embeddings to retain sequence order.
    """
    def __init__(self, embed_dim, max_len=10000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Tensor with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class SpatiotemporalTransformer(nn.Module):
    """
    Transformer model for spatiotemporal forecasting tasks.
    Integrates the spatiotemporal embeddings and processes the data using a Transformer encoder.
    """
    def __init__(self, time_dim, num_spatial_ids, num_event_types, num_building_types, model_dim, num_heads, num_layers):
        """
        Args:
            time_dim (int): Number of time features (e.g., hour, day_of_week, week, year).
            num_spatial_ids (int): Number of unique spatial entities (neighborhoods).
            num_event_types (int): Number of unique event types (categorical).
            num_building_types (int): Number of unique building types.
            model_dim (int): Dimension of the model embeddings.
            num_heads (int): Number of attention heads in the transformer.
            num_layers (int): Number of transformer encoder layers.
        """
        super(SpatiotemporalTransformer, self).__init__()

        # Spatiotemporal Embedding
        self.embedding = SpatiotemporalEmbedding(
            time_dim=time_dim,
            num_spatial_ids=num_spatial_ids,
            num_event_types=num_event_types,
            num_building_types=num_building_types,
            num_equipment_ids=0,  # Removed equipment ID
            model_dim=model_dim
        )

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(embed_dim=model_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=512, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Layer
        self.fc_out = nn.Linear(model_dim, 1)  # Predict a single numerical target per sequence element

    def forward(self, time_features, spatial_ids, population, event_counts, event_types, building_counts,
                building_type_ids, positions):
        """
        Forward pass through the model.

        Args:
            time_features (Tensor): Shape (batch_size, seq_len, time_dim).
            spatial_ids (Tensor): Shape (batch_size, seq_len).
            population (Tensor): Shape (batch_size, seq_len).
            event_counts (Tensor): Shape (batch_size, seq_len).
            event_types (Tensor): Shape (batch_size, seq_len).
            building_counts (Tensor): Shape (batch_size, seq_len).
            building_type_ids (Tensor): Shape (batch_size, seq_len).
            positions (Tensor): Shape (batch_size, seq_len).

        Returns:
            Tensor: Predicted values, shape (batch_size, seq_len).
        """
        # Generate spatiotemporal embeddings
        embeddings = self.embedding(
            time_features, spatial_ids, population, event_counts,
            event_types, building_counts, building_type_ids, positions
        )

        # Add positional encodings
        embeddings = self.positional_encoding(embeddings)

        # Pass through the transformer encoder
        transformer_output = self.transformer_encoder(embeddings)

        # Predict output values
        output = self.fc_out(transformer_output).squeeze(-1)  # Shape: (batch_size, seq_len)
        return output