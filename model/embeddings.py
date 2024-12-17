import logging
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import geopandas as gpd

from torch.utils.data import Dataset

class SpatiotemporalEmbedding(nn.Module):
    """
    Embedding layer for spatiotemporal data with multiple input features.
    Combines embeddings for time features, spatial IDs, population, equipment IDs,
    event counts, event types, building counts, and building type IDs, along with position embeddings.
    """
    def __init__(self, time_dim, num_spatial_ids, num_event_types, num_building_types, num_equipment_ids, model_dim):
        """
        Args:
            time_dim (int): Number of time features (e.g., hour, day_of_week, week, year).
            num_spatial_ids (int): Number of unique spatial entities (e.g., neighborhoods).
            num_event_types (int): Number of unique event types (categorical).
            num_building_types (int): Number of unique building types (categorical).
            num_equipment_ids (int): Number of unique equipment types (categorical).
            model_dim (int): Dimension of the output embeddings (d_model).
        """
        super(SpatiotemporalEmbedding, self).__init__()

        self.time_embedding = nn.Linear(time_dim, model_dim)
        self.spatial_embedding = nn.Embedding(num_spatial_ids, model_dim)
        self.population_embedding = nn.Linear(1, model_dim)
        self.event_count_embedding = nn.Linear(1, model_dim)  # Numerical input
        self.event_type_embedding = nn.Embedding(num_event_types, model_dim)  # Categorical input
        self.building_count_embedding = nn.Linear(1, model_dim)  # Numerical input
        self.building_type_embedding = nn.Embedding(num_building_types, model_dim)  # Categorical input
        self.equipment_embedding = nn.Embedding(num_equipment_ids, model_dim)
        self.position_embedding = nn.Embedding(10000, model_dim)  # Max sequence length

    def forward(self, time_features, spatial_ids, population, event_counts, event_types, building_counts,
                building_type_ids, equipment_ids, positions):
        """
        Forward pass to generate spatiotemporal embeddings.

        Args:
            time_features (Tensor): Time features, shape (batch_size, seq_len, time_dim).
            spatial_ids (Tensor): Spatial IDs, shape (batch_size, seq_len).
            population (Tensor): Population data, shape (batch_size, seq_len).
            event_counts (Tensor): Event counts, shape (batch_size, seq_len).
            event_types (Tensor): Event types, shape (batch_size, seq_len).
            building_counts (Tensor): Building counts, shape (batch_size, seq_len).
            building_type_ids (Tensor): Building type IDs, shape (batch_size, seq_len).
            equipment_ids (Tensor): Equipment type IDs, shape (batch_size, seq_len).
            positions (Tensor): Position indices, shape (batch_size, seq_len).

        Returns:
            Tensor: Combined spatiotemporal embeddings, shape (batch_size, seq_len, model_dim).
        """
        # Compute embeddings
        time_embeds = self.time_embedding(time_features)  # Time features
        spatial_embeds = self.spatial_embedding(spatial_ids)  # Spatial IDs
        population_embeds = self.population_embedding(population.unsqueeze(-1))  # Population
        event_count_embeds = self.event_count_embedding(event_counts.unsqueeze(-1))  # Event counts
        event_type_embeds = self.event_type_embedding(event_types)  # Event types
        building_count_embeds = self.building_count_embedding(building_counts.unsqueeze(-1))  # Building counts
        building_type_embeds = self.building_type_embedding(building_type_ids)  # Building types
        equipment_embeds = self.equipment_embedding(equipment_ids)  # Equipment IDs
        position_embeds = self.position_embedding(positions)  # Position indices

        # Combine all embeddings
        combined_embeds = (
            time_embeds +
            spatial_embeds +
            population_embeds +
            event_count_embeds +
            event_type_embeds +
            building_count_embeds +
            building_type_embeds +
            equipment_embeds +
            position_embeds
        )

        return combined_embeds  # Shape: (batch_size, seq_len, model_dim)