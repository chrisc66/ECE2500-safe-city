import logging
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import geopandas as gpd

from torch.utils.data import Dataset
from node2vec import Node2Vec


def generate_node2vec_embeddings(neighbourhood_info_df, node2vec_dim=32):
    """
    Generates Node2Vec embeddings for neighborhoods based on a sample adjacency graph.
    node2vec_dim: Dimension of the embeddings to be generated.
    """

    logger = logging.getLogger(__name__)
    logger.info(f"generate_node2vec_embeddings start")
    num_neighborhood_info = len(neighbourhood_info_df)
    G = nx.Graph()

    # Convert "Geometry Multipolygon" column to GeoSeries
    neighbourhood_info_df["geometry"] = gpd.GeoSeries.from_wkt(neighbourhood_info_df["Geometry Multipolygon"])
    neighbourhood_info_df["nid"] = range(num_neighborhood_info)

    # Assuming you have a way to define neighborhood connections based on proximity
    # You can use the geometry information for this.
    # Here's a placeholder for how you might connect neighborhoods based on proximity:

    for i in range(num_neighborhood_info):
        for j in range(i + 1, num_neighborhood_info):
            # Use the new 'geometry' column for spatial operations
            if neighbourhood_info_df["geometry"].iloc[i].intersects(neighbourhood_info_df["geometry"].iloc[j]):
                G.add_edge(i, j)

    # Alternatively, you could build a graph based on other criteria like sharing a boundary
    node2vec = Node2Vec(G, dimensions=node2vec_dim, walk_length=10, num_walks=100, p=1, q=1)
    node2vec_model = node2vec.fit()
    node2vec_embeddings_np = np.array([node2vec_model.wv[str(i)] for i in range(num_neighborhood_info)])
    node2vec_embeddings = torch.from_numpy(node2vec_embeddings_np)
    node2vec_emb_layer = nn.Embedding.from_pretrained(node2vec_embeddings, freeze=True)

    logger.info(f"generate_node2vec_embeddings finish")
    return node2vec_emb_layer


class Time2Vec(nn.Module):
    """
    Time2Vec embedding module for temporal features

    This captures both linear and periodic components for time-based features.
    """

    def __init__(self, input_dim, embed_dim, act_function=torch.sin):
        super(Time2Vec, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.embed_dim = embed_dim // input_dim  # Embedding dimension per time feature
        self.act_function = act_function  # Activation function for periodicity
        self.weight = nn.Parameter(torch.randn(input_dim, self.embed_dim))
        self.bias = nn.Parameter(torch.randn(input_dim, self.embed_dim))

    def forward(self, x):
        # Diagonal embedding for each time feature (day of week, hour, etc.)
        x = torch.diag_embed(x)
        x_affine = torch.matmul(x, self.weight) + self.bias
        x_affine_0, x_affine_remain = torch.split(x_affine, [1, self.embed_dim - 1], dim=-1)
        x_affine_remain = self.act_function(x_affine_remain)
        return torch.cat([x_affine_0, x_affine_remain], dim=-1).view(x.size(0), x.size(1), -1)


class NeighborhoodDataset(Dataset):
    def __init__(
        self,
        neighborhood_ids,
        time_features,
        building_type_ids,
        building_counts,
        population,
        event_type_ids,
        equipment_ids,
        targets,
    ):
        self.logger = logging.getLogger(__name__)
        self.neighborhood_ids = neighborhood_ids  # Tensor of input neighborhood_ids
        self.time_features = time_features  # Tensor of input time_features
        self.building_type_ids = building_type_ids  # Tensor of input building_type_ids
        self.building_counts = building_counts  # Tensor of input building_counts
        self.population = population  # Tensor of input population
        self.event_type_ids = event_type_ids  # Tensor of input event_type_ids
        self.equipment_ids = equipment_ids  # Tensor of input equipment_ids
        self.targets = targets  # Tensor of target values

    def __len__(self):
        return len(self.neighborhood_ids)  # Number of neighborhoods

    def __getitem__(self, idx):
        return (
            self.neighborhood_ids[idx],
            self.time_features[idx],
            self.building_type_ids[idx],
            self.building_counts[idx],
            self.population[idx],
            self.event_type_ids[idx],
            self.equipment_ids[idx],
            self.targets[idx],
        )


class CombinedEmbedding(nn.Module):
    """
    Combined Embedding Module

    Combines embeddings from Node2Vec, Time2Vec, building type/counts, population, event type, and equipment.
    Projects the combined embedding to a target dimension (e.g., 64) for compatibility with transformer layers.
    """

    def __init__(
        self,
        node2vec_emb_layer,
        time2vec_embed_dim,
        time_feature_dim,
        num_building_types,  # unused
        building_type_embed_dim,
        population_embed_dim,
        num_event_types,
        event_type_embed_dim,
        num_equipment_types,
        equipment_embed_dim,
        target_embed_dim=64,
    ):  # Add target_embed_dim for projection
        super(CombinedEmbedding, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.node2vec_emb_layer = node2vec_emb_layer  # Precomputed Node2Vec embeddings
        self.time2vec = Time2Vec(input_dim=time_feature_dim, embed_dim=time2vec_embed_dim)
        self.building_type_embedding = nn.Embedding(num_building_types, building_type_embed_dim)
        self.population_embedding = nn.Linear(1, population_embed_dim)
        # self.income_embedding = nn.Linear(1, income_embed_dim)
        self.event_type_embedding = nn.Embedding(num_event_types, event_type_embed_dim)
        self.equipment_embedding = nn.Embedding(num_equipment_types, equipment_embed_dim)

        # Compute the combined embedding dimension before projection
        self.projection_dim = (
            node2vec_emb_layer.embedding_dim
            + time2vec_embed_dim
            + building_type_embed_dim
            + population_embed_dim
            + event_type_embed_dim
            + equipment_embed_dim
        )

        # Projection layer to reduce to target_embed_dim
        self.projection_layer = nn.Linear(self.projection_dim, target_embed_dim)

    def forward(
        self,
        neighborhood_ids,
        time_features,
        building_type_ids,
        building_counts,
        population,
        event_type_ids,
        equipment_ids,
    ):
        # Generate embeddings
        spatial_embeddings = self.node2vec_emb_layer(neighborhood_ids.long()).squeeze(2)
        self.logger.info(f"Spatial Embedding Shape: {spatial_embeddings.shape}")

        temporal_embeddings = self.time2vec(time_features)
        self.logger.info(f"Temporal Embedding Shape: {temporal_embeddings.shape}")

        # Building embeddings
        building_type_embeds = self.building_type_embedding(building_type_ids.long())
        building_type_embeds = building_type_embeds[:, :, 0, :]
        self.logger.info(f"Building Type Embeds Shape: {building_type_embeds.shape}")

        # Adjust building_counts to match building_type_embed_dim
        building_counts = building_counts.unsqueeze(-1)
        building_counts = building_counts.repeat(1, 1, 1, self.building_type_embedding.embedding_dim)
        building_counts = building_counts[:, :, 0, :]
        self.logger.info(f"Building Counts Shape after adjustment: {building_counts.shape}")

        # Multiply and aggregate
        building_embeddings = (building_type_embeds.unsqueeze(1) * building_counts).sum(dim=1)
        self.logger.info(f"Building Embedding Shape: {building_embeddings.shape}")

        population = population.unsqueeze(-1)
        population_embeddings = self.population_embedding(population).squeeze(2)
        self.logger.info(f"Population Embedding Shape: {population_embeddings.shape}")

        event_type_embeddings = self.event_type_embedding(event_type_ids).squeeze(2)
        self.logger.info(f"Event Type Embedding Shape: {event_type_embeddings.shape}")

        equipment_embeddings = self.equipment_embedding(equipment_ids).squeeze(2)
        self.logger.info(f"Equipment Embedding Shape: {equipment_embeddings.shape}")

        combined_embedding = torch.cat(
            [
                spatial_embeddings,
                temporal_embeddings,
                building_embeddings,
                population_embeddings,
                event_type_embeddings,
                equipment_embeddings,
            ],
            dim=-1,
        )
        self.logger.info(f"Combined Embedding Shape before projection: {combined_embedding.shape}")

        combined_embedding = self.projection_layer(combined_embedding)
        self.logger.info(f"Combined Embedding Shape after projection: {combined_embedding.shape}")

        return combined_embedding
