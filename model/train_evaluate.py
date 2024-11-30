import os
import sys
import logging

logger = logging.getLogger("ST-TEM")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("gensim").setLevel(logging.ERROR)

import torch
import pandas as pd
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from transformer import PositionalEncoding, ST_TEM
from embeddings import Time2Vec, NeighborhoodDataset, CombinedEmbedding, generate_node2vec_embeddings


DATASET_PATH = "../dataset/EdmontonFireRescueServicesData"
UNIT_TRIP_PATH = os.path.join(DATASET_PATH, "EFRS_Unit_Trip_Summary.csv")
WEEKLY_EVENTS_FILTER_PATH = os.path.join(DATASET_PATH, "weekly_events_filter.csv")
NEIGHBOURHOOD_PATH = os.path.join(DATASET_PATH, "City_of_Edmonton_-_Neighbourhoods_20241022.csv")
NEIGHBOURHOOD_FEATURES_PATH = os.path.join(DATASET_PATH, "neighbourhood_features.csv")

unit_trip_df = pd.read_csv(UNIT_TRIP_PATH)
weekly_events_df = pd.read_csv(WEEKLY_EVENTS_FILTER_PATH)
neighbourhood_df = pd.read_csv(NEIGHBOURHOOD_PATH)
neighbourhood_feature_df = pd.read_csv(NEIGHBOURHOOD_FEATURES_PATH)

#################################################
# Define Parameters for Features and Embeddings
#################################################

# Feature Definitions
build_type_list = [
    "Apartment_Condo_1_to_4_stories",
    "Apartment_Condo_5_or_more_stories",
    "Duplex_Fourplex",
    "Hotel_Motel",
    "Institution_Collective_Residence",
    "Manufactured_Mobile_Home",
    "RV_Tent_Other",
    "Row_House",
    "Single_Detached_House",
]
income_list = ["Low_Income", "Low_medium_Income", "Medium_Income", "High_Income"]
event_type_list = weekly_events_df["Rc_description"].unique()
unit_type_list = unit_trip_df["unityp"].unique()

num_neighborhoods = len(neighbourhood_feature_df)
num_income = len(income_list)
num_building_types = len(build_type_list)
num_event_types = len(event_type_list)
num_equipment_types = len(unit_type_list)

neighbourhood_mappings = weekly_events_df["Neighbourhood Number"].unique()
building_counts_np = neighbourhood_feature_df[build_type_list].fillna(0).astype(int).to_numpy()
population_np = neighbourhood_feature_df["Population"].fillna(0).astype(int).to_numpy()
income_np = neighbourhood_feature_df[income_list].fillna(0).astype(int).to_numpy()

# Embedding Parameter and Module
node2vec_dim = 32
time2vec_embed_dim = 64
time_feature_dim = 2  # week, year
building_type_embed_dim = 16
population_embed_dim = 8
event_type_embed_dim = 16
equipment_embed_dim = 16
target_embed_dim = 64

mini_batch_size = 29
batch_size = 13
spatial_dimension = num_neighborhoods
assert spatial_dimension == batch_size * mini_batch_size

#################################################
# Create Features and Targets
#################################################

# Neighborhood Features
neighborhood_ids = torch.arange(num_neighborhoods)
logger.info(f"neighborhood_ids shape {neighborhood_ids.shape}")

# Time Features
time_features = torch.zeros(spatial_dimension, 1, time_feature_dim)  # [# years, # weeks]
for nid in range(num_neighborhoods):
    # Filter rows for the current neighborhood
    neighborhood_data = weekly_events_df[weekly_events_df["Neighbourhood Number"] == neighbourhood_mappings[nid]]

    if not neighborhood_data.empty:
        # Use "year" and "week" as features
        year_mean = neighborhood_data["year"].mean()
        week_mean = neighborhood_data["week"].mean()

        # Combine them into time features (e.g., [mean year, mean week])
        time_features[nid, 0, :] = torch.tensor([year_mean, week_mean])
    else:
        # Default to zero if no data for the neighborhood
        time_features[nid, 0, :] = torch.zeros(time_feature_dim)

logger.info(f"time_features shape: {time_features.shape}")

# Building Features
building_type_ids = torch.arange(num_building_types).repeat(spatial_dimension, 1)
logger.info(f"building_type_ids shape {building_type_ids.shape}")

# Building Counts
building_counts_np = neighbourhood_feature_df[build_type_list].fillna(0).to_numpy(dtype=np.int32)
building_counts = torch.from_numpy(building_counts_np).unsqueeze(1)  # Temporal dimension = 1
logger.info(f"building_counts shape {building_counts.shape}")

# Demographic Features
population = torch.from_numpy(population_np).float()
logger.info(f"population shape {population.shape}")

income = torch.from_numpy(income_np).float()
logger.info(f"income shape {income.shape}")

# Event Features
event_type_ids = torch.randint(0, num_event_types, (spatial_dimension, 1))  # Temporal dimension = 1
logger.info(f"event_type_ids shape {event_type_ids.shape}")

equipment_ids = torch.randint(0, num_equipment_types, (spatial_dimension, 1))  # Temporal dimension = 1
logger.info(f"equipment_ids shape {equipment_ids.shape}")

# Target Values
agg_data = (
    weekly_events_df.groupby(["year", "week", "Neighbourhood Number"])
    .agg(
        {
            "Rc_description": " ".join,  # Concatenate descriptions (or other aggregation if needed)
            "event_count": "sum",  # Sum event counts
        }
    )
    .reset_index()
)

unique_week_year_combinations = agg_data[["year", "week"]].drop_duplicates()
targets = torch.zeros((num_neighborhoods,), dtype=torch.float32)

for i, neighbourhood in enumerate(neighbourhood_feature_df["Neighbourhood_Number"]):
    neighbourhood_data = agg_data[agg_data["Neighbourhood Number"] == neighbourhood]
    # Sum of event counts for each (year, week) combination
    for k, (year, week) in enumerate(unique_week_year_combinations.itertuples(index=False)):
        event_count = neighbourhood_data[(neighbourhood_data["year"] == year) & (neighbourhood_data["week"] == week)][
            "event_count"
        ].sum()
        targets[i] = event_count  # Aggregated for the entire week
logger.info(f"targets shape {targets.shape}")

#################################################
# Transformer Model and Hyperparameters
#################################################

# Embedding Layer
embedding_module = CombinedEmbedding(
    logger=logger,
    node2vec_emb_layer=generate_node2vec_embeddings(
        logger=logger, neighbourhood_info_df=neighbourhood_df, node2vec_dim=node2vec_dim
    ),
    time2vec_embed_dim=time2vec_embed_dim,
    time_feature_dim=time_feature_dim,
    num_building_types=num_building_types,
    building_type_embed_dim=building_type_embed_dim,
    population_embed_dim=population_embed_dim,
    num_event_types=num_event_types,
    event_type_embed_dim=event_type_embed_dim,
    num_equipment_types=num_equipment_types,
    equipment_embed_dim=equipment_embed_dim,
    target_embed_dim=target_embed_dim,
)

# Splitting dataset
train_indices, val_indices = train_test_split(np.arange(num_neighborhoods), test_size=0.2, random_state=42)

train_neighborhood_ids = neighborhood_ids[train_indices]
train_time_features = time_features[train_indices]
train_building_type_ids = building_type_ids[train_indices]
train_building_counts = building_counts[train_indices]
train_population = population[train_indices]
train_event_type_ids = event_type_ids[train_indices]
train_equipment_ids = equipment_ids[train_indices]
train_targets = targets[train_indices]

val_neighborhood_ids = neighborhood_ids[val_indices]
val_time_features = time_features[val_indices]
val_building_type_ids = building_type_ids[val_indices]
val_building_counts = building_counts[val_indices]
val_population = population[val_indices]
val_event_type_ids = event_type_ids[val_indices]
val_equipment_ids = equipment_ids[val_indices]
val_targets = targets[val_indices]

# Dataset and DataLoader
val_dataset = NeighborhoodDataset(
    logger=logger,
    neighborhood_ids=val_neighborhood_ids,
    time_features=val_time_features,
    building_type_ids=val_building_type_ids,
    building_counts=val_building_counts,
    population=val_population,
    event_type_ids=val_event_type_ids,
    equipment_ids=val_equipment_ids,
    targets=val_targets,
)
train_dataset = NeighborhoodDataset(
    logger=logger,
    neighborhood_ids=train_neighborhood_ids,
    time_features=train_time_features,
    building_type_ids=train_building_type_ids,
    building_counts=train_building_counts,
    population=train_population,
    event_type_ids=train_event_type_ids,
    equipment_ids=train_equipment_ids,
    targets=train_targets,
)

val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)

# Transformer Model
transformer = ST_TEM(logger=logger, embedding_module=embedding_module, embed_dim=64, num_heads=4, num_layers=2)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
criterion = nn.PoissonNLLLoss()
num_epochs = 500

#################################################
# Train and Validate Model
#################################################
transformer.train_model(
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=num_epochs,
    dataloader=train_dataloader,
    save_model=False,
)
all_predictions, all_targets = transformer.validate_model(dataloader=val_dataloader)
transformer.plot_result(all_predictions, all_targets)
