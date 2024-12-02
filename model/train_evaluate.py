import os
import sys
import logging
import time
import random

logger = logging.getLogger(__name__)
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

TENSOR_PATH = "../tensors"
if not os.path.exists(TENSOR_PATH):
    os.mkdir(TENSOR_PATH)
NBHD_IDS_TENSOR = os.path.join(TENSOR_PATH, "neighborhood_ids.pt")
TIME_FEATURES_TENSOR = os.path.join(TENSOR_PATH, "time_features.pt")
BLD_TYPE_IDS_TENSOR = os.path.join(TENSOR_PATH, "build_type_ids.pt")
BLD_COUNTS_TENSOR = os.path.join(TENSOR_PATH, "building_counts.pt")
POPULATION_TENSOR = os.path.join(TENSOR_PATH, "population.pt")
INCOME_LEVEL_TENSOR = os.path.join(TENSOR_PATH, "income_level.pt")
EVENT_TYPE_IDS_TENSOR = os.path.join(TENSOR_PATH, "event_type_ids.pt")
EQUIPMENT_IDS_TENSOR = os.path.join(TENSOR_PATH, "equipment_ids.pt")
TARGETS_TENSOR = os.path.join(TENSOR_PATH, "targets.pt")

FIGURE_PATH = "../figures"
if not os.path.exists(FIGURE_PATH):
    os.mkdir(FIGURE_PATH)

logger.info(f"Start loading dataset")
start = time.time()
unit_trip_df = pd.read_csv(UNIT_TRIP_PATH)
weekly_events_df = pd.read_csv(WEEKLY_EVENTS_FILTER_PATH)
neighbourhood_df = pd.read_csv(NEIGHBOURHOOD_PATH)
neighbourhood_feature_df = pd.read_csv(NEIGHBOURHOOD_FEATURES_PATH)
end = time.time()
logger.info(f"Finish loading dataset, time elapsed {(end-start):.2f}s")

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
neighborhood_number_list = neighbourhood_feature_df["Neighbourhood_Number"].unique()

weekly_nbhd_events = (
    weekly_events_df.groupby(["year", "week", "Neighbourhood Number"])
    .agg(
        {
            "Rc_description": " ".join,  # Concatenate descriptions (or other aggregation if needed)
            "event_count": "sum",  # Sum event counts
        }
    )
    .reset_index()
)
unique_year_week = weekly_nbhd_events[["year", "week"]].drop_duplicates().to_numpy()

spatial_dimension = len(neighborhood_number_list)  # spatial_dimension = num_neighbourhood
temporal_dimension = len(unique_year_week)
num_income = len(income_list)
num_building_types = len(build_type_list)
num_event_types = len(event_type_list)
num_equipment_types = len(unit_type_list)

# Embedding Parameter and Module
node2vec_dim = 32
time2vec_embed_dim = 64
time_feature_dim = 2  # week, year
building_type_embed_dim = 9
population_embed_dim = 8
event_type_embed_dim = 16
equipment_embed_dim = 16
target_embed_dim = 64

mini_batch_size = 29
batch_size = 13
assert spatial_dimension == batch_size * mini_batch_size

#################################################
# Create Features and Targets
#################################################

logger.info(f"Start creating features and targets")
start = time.time()

# Neighborhood Features
if os.path.isfile(NBHD_IDS_TENSOR):
    neighborhood_ids = torch.load(NBHD_IDS_TENSOR, weights_only=True)
else:
    neighborhood_ids = torch.zeros(temporal_dimension, spatial_dimension, 1)
    for t in range(temporal_dimension):
        neighborhood_ids[t, :, 0] = torch.arange(spatial_dimension).to(torch.int64)
    torch.save(neighborhood_ids, NBHD_IDS_TENSOR)
logger.info(f"neighborhood_ids shape {neighborhood_ids.shape}")

# Time Features
if os.path.isfile(TIME_FEATURES_TENSOR):
    time_features = torch.load(TIME_FEATURES_TENSOR, weights_only=True)
else:
    time_features = torch.zeros(temporal_dimension, spatial_dimension, time_feature_dim)  # [# years, # weeks]
    for s in range(spatial_dimension):
        time_features[:, s, :] = torch.from_numpy(unique_year_week).to(torch.int64)
    torch.save(time_features, TIME_FEATURES_TENSOR)
logger.info(f"time_features shape: {time_features.shape}")

# Building Features
if os.path.isfile(BLD_TYPE_IDS_TENSOR):
    building_type_ids = torch.load(BLD_TYPE_IDS_TENSOR, weights_only=True)
else:
    building_type_ids = torch.zeros(temporal_dimension, spatial_dimension, num_building_types)
    for t in range(temporal_dimension):
        for s in range(spatial_dimension):
            building_type_ids[t, s, :] = torch.arange(num_building_types).to(torch.int64)
    torch.save(building_type_ids, BLD_TYPE_IDS_TENSOR)
logger.info(f"building_type_ids shape {building_type_ids.shape}")

# Building Counts
if os.path.isfile(BLD_COUNTS_TENSOR):
    building_counts = torch.load(BLD_COUNTS_TENSOR, weights_only=True)
else:
    building_counts_np = neighbourhood_feature_df[build_type_list].fillna(0).astype(int).to_numpy()
    building_counts = torch.zeros(temporal_dimension, spatial_dimension, num_building_types)
    for t in range(temporal_dimension):
        building_counts[t, :, :] = torch.from_numpy(building_counts_np).to(torch.int64)
    torch.save(building_counts, BLD_COUNTS_TENSOR)
logger.info(f"building_counts shape {building_counts.shape}")

# Demographic Features
if os.path.isfile(POPULATION_TENSOR):
    population = torch.load(POPULATION_TENSOR, weights_only=True)
else:
    population_np = neighbourhood_feature_df["Population"].fillna(0).astype(int).to_numpy()
    population = torch.zeros(temporal_dimension, spatial_dimension, 1)
    for t in range(temporal_dimension):
        population[t, :, 0] = torch.from_numpy(population_np).to(torch.int64)
    torch.save(population, POPULATION_TENSOR)
logger.info(f"population shape {population.shape}")

if os.path.isfile(INCOME_LEVEL_TENSOR):
    income_level = torch.load(INCOME_LEVEL_TENSOR, weights_only=True)
else:
    income_level_np = neighbourhood_feature_df[income_list].fillna(0).astype(int).to_numpy()
    income_level = torch.zeros(temporal_dimension, spatial_dimension, num_income)
    for t in range(temporal_dimension):
        income_level[t, :, :] = torch.from_numpy(income_level_np).to(torch.int64)
    torch.save(income_level, INCOME_LEVEL_TENSOR)
logger.info(f"income_level shape {income_level.shape}")

# Event Features
if os.path.isfile(EVENT_TYPE_IDS_TENSOR):
    event_type_ids = torch.load(EVENT_TYPE_IDS_TENSOR, weights_only=True)
else:
    event_type_ids = torch.zeros(temporal_dimension, spatial_dimension, 1).to(torch.int64)
    torch.save(event_type_ids, EVENT_TYPE_IDS_TENSOR)
logger.info(f"event_type_ids shape {event_type_ids.shape}")

if os.path.isfile(EQUIPMENT_IDS_TENSOR):
    equipment_ids = torch.load(EQUIPMENT_IDS_TENSOR, weights_only=True)
else:
    equipment_ids = torch.zeros(temporal_dimension, spatial_dimension, 1).to(torch.int64)
    torch.save(event_type_ids, EQUIPMENT_IDS_TENSOR)
logger.info(f"equipment_ids shape {equipment_ids.shape}")

# Target Values
if os.path.isfile(TARGETS_TENSOR):
    targets = torch.load(TARGETS_TENSOR, weights_only=True)
else:
    targets = torch.zeros((temporal_dimension, spatial_dimension, 1), dtype=torch.float32)
    for t in range(temporal_dimension):
        year = unique_year_week[t][0]
        week = unique_year_week[t][1]
        for s in range(spatial_dimension):
            nid = neighborhood_number_list[s]
            # TODO: add event_type (Rc_description) dimension and remove sum
            event_cnt = weekly_nbhd_events[
                (weekly_nbhd_events["year"] == year)
                & (weekly_nbhd_events["week"] == week)
                & (weekly_nbhd_events["Neighbourhood Number"] == nid)
            ]["event_count"].sum()
            building_type_ids[t, s, 0] = torch.tensor(event_cnt, dtype=torch.int32)
    torch.save(targets, TARGETS_TENSOR)
logger.info(f"targets shape {targets.shape}")

end = time.time()
logger.info(f"Finish creating features and targets, time elapsed {(end-start):.2f}s")

#################################################
# Transformer Model and Hyperparameters
#################################################

# Embedding Layer
embedding_module = CombinedEmbedding(
    node2vec_emb_layer=generate_node2vec_embeddings(neighbourhood_info_df=neighbourhood_df, node2vec_dim=node2vec_dim),
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
train_indices, val_indices = train_test_split(
    np.arange(temporal_dimension), test_size=0.2, random_state=random.randint(0, 100)
)

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
    neighborhood_ids=train_neighborhood_ids,
    time_features=train_time_features,
    building_type_ids=train_building_type_ids,
    building_counts=train_building_counts,
    population=train_population,
    event_type_ids=train_event_type_ids,
    equipment_ids=train_equipment_ids,
    targets=train_targets,
)

val_dataloader = DataLoader(val_dataset, batch_size=mini_batch_size, shuffle=False)
train_dataloader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=False)

# Transformer Model
transformer = ST_TEM(embedding_module=embedding_module, embed_dim=64, num_heads=4, num_layers=2)
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
transformer.plot_result(predictions=all_predictions, targets=all_targets, figure_path=FIGURE_PATH)
