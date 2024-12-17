import os
import sys
import logging
import pandas as pd

from shapely.wkt import loads
from pandarallel import pandarallel
from utils import find_neighbourhood

# Initialize logging and parallel processing
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
pandarallel.initialize(nb_workers=8, progress_bar=True)

# Paths
DATASET_PATH = "../dataset/EdmontonFireRescueServicesData"
UNIT_TRIP_PATH = os.path.join(DATASET_PATH, "EFRS_Unit_Trip_Summary.csv")
EVENT_TRIP_PATH = os.path.join(DATASET_PATH, "EFRS_Event_Trip_Summary.csv")
UNIT_HISTORY_2023_PATH = os.path.join(DATASET_PATH, "UN_HI_2023.csv")
NEIGHBOURHOOD_PATH = os.path.join(DATASET_PATH, "City_of_Edmonton_-_Neighbourhoods_20241022.csv")
FIRE_STATION_PATH = os.path.join(DATASET_PATH, "Fire_Stations_20241027.csv")
NEIGHBOURHOOD_FEATURES_PATH = os.path.join(DATASET_PATH, "neighbourhood_static_data_with_five_years_events.csv")

# Load data
unit_trip_df = pd.read_csv(UNIT_TRIP_PATH)
event_trip_df = pd.read_csv(EVENT_TRIP_PATH)
unit_history_2023_df = pd.read_csv(UNIT_HISTORY_2023_PATH)
nbhd_df = pd.read_csv(NEIGHBOURHOOD_PATH)
nbhd_feat_df = pd.read_csv(NEIGHBOURHOOD_FEATURES_PATH)

# Column definitions
build_type_list = [
    'Apartment_Condo_1_to_4_stories', 'Apartment_Condo_5_or_more_stories', 'Duplex_Fourplex',
    'Hotel_Motel', 'Institution_Collective_Residence', 'Manufactured_Mobile_Home',
    'RV_Tent_Other', 'Row_House', 'Single_Detached_House'
]

event_selected_columns = [
    'Eid', 'Rc', 'Rc_description', 'day_of_year', 'day_of_week', 'hour', 'week', 'year', 'Neighbourhood Number'
]

filter_event_list = [
    "Emergency Life Threatening - Immediate",
    "Emergency Life Threatening",
    "Emergency Non-Life Threatening",
    "Potential Life Threatening",
    "Non-Emergency",
]

# Merge static neighborhood features with geometries
nbhd_df["MultiPolygon_obj"] = nbhd_df["Geometry Multipolygon"].parallel_apply(loads)
nbhd_feat_full_df = pd.merge(
    nbhd_feat_df, nbhd_df, left_on="Neighbourhood_Number", right_on="Neighbourhood Number", how="inner"
)
nbhd_feat_full_df["Population"] = nbhd_feat_full_df["Area_Sq_Km"] * nbhd_feat_full_df["Population_per_Sq_km"]

# Match events to neighborhoods
event_trip_df["Neighbourhood Number"] = event_trip_df.parallel_apply(
    lambda row: find_neighbourhood(row["Latitude"], row["Longitude"], nbhd_df), axis=1
)

# Add temporal features
event_trip_df["Sd_date_dt"] = pd.to_datetime(event_trip_df["Sd_date"])
event_trip_df["day_of_year"] = event_trip_df["Sd_date_dt"].dt.dayofyear
event_trip_df["day_of_week"] = event_trip_df["Sd_date_dt"].dt.dayofweek
event_trip_df["hour"] = event_trip_df["Sd_date_dt"].dt.hour
event_trip_df["week"] = event_trip_df["Sd_date_dt"].dt.strftime("%V")
event_trip_df["year"] = event_trip_df["Sd_date_dt"].dt.year

# Aggregate weekly events
weekly_event_df = (
    event_trip_df.groupby(["year", "week", "Neighbourhood Number", "Rc_description", "Rc"])
    .size()
    .reset_index(name="event_count")
)
weekly_events_df_filter = weekly_event_df[weekly_event_df["Rc_description"].isin(filter_event_list)]

# Prepare flattened spatiotemporal data
def prepare_spatiotemporal_data(event_df, nbhd_df):
    spatiotemporal_data = []
    for _, event_row in event_df.iterrows():
        for _, nbhd_row in nbhd_df.iterrows():
            spatiotemporal_data.append({
                "hour": event_row["hour"],
                "day_of_week": event_row["day_of_week"],
                "week": event_row["week"],
                "year": event_row["year"],
                "spatial_id": nbhd_row["Neighbourhood_Number"],
                "population": nbhd_row["Population"],
                "event_count": event_row["event_count"],
                "event_type": event_row["Rc"],
                "building_count": nbhd_row["Single_Detached_House"],  # Replace with relevant building counts
                "building_type_id": 0,  # Placeholder: Map building types to IDs if available
                "position": event_row.name * len(nbhd_df) + nbhd_row.name
            })
    return pd.DataFrame(spatiotemporal_data)


# Flattened data
logger.info("Preparing spatiotemporal data...")
spatiotemporal_df = prepare_spatiotemporal_data(weekly_events_df_filter, nbhd_feat_full_df)

# Save processed data
spatiotemporal_df.to_csv(os.path.join(DATASET_PATH, "spatiotemporal_dataset.csv"), index=False)

logger.info("Spatiotemporal dataset saved successfully!")
