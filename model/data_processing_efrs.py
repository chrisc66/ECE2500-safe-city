import os
import sys
import logging
import pandas as pd

from shapely.wkt import loads
from pandarallel import pandarallel
from utils import find_neighbourhood

DATASET_PATH = "../dataset/EdmontonFireRescueServicesData"
UNIT_TRIP_PATH = os.path.join(DATASET_PATH, "EFRS_Unit_Trip_Summary.csv")
EVENT_TRIP_PATH = os.path.join(DATASET_PATH, "EFRS_Event_Trip_Summary.csv")
UNIT_HISTORY_2023_PATH = os.path.join(DATASET_PATH, "UN_HI_2023.csv")
NEIGHBOURHOOD_PATH = os.path.join(DATASET_PATH, "City_of_Edmonton_-_Neighbourhoods_20241022.csv")
FIRE_STATION_PATH = os.path.join(DATASET_PATH, "Fire_Stations_20241027.csv")
NEIGHBOURHOOD_FEATURES_PATH = os.path.join(DATASET_PATH, "neighbourhood_static_data_with_five_years_events.csv")

unit_trip_df = pd.read_csv(UNIT_TRIP_PATH)
event_trip_df = pd.read_csv(EVENT_TRIP_PATH)
unit_history_2023_df = pd.read_csv(UNIT_HISTORY_2023_PATH)
nbhd_df = pd.read_csv(NEIGHBOURHOOD_PATH)
station_df = pd.read_csv(FIRE_STATION_PATH)
nbhd_feat_df = pd.read_csv(NEIGHBOURHOOD_FEATURES_PATH)

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
pandarallel.initialize(nb_workers=8, progress_bar=True)

# Column definitions
# fmt: off
build_type_list = [
    'Apartment_Condo_1_to_4_stories', 'Apartment_Condo_5_or_more_stories', 'Duplex_Fourplex',
    'Hotel_Motel', 'Institution_Collective_Residence', 'Manufactured_Mobile_Home',
    'RV_Tent_Other', 'Row_House', 'Single_Detached_House'
]

# fmt: off
nbhd_selected_columns = [
    'Neighbourhood_Number', 'Neighbourhood_Name', 'Ward', 'Population',
    # buidling type
    'Apartment_Condo_1_to_4_stories', 'Apartment_Condo_5_or_more_stories', 'Duplex_Fourplex',
    'Hotel_Motel', 'Institution_Collective_Residence', 'Manufactured_Mobile_Home',
    'RV_Tent_Other', 'Row_House', 'Single_Detached_House',
    # duration
    'five_Years_or_More', 'three_Years_to_Less_than_five_Years',
    'one_Year_to_Less_than_three_Years', 'Less_than_one_Year',
    # income
    'Low_Income', 'Low_medium_Income', 'Medium_Income', 'High_Income',
    # education
    'No_Certificate_Diploma_or_Degree', 'High_School_Trades_or_Apprenticeship_Certificate',
    'College_or_University_Certificate_or_Diploma',
    'University_Bachelor_and_Medical_Degree', 'Master_and_Doctorate_Degree',
    # age
    'Children', 'Youth', 'Adults', 'Seniors',
    # zoning
    'No_traffic_lights', 'No_bus_stops',
    'Food', 'Education', 'Healthcare', 'Entertainment',
    'Public_Service', 'commercial', 'retail',
    # event type
    'ALARMS', 'CITIZEN_ASSIST', 'COMMUNITY_EVENT', 'FIRE',
    'HAZARDOUS_MATERIALS', 'MEDICAL', 'MOTOR_VEHICLE_INCIDENT',
    'OUTSIDE_FIRE', 'RESCUE', 'TRAINING_MAINTENANCE', 'VEHICLE_FIRE'
]

event_selected_columns = [
    'Eid', 'Rc', 'Rc_description',
    # time
    'day_of_year', 'day_of_week', 'hour', 'week', 'year',
    # neighbourhood
    'Neighbourhood Number'
]

filter_event_list = [
    "Emergency Life Threatening - Immediate",
    "Emergency Life Threatening",
    "Emergency Non-Life Threatening",
    "Potential Life Threatening",
    "Non-Emergency",
]
# fmt: on

event_type_list = event_trip_df["Rc_description"].unique()
unit_type_list = unit_trip_df["unityp"].unique()

num_neighborhoods = len(nbhd_df)
num_building_types = len(build_type_list)
num_event_types = len(event_type_list)
num_equipment_types = len(unit_type_list)

# Processing neighbourhood dataframe
nbhd_df["MultiPolygon_obj"] = nbhd_df["Geometry Multipolygon"].parallel_apply(loads)

nbhd_feat_full_df = pd.merge(
    nbhd_feat_df, nbhd_df, left_on="Neighbourhood_Number", right_on="Neighbourhood Number", how="inner"
)
nbhd_feat_full_df["Population"] = nbhd_feat_full_df["Area_Sq_Km"] * nbhd_feat_full_df["Population_per_Sq_km"]

event_trip_df["Neighbourhood Number"] = event_trip_df.parallel_apply(
    lambda row: find_neighbourhood(row["Latitude"], row["Longitude"], nbhd_df), axis=1
)

# Processing event dataframe
event_trip_df["Sd_date_dt"] = pd.to_datetime(event_trip_df["Sd_date"])
event_trip_df["date"] = event_trip_df["Sd_date_dt"].dt.date
event_trip_df["day_of_year"] = event_trip_df["Sd_date_dt"].dt.dayofyear
event_trip_df["day_of_week"] = event_trip_df["Sd_date_dt"].dt.dayofweek
event_trip_df["hour"] = event_trip_df["Sd_date_dt"].dt.hour
event_trip_df["week"] = event_trip_df["Sd_date_dt"].dt.strftime("%V")  # week number
event_trip_df["year"] = event_trip_df["Sd_date_dt"].dt.year

# Creating slim versions
nbhd_df_slim = nbhd_feat_full_df[nbhd_selected_columns].apply(pd.to_numeric, errors="ignore")

event_trip_df_slim = event_trip_df[event_selected_columns]
event_trip_df_slim = event_trip_df_slim.apply(pd.to_numeric, errors="ignore")

# Creating weekly event aggregation
weekly_event_df = (
    event_trip_df_slim.groupby(["year", "week", "Neighbourhood Number", "Rc_description", "Rc"])
    .size()
    .reset_index(name="event_count")
)

weekly_events_df_filter = weekly_event_df[weekly_event_df["Rc_description"].isin([i for i in filter_event_list])]

# Saving processed dataframes
nbhd_df_save = nbhd_df_slim.replace(to_replace=r"\n", value=" ", regex=True)
nbhd_df_save.to_csv(os.path.join(DATASET_PATH, "neighbourhood_features.csv"))

event_df_save = event_trip_df_slim.replace(to_replace=r"\n", value=" ", regex=True)
event_df_save.to_csv(os.path.join(DATASET_PATH, "event_trip_features.csv"))

weekly_event_df_save = weekly_event_df.replace(to_replace=r"\n", value=" ", regex=True)
weekly_event_df_save.to_csv(os.path.join(DATASET_PATH, "weekly_events.csv"))

weekly_event_df_filter_save = weekly_events_df_filter.replace(to_replace=r"\n", value=" ", regex=True)
weekly_event_df_filter_save.to_csv(os.path.join(DATASET_PATH, "weekly_events_filter.csv"))
