from shapely.geometry import Point


def find_neighbourhood(latitude, longitude, neighbourhood_info_df):
    """Finds the Neighbourhood Number for a given latitude and longitude.

    Args:
      latitude: The latitude.
      longitude: The longitude.
      neighbourhood_info_df: The DataFrame containing neighbourhood data.

    Returns:
      The Neighbourhood Number, or None if not found.
    """
    point = Point(longitude, latitude)
    for _, row in neighbourhood_info_df.iterrows():
        multipolygon = row["MultiPolygon_obj"]
        if multipolygon.contains(point):
            return row["Neighbourhood Number"]
    return None
