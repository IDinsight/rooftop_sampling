import folium
import numpy as np
import s2sphere
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely import Point, Polygon
import osmnx as ox
from dotenv import load_dotenv
import os
import requests
from shapely.ops import nearest_points
from geopy.distance import geodesic

load_dotenv()
api_key = os.getenv("GOOGLE_MAPS_API_KEY")


def count_neighbors_in_radius(gdf, radius=100):
    """
    Counts the number of points within a given radius (including the point itself) for each point in a GeoDataFrame.
    Note that we could speed up this code by using scipy.spatial.cKDTree and performing the operation by groups but keeping the code as is for simplicity.

    Parameters:
    - gdf (GeoDataFrame): The input GeoDataFrame containing points.
    - radius (float): The distance threshold in meters. Default is 100 meters.

    Returns:
    - (GeoSeries): Counts of neighbors within the given radius for each point.
    """
    # convert gdf to EPSG 32633
    gdf = gdf.to_crs(epsg=32633)
    joined = gpd.sjoin(
        gdf, gdf.set_geometry(gdf.buffer(radius)), how="inner", predicate="intersects"
    )
    joined["left_index"] = joined.index
    neighbor_counts = joined.groupby("left_index").size()
    gdf["neighbor_count"] = gdf.index.map(neighbor_counts).fillna(0)
    return gdf["neighbor_count"]


def get_s2_cell_id(point: Point, level: int = 6) -> int:
    """
    Returns the S2 cell ID for a given point.

    Args:
        point (Point): The point for which to calculate the S2 cell ID.
        level (int, optional): The level of the S2 cell. Defaults to 6 which is the S2 level for India data in the Vida dataset. The level for the Philippines is 4.

    Returns:
        int: The S2 cell ID.
    """
    lat = point.y
    lon = point.x
    latlng = s2sphere.LatLng.from_degrees(lat, lon)
    cell_id = s2sphere.CellId.from_lat_lng(latlng).parent(level)
    return cell_id.id()


def gen_rooftop_map(poly: Polygon, gdf: GeoDataFrame) -> folium.Map:
    """
    Generates a folium map showing rooftops and boundary.

    Args:
        poly (Polygon): Boundary of the PSU
        gdf (GeoDataFrame): The GeoDataFrame containing rooftop polygons.

    Returns:
        folium.Map: The generated folium map.
    """
    m = folium.Map(
        location=[poly.centroid.y, poly.centroid.x],
        zoom_start=20,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
    )

    # add rooftop polygons to the map
    for idx, row in gdf.iterrows():
        folium.GeoJson(
            row.geometry,
            style_function=lambda x: {
                "color": "red",
                "fillColor": "red",
                "weight": 1,
                "fillOpacity": 0.1,
            },
        ).add_to(m)

    # add the polygon to the map
    folium.GeoJson(
        poly,
        style_function=lambda x: {
            "color": "blue",
            "fillColor": "blue",
            "weight": 1,
            "fillOpacity": 0.1,
        },
    ).add_to(m)
    return m


# Function to find the nearest point on a road
def get_nearest_point_on_road_free(point: Point, buffer_distance = 200):
    """
    Finds the nearest point to the given point which lies on a road using OSMs open source road data. Takes a really long time to run. 

    Parameters:
    - point (Point): The point for which the nearest point on the road network needs to be found.
    - buffer_distance (float): The distance within which to extract the walkable street network.

    Returns:
    - nearest_point (Point): The nearest point on the road network to the given point.
    """
    # Extract the walkable street network for this point within a certain distance
    try:
            G = ox.graph_from_point((point.y, point.x), dist=buffer_distance, network_type='all')
    except ValueError as e:
        return None
        
    # Check if the graph has edges
    if not any(G.edges):
        return None  # Return None if no edges are found in the graph
    
    # Convert the graph to a GeoDataFrame of edges (roads)
    edges = ox.graph_to_gdfs(G, nodes=False)

    nearest_geom = edges.geometry.unary_union
    nearest_point = nearest_points(nearest_geom, point)[0]
    return nearest_point


def get_nearest_point_on_road(point: Point):
    """
    Retrieves the nearest point on the road for a given point using the Google Roads API.

    Args:
        point (Point): The point for which to find the nearest point on the road.

    Returns:
        Point: The nearest point on the road, or None if no point is found.

    """
    url = f"https://roads.googleapis.com/v1/snapToRoads?path={point.y},{point.x}&key={api_key}"
    response = requests.get(url)
    snapped_point = response.json().get('snappedPoints', [{}])[0].get('location')
    return Point(snapped_point['longitude'], snapped_point['latitude']) if snapped_point else None



def get_nearest_point_on_road_batch(points_series):
    """
    WARNING: THIS CODE DOESN'T WORK!!! DO NOT USE IT.
    Get the nearest points on roads for a batch of points using Google Maps Snap to Roads API.
    
    Parameters:
    points_series (GeoSeries): A GeoSeries containing points.
    
    Returns:
    GeoSeries: A GeoSeries containing the nearest points on roads.
    """
    def call_snap_to_roads_api(points_batch):
        url = "https://roads.googleapis.com/v1/snapToRoads"
        params = {
            'path': '|'.join([f"{point.y},{point.x}" for point in points_batch]),
            'interpolate': 'false',
            'key': api_key
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        snapped_points = response.json().get('snappedPoints', [])
        return snapped_points

    all_snapped_points = []
    batch_size = 100
    num_batches = (len(points_series) + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch_points = points_series.iloc[i * batch_size:(i + 1) * batch_size]
        snapped_points = call_snap_to_roads_api(batch_points)
        all_snapped_points.extend(snapped_points)

    snapped_points_coords = [(point['location']['latitude'], point['location']['longitude']) for point in all_snapped_points]
    snapped_points_series = gpd.GeoSeries(gpd.points_from_xy([coord[1] for coord in snapped_points_coords], [coord[0] for coord in snapped_points_coords]))

    return snapped_points_series


def dist_in_meters(point1, point2):
    # Check if either of the points is None
    if point1 is None or point2 is None:
        return None
    
    # Extract latitude and longitude from the Point objects
    lat1, lon1 = point1.y, point1.x
    lat2, lon2 = point2.y, point2.x

    # Calculate the distance in meters
    return geodesic((lat1, lon1), (lat2, lon2)).meters