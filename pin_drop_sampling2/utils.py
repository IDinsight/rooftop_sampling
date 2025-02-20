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
from rpy2 import robjects


load_dotenv()
api_key = os.getenv("GOOGLE_MAPS_API_KEY")


def count_neighbors_in_radius(gdf, radius=100) -> np.ndarray:
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
    return gdf["neighbor_count"].values


def randomly_sample_rooftops(rooftop_neighbour_count: list | np.ndarray, 
                             n_samples: int, 
                             random_seed: int = 42) -> np.ndarray:
    """
    Randomly sample rooftops proportional to number of neighbours.

    Parameters:
    - rooftop_neighbour_count (list | np.ndarray): Number of neighbours for each rooftop.
    - n_samples (int): Number of samples to draw.
    - random_seed (int): Random seed.

    Returns:
    - np.ndarray: Sampled rooftop indices.
    """
    # Set random seed
    np.random.seed(random_seed)
    return np.random.choice(
        np.arange(len(rooftop_neighbour_count)), 
        size=n_samples, 
        p=rooftop_neighbour_count / np.sum(rooftop_neighbour_count), 
        replace=False)


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

    Parameters:
    - poly (Polygon): Boundary of the PSU
    - gdf (GeoDataFrame): The GeoDataFrame containing rooftop polygons.

    Returns:
        folium.Map: The generated folium map.
    """
    m = folium.Map(
        location=[poly.centroid.y, poly.centroid.x],
        zoom_start=12,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        alpha=0.01,
    )

    # add rooftop polygons to the map
    for idx, row in gdf.iterrows():
        folium.GeoJson(
            row.geometry,
            style_function=lambda x: {
            "color": "red",
            "fillColor": "red",
            "weight": 1,
            "fillOpacity": 1.,
            },
            marker=folium.CircleMarker(radius=.01) 
        ).add_to(m)

    # add the polygon to the map
    folium.GeoJson(
        poly,
        style_function=lambda x: {
            "color": "blue",
            "fillColor": "None",
            "weight": 1,
            "fillOpacity": 0.2,
        },
    ).add_to(m)
    return m


# Function to find the nearest point on a road
def get_nearest_point_on_road_free(point: Point, buffer_distance: float = 200) -> Point:
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


def get_nearest_point_on_road(point: Point) -> Point:
    """
    Retrieves the nearest point on the road for a given point using the Google Roads API.

    Parameters:
        - point (Point): The point for which to find the nearest point on the road.

    Returns:
        Point: The nearest point on the road, or None if no point is found.
    """
    url = f"https://roads.googleapis.com/v1/snapToRoads?path={point.y},{point.x}&key={api_key}"
    response = requests.get(url)
    snapped_point = response.json().get('snappedPoints', [{}])[0].get('location')
    return Point(snapped_point['longitude'], snapped_point['latitude']) if snapped_point else None



def get_nearest_point_on_road_batch(points_series: gpd.GeoSeries) -> gpd.GeoSeries:
    """
    WARNING: THIS CODE DOESN'T WORK!!! DO NOT USE IT.
    Get the nearest points on roads for a batch of points using Google Maps Snap to Roads API.
    
    Parameters:
    - points_series (GeoSeries): A GeoSeries containing points.
    
    Returns:
    - GeoSeries: A GeoSeries containing the nearest points on roads.
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


def dist_in_meters(point1: Point, point2: Point) -> float:
    """
    Calculate the distance between two points in meters.

    Parameters:
    - point1 (Point): The first point.
    - point2 (Point): The second point.

    Returns:
    - float: The distance between the two points in meters.
    """
    # Check if either of the points is None
    if point1 is None or point2 is None:
        return None
    
    # Extract latitude and longitude from the Point objects
    lat1, lon1 = point1.y, point1.x
    lat2, lon2 = point2.y, point2.x

    # Calculate the distance in meters
    return geodesic((lat1, lon1), (lat2, lon2)).meters


def sample_locations_with_pps(
    all_psus: list | np.ndarray,
    sizes: list | np.ndarray,
    n_samples: int,
    with_replacement: bool = False,
    random_seed: int = 42,    
) -> tuple:
    """
    Sample PSUs with PPS.

    Parameters:
    - all_psus: List of PSUs from which to sample
    - sizes: Sizes corresponding to each PSU
    - n_samples: Number of samples
    - with_replacement: Whether to sample with replacement
    - random_seed: Random seed

    Returns:
    - tuple: A tuple containing the sampled PSUs, sampled PSUs index, and the probabilities.
    """
    assert len(all_psus) == len(sizes), "Length of all_psus and sizes should be the same"
    # Get sample sizes for locations
    r_sizes = robjects.FloatVector(sizes)
    r_n_samp = robjects.IntVector([n_samples])
    r_seed = robjects.IntVector([random_seed])

    # Sample locations with PPS using R packages
    rcode = ""
    if with_replacement:
        rcode = str(
            """
            function(size, n_samp, seed) {
                library(TeachingSampling)
                set.seed(seed)
                samples <- S.PPS(m=n_samp, x=size)
                return(samples)
            }
            """
        )
    else:
        rcode = str(
            """
            function(size, n_samp, seed) {
                library(TeachingSampling)
                set.seed(seed)
                samples <- S.piPS(n=n_samp, x=size)
                return(samples)
            }
            """
        )
    r_func = robjects.r(rcode)
    r_samples = np.array(
        r_func(size=r_sizes, n_samp=r_n_samp, seed=r_seed)
    ).T  # <-- transpose so that 1st row = samples; second row = probs

    # Get sampled locations and corresponding shape data
    sampled_psus_idx = r_samples[0].astype(int) - 1  # <-- R is 1-indexed
    pps_sampled_psus = all_psus[sampled_psus_idx]

    return (
        pps_sampled_psus,
        sampled_psus_idx,
        r_samples[1],
    )