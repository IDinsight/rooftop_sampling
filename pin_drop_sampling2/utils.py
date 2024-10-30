import folium
import numpy as np
import s2sphere
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely import Point, Polygon
import osmnx as ox
from shapely.ops import nearest_points

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
def get_nearest_point_on_road(point: Point, buffer_distance = 200):
    """
    Finds the nearest point to the given point which lies on a road using OSMs open source road data.

    Parameters:
    - point (Point): The point for which the nearest point on the road network needs to be found.
    - buffer_distance (float): The distance within which to extract the walkable street network.

    Returns:
    - nearest_point (Point): The nearest point on the road network to the given point.
    """
    # Extract the walkable street network for this point within a certain distance
    try:
            # Attempt to extract the street network for this point within a certain distance
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



