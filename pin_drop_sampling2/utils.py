import folium
import numpy as np
import s2sphere
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely import Point, Polygon

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
