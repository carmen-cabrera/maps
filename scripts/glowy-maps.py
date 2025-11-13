import os
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLACE = "Nairobi, Kenya"
OUT_DIR = Path("data")

HIGHWAY_FILTER = (
    '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
)

TAGS_FUEL = {"amenity": "fuel"}
TAGS_CHARGING = {"amenity": "charging_station"}
TAGS_WATER = {
    "natural": "water",
    "water": ["river", "canal", "reservoir", "lake", "pond"],
}

# OSMnx settings
ox.settings.use_cache = True
ox.settings.requests_timeout = 300


# ---------------------------------------------------------------------------
# Geometry & data helpers
# ---------------------------------------------------------------------------

def get_city_circle(
    place: str,
    working_crs: str = "EPSG:3857",
    out_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Get a circular polygon roughly covering the given place.

    Returns a GeoDataFrame with a single row containing the circle geometry
    in `out_crs`.
    """
    gdf = ox.geocode_to_gdf(place)
    gdf_merc = gdf.to_crs(working_crs)
    minx, miny, maxx, maxy = gdf_merc.total_bounds

    center_point = Point(
        minx + (maxx - minx) / 2,
        miny + (maxy - miny) / 2,
    )
    radius = max((maxx - minx) / 2, (maxy - miny) / 2)

    center = gpd.GeoDataFrame(
        geometry=[center_point],
        crs=working_crs,
    )
    circle_geom = center.buffer(radius).geometry.iloc[0]

    circle = gpd.GeoDataFrame(
        geometry=[circle_geom],
        crs=working_crs,
    ).to_crs(out_crs)

    return circle


def to_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert geometries to representative points for POI mapping.

    Any non-point geometry is replaced with its representative point.
    Assumes / sets CRS to EPSG:4326.
    """
    if gdf.empty:
        return gdf

    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    else:
        gdf = gdf.to_crs(4326)

    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: geom if geom.geom_type == "Point" else geom.representative_point()
    )
    return gdf


def fetch_pois(
    polygon: gpd.GeoSeries,
    tags: dict,
    empty_message: str,
) -> gpd.GeoDataFrame:
    """
    Fetch OSM features by tags within a polygon and normalize to points.
    """
    g = ox.features_from_polygon(polygon, tags)

    if g.empty:
        raise SystemExit(empty_message)

    points = to_points(g)
    points = points[["geometry"]].set_crs(4326)

    points["lon"] = points.geometry.x
    points["lat"] = points.geometry.y

    return points


def build_road_network(
    polygon: gpd.GeoSeries,
    target_crs: str,
) -> gpd.GeoDataFrame:
    """
    Build a road network graph for the polygon and return edges as GeoDataFrame.
    """
    graph = ox.graph_from_polygon(polygon, custom_filter=HIGHWAY_FILTER)
    edges = ox.graph_to_gdfs(graph, nodes=False)
    return edges.to_crs(target_crs)


def fetch_water_bodies(
    polygon: gpd.GeoSeries,
    target_crs: str,
) -> gpd.GeoDataFrame:
    """
    Fetch water bodies (polygonal features) within polygon.
    """
    water = ox.features_from_polygon(polygon, TAGS_WATER)
    water = water[water.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    if water.empty:
        return water

    return water.to_crs(target_crs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def split_roads_by_type(edges: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Split roads into residential and non-residential groups.
    """
    def is_residential(val) -> bool:
        if isinstance(val, list):
            return "residential" in val
        return val == "residential"

    edges_res = edges[edges["highway"].apply(is_residential)]
    edges_major = edges[~edges.index.isin(edges_res.index)]
    return edges_res, edges_major


def plot_stations(
    edges: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    charging: gpd.GeoDataFrame,
    circle: gpd.GeoDataFrame,
    title: str | None = None,
) -> None:
    """
    Plot petrol stations and EV charging points on top of the road network,
    water bodies, and city circle.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # --- Roads ---
    edges_residential, edges_major = split_roads_by_type(edges)

    edges_major.plot(ax=ax, color="dimgray", linewidth=1, alpha=0.4, zorder=0)
    if not edges_residential.empty:
        edges_residential.plot(
            ax=ax,
            color="dimgray",
            linewidth=0.4,
            alpha=0.3,
            zorder=0,
        )

    # --- Water ---
    if not water.empty:
        water.plot(ax=ax, color="dimgray", edgecolor="none", alpha=0.7, zorder=0)

    # --- EV charging heat-style scatter ---
    for i in np.arange(0.1, 1.01, 0.05):
        t = i / 1.0
        base_color = np.array([0.1, 0.7, 0.99])  # blue-cyan tone
        color = base_color + (1 - base_color) * (1 - t)
        ax.scatter(
            charging.geometry.x,
            charging.geometry.y,
            s=(7 * i) ** 2,
            color=(*color, 0.0083 / i),
            zorder=2,
        )

    ax.scatter(
        charging.geometry.x,
        charging.geometry.y,
        s=(4 * 0.01) ** 2,
        color="white",
        zorder=3,
    )

    # --- Petrol stations heat-style scatter ---
    for i in np.arange(0.1, 1.01, 0.05):
        t = i / 1.0
        base_color = np.array([0.95, 0.008, 0.5])  # purple tone
        color = base_color + (1 - base_color) * (1 - t)
        ax.scatter(
            stations.geometry.x,
            stations.geometry.y,
            s=(10 * i) ** 2,
            color=(*color, 0.0083 / i),
            zorder=2,
        )

    ax.scatter(
        stations.geometry.x,
        stations.geometry.y,
        s=(4 * 0.01) ** 2,
        color="white",
        zorder=3,
    )

    # --- Circle outline & styling ---
    ax.set_facecolor("black")
    circle.plot(ax=ax, fc="none", ec="white", lw=0.7)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    if title:
        ax.set_title(title, color="white", pad=12)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    circle = get_city_circle(PLACE)
    circle_geom = circle.loc[0, "geometry"]

    # Fetch POIs
    stations = fetch_pois(
        circle_geom,
        TAGS_FUEL,
        empty_message="No petrol stations found. Try a different place name or use a bbox.",
    )
    charging = fetch_pois(
        circle_geom,
        TAGS_CHARGING,
        empty_message="No charging points found. Try a different place name or use a bbox.",
    )

    # Roads and water in same CRS as charging layer
    target_crs = charging.crs
    edges = build_road_network(circle_geom, target_crs)
    water = fetch_water_bodies(circle_geom, target_crs)

    # Plot
    plot_stations(
        edges=edges,
        water=water,
        stations=stations,
        charging=charging,
        circle=circle.to_crs(target_crs),
        title=f"Fuel and EV Charging Infrastructure – {PLACE}",
    )

    # Example: save data if desired
    # stations.to_file(OUT_DIR / "nairobi_fuel_stations.geojson", driver="GeoJSON")
    # charging.to_file(OUT_DIR / "nairobi_ev_charging.geojson", driver="GeoJSON")
    # fig.savefig(OUT_DIR / "nairobi_fuel_ev_map.pdf")


if __name__ == "__main__":
    main()




