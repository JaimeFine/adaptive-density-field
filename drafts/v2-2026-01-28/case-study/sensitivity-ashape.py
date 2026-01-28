import pandas as pd
import numpy as np
import networkx as nx
from infomap import Infomap
from scipy.spatial import cKDTree
import time
import alphashape
import shapely.geometry as geom
import geopandas as gpd

# --- INITIAL STEPS ---
t_start = time.perf_counter()

print(f"[{time.perf_counter()-t_start:.2f}s] Loading data...")
df = pd.read_csv("C:/Users/13647/OneDrive/Desktop/MiMundo/Projects/TrajectoryAnalysis/data/trajectory_adf_zoi.csv")

df = df[df["ZOI"] == 1]

# --- CONVERT TO GEOPANDAS + PROJECT TO METERS ---
gdf_points = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs="EPSG:4326"  # WGS84
)

# Use a local UTM zone for meters (Chengdu ~ EPSG:32648)
gdf_points = gdf_points.to_crs(epsg=32648)
coords_m = np.vstack([gdf_points.geometry.x.values, gdf_points.geometry.y.values]).T
adf_values = df['ADF'].values

# --- BUILD GRAPH ---
t0 = time.perf_counter()
print(f"[{time.perf_counter()-t_start:.2f}s] Building KDTree...")
tree = cKDTree(coords_m)

sigma_m = 1000.0
max_dist = sigma_m * 5
pairs = tree.query_pairs(r=max_dist)
print(f"[{time.perf_counter()-t_start:.2f}s] Found {len(pairs)} potential edges.")

if len(pairs) > 0:
    # Convert to array (vectorized)
    pair_array = np.fromiter(
        (p for pair in pairs for p in pair),
        dtype=np.int32
    ).reshape(-1, 2)

    i_idx = pair_array[:, 0]
    j_idx = pair_array[:, 1]

    # Vectorized distances
    diffs = coords_m[i_idx] - coords_m[j_idx]
    dists = np.linalg.norm(diffs, axis=1)

    # Vectorized weights
    adf_min = np.minimum(adf_values[i_idx], adf_values[j_idx])
    weights = adf_min * np.exp(-dists / sigma_m)

    # Keep positive weights
    mask = weights > 0
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    weights = weights[mask]
else:
    i_idx = j_idx = weights = np.array([])

print(f"Graph construction complete in {time.perf_counter() - t0:.2f}s")

# --- RUN INFOMAP ---
print(f"[{time.perf_counter()-t_start:.2f}s] Running Infomap...")
infomap_wrapper = Infomap("--two-level --silent")
for u, v, w in zip(i_idx, j_idx, weights):
    infomap_wrapper.add_link(u, v, float(w))
infomap_wrapper.run()

communities = {node.node_id: node.module_id for node in infomap_wrapper.nodes}
df['community'] = df.index.map(communities).fillna(-1)

# --- ALPHA-SHAPE HULLS (METERS) ---
unique_comms = [c for c in df['community'].unique() if c != -1]

alpha_m = 0.002
community_polygons = {}
print("\nBuilding α-shape polygons in meters...")
for comm_id in unique_comms:
    points = df[df['community'] == comm_id][['lon', 'lat']]
    
    # Project points to meters again for alpha-shape
    gdf_comm = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(points.lon, points.lat),
        crs="EPSG:4326"
    ).to_crs(epsg=32648)
    
    coords_comm_m = np.vstack([gdf_comm.geometry.x.values, gdf_comm.geometry.y.values]).T
    num_points = len(coords_comm_m)
    
    if num_points >= 4:
        poly = alphashape.alphashape(coords_comm_m, alpha_m)
        community_polygons[comm_id] = poly
    else:
        community_polygons[comm_id] = geom.MultiPoint(coords_comm_m)

# --- CREATE GEOJSON FOR LEAFLET ---
gdf_list = []
for comm_id, poly in community_polygons.items():
    gdf_list.append(gpd.GeoDataFrame(
        {'community':[comm_id]},
        geometry=[poly],
        crs="EPSG:32648"
    ))

gdf = pd.concat(gdf_list, ignore_index=True)
# Convert back to lat/lon for Leaflet
gdf = gdf.to_crs(epsg=4326)
gdf.to_file("adf_polygon.geojson", driver="GeoJSON")

print(f"Saved α-shape polygons in meters to 'zoi_polygons_meters.geojson'")
