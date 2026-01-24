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
df = pd.read_csv("C:/Users/13647/OneDrive/Desktop/MiMundo/Projects/TrajectoryAnalysis/case-study/trajectory_vbkde_zoi.csv")

# --- CONVERT TO GEOPANDAS + PROJECT TO METERS ---
gdf_points = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs="EPSG:4326"  # WGS84
)

# Use a local UTM zone for meters (Chengdu ~ EPSG:32648)
gdf_points = gdf_points.to_crs(epsg=32648)
coords_m = np.vstack([gdf_points.geometry.x.values, gdf_points.geometry.y.values]).T
adf_values = df['VBKDE'].values

# --- BUILD GRAPH ---
t0 = time.perf_counter()
print(f"[{time.perf_counter()-t_start:.2f}s] Building KDTree...")
tree = cKDTree(coords_m)

sigma_m = 1000.0
max_dist = sigma_m * 5
pairs = tree.query_pairs(r=max_dist)
print(f"[{time.perf_counter()-t_start:.2f}s] Found {len(pairs)} potential edges.")

G = nx.Graph()
G.add_nodes_from(range(len(df)))

log_interval = 100
count = 0
total_pairs = len(pairs)
chunk_start = time.perf_counter()

for i, j in pairs:
    count += 1
    
    dist = np.linalg.norm(coords_m[i] - coords_m[j])
    weight = min(adf_values[i], adf_values[j]) * np.exp(-dist / sigma_m)
    
    if weight > 0:
        G.add_edge(i, j, weight=weight)
    
    if count % log_interval == 0 or count == total_pairs:
        now = time.perf_counter()
        elapsed = now - chunk_start
        speed = log_interval / elapsed if elapsed > 0 else 0
        percent = (count / total_pairs) * 100
        print(f"{percent:>6.1f}% | {count:>10} edges | speed: {speed:>6.0f}/s")
        chunk_start = time.perf_counter()

print(f"Graph construction complete in {time.perf_counter() - t0:.2f}s")

# --- RUN INFOMAP ---
print(f"[{time.perf_counter()-t_start:.2f}s] Running Infomap...")
infomap_wrapper = Infomap("--two-level --silent")
for u, v, data in G.edges(data=True):
    infomap_wrapper.add_link(u, v, data['weight'])
infomap_wrapper.run()

communities = {node.node_id: node.module_id for node in infomap_wrapper.nodes}
df['community'] = df.index.map(communities).fillna(-1)

# --- ALPHA-SHAPE HULLS (METERS) ---
unique_comms = [c for c in df['community'].unique() if c != -1]

alpha_m = 0.0002
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
gdf.to_file("vbkde_polygon.geojson", driver="GeoJSON")

print(f"Saved α-shape polygons in meters to 'zoi_polygons_meters.geojson'")
