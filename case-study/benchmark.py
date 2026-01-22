import pandas as pd
import numpy as np
import networkx as nx
from infomap import Infomap
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import time
import alphashape
import shapely.geometry as geom
import geopandas as gpd

# --- INITIAL STEPS ---
t_start = time.perf_counter()

print(f"[{time.perf_counter()-t_start:.2f}s] Loading data...")
df = pd.read_csv("C:/Users/13647/OneDrive/Desktop/MiMundo/Projects/TrajectoryAnalysis/data/trajectory_adf_zoi.csv")

print(f"[{time.perf_counter()-t_start:.2f}s] Building KDTree...")
coords = df[['lon', 'lat']].values
adf_values = df['ADF'].values
tree = cKDTree(coords)

t0 = time.perf_counter()
print(f"[{time.perf_counter()-t_start:.2f}s] Searching for neighbors...")
sigma = 0.01 
max_dist = sigma * 5
pairs = tree.query_pairs(r=max_dist)

print(f"[{time.perf_counter()-t_start:.2f}s] Building Graph with {len(pairs)} potential edges...")
G = nx.Graph()
G.add_nodes_from(range(len(df)))

log_interval = 100  # Print an update every 100 edges
count = 0
total_pairs = len(pairs)
chunk_start = time.perf_counter()
for i, j in pairs:
    count += 1
    
    # Core Logic
    dist = np.linalg.norm(coords[i] - coords[j])
    weight = min(adf_values[i], adf_values[j]) * np.exp(-dist / sigma)
    
    if weight > 0:
        G.add_edge(i, j, weight=weight)
    
    # --- THE LIVE BENCHMARK LOOP ---
    if count % log_interval == 0 or count == total_pairs:
        now = time.perf_counter()
        elapsed = now - chunk_start
        speed = log_interval / elapsed if elapsed > 0 else 0
        percent = (count / total_pairs) * 100
        
        print(f"{percent:>13.1f}% | {count:>18} | {speed:>14.0f}")
        
        # Reset chunk timer
        chunk_start = time.perf_counter()

t_graph = time.perf_counter() - t0
print("-" * 60)
print(f"Graph construction complete in {t_graph:.2f}s")

print(f"[{time.perf_counter()-t_start:.2f}s] Running Infomap...")
infomap_wrapper = Infomap("--two-level --silent")
for u, v, data in G.edges(data=True):
    infomap_wrapper.add_link(u, v, data['weight'])
infomap_wrapper.run()

communities = {node.node_id: node.module_id for node in infomap_wrapper.nodes}

# FIX 1: Map the IDs and fill missing ones with -1 (isolated points)
df['community'] = df.index.map(communities).fillna(-1)

# --- 5. LIVE HULL BENCHMARKING ---
print("\n" + "="*50)
print(f"{'ZOI ID':<10} | {'Points':<10} | {'Compute Time':<15}")
print("-" * 50)

community_hulls = {}
# FIX 2: Only look at valid communities (ignore -1 if you don't want hulls for noise)
unique_comms = [c for c in df['community'].unique() if c != -1]

hull_start_time = time.perf_counter()

alpha = 0.02  # tune for scale (smaller alpha = tighter around points)
community_polygons = {}
for comm_id in unique_comms:
    iter_start = time.perf_counter()
    
    points = df[df['community'] == comm_id][['lon','lat']].values
    num_points = len(points)
    
    if num_points >= 4:
        poly = alphashape.alphashape(points, alpha)
        community_polygons[comm_id] = poly
    else:
        # For very small clusters, just use points as geometry
        community_polygons[comm_id] = geom.MultiPoint(points)
    
    iter_end = time.perf_counter()
    print(f"{int(comm_id):<10} | {num_points:<10} | {iter_end - iter_start:>14.6f}s")

print("-" * 50)
print(f"Total Hull Calculation Time: {time.perf_counter() - hull_start_time:.4f}s")
print("="*50 + "\n")

gdf_list = []
for comm_id, poly in community_polygons.items():
    gdf_list.append(gpd.GeoDataFrame(
        {'community':[comm_id]}, 
        geometry=[poly], 
        crs="EPSG:4326"  # WGS84 lat/lon
    ))

gdf = pd.concat(gdf_list, ignore_index=True)
gdf.to_file("zoi_polygons.geojson", driver="GeoJSON")