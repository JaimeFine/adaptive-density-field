import pandas as pd
import numpy as np
import networkx as nx
from infomap import Infomap
from scipy.spatial import cKDTree, ConvexHull
import matplotlib.pyplot as plt
import time

# --- INITIAL STEPS ---
t_start = time.perf_counter()

print(f"[{time.perf_counter()-t_start:.2f}s] Loading data...")
df = pd.read_csv("C:/Users/13647/OneDrive/Desktop/MiMundo/Projects/TrajectoryAnalysis/data/trajectory_adf_zoi.csv")

print(f"[{time.perf_counter()-t_start:.2f}s] Building KDTree...")
coords = df[['lon', 'lat']].values
adf_values = df['ADF'].values
tree = cKDTree(coords)

print(f"[{time.perf_counter()-t_start:.2f}s] Searching for neighbors...")
sigma = 0.01 
max_dist = sigma * 5
pairs = tree.query_pairs(r=max_dist)

print(f"[{time.perf_counter()-t_start:.2f}s] Building Graph with {len(pairs)} potential edges...")
G = nx.Graph()
G.add_nodes_from(range(len(df)))
for i, j in pairs:
    dist = np.linalg.norm(coords[i] - coords[j])
    weight = min(adf_values[i], adf_values[j]) * np.exp(-dist / sigma)
    if weight > 0:
        G.add_edge(i, j, weight=weight)

print(f"[{time.perf_counter()-t_start:.2f}s] Running Infomap...")
infomap_wrapper = Infomap("--two-level --silent")
for u, v, data in G.edges(data=True):
    infomap_wrapper.add_link(u, v, data['weight'])
infomap_wrapper.run()

communities = {node.node_id: node.module_id for node in infomap_wrapper.nodes}
df['community'] = df.index.map(communities)

# --- LIVE HULL BENCHMARKING ---
print("\n" + "="*50)
print(f"{'ZOI ID':<10} | {'Points':<10} | {'Compute Time':<15}")
print("-" * 50)

community_hulls = {}
unique_comms = df['community'].unique()
hull_start_time = time.perf_counter()

for comm_id in unique_comms:
    iter_start = time.perf_counter()
    
    points = df[df['community'] == comm_id][['lon', 'lat']].values
    num_points = len(points)
    
    if num_points >= 3:
        hull = ConvexHull(points)
        community_hulls[comm_id] = points[hull.vertices]
    else:
        community_hulls[comm_id] = points
    
    iter_end = time.perf_counter()
    # PRINT RESULT IMMEDIATELY AFTER EACH CLUSTER
    print(f"{int(comm_id):<10} | {num_points:<10} | {iter_end - iter_start:>14.6f}s")

print("-" * 50)
print(f"Total Hull Calculation Time: {time.perf_counter() - hull_start_time:.4f}s")
print("="*50 + "\n")

# --- PLOTTING ---
print(f"[{time.perf_counter()-t_start:.2f}s] Generating Plot...")
plt.figure(figsize=(10,8))
for comm_id, hull_points in community_hulls.items():
    plt.fill(hull_points[:,0], hull_points[:,1], alpha=0.3)
plt.scatter(df['lon'], df['lat'], c='k', s=10, alpha=0.1)
plt.title("Live-Benchmarked ZOIs")
plt.show()