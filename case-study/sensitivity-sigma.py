import pandas as pd
import numpy as np
import networkx as nx
from infomap import Infomap
from scipy.spatial import cKDTree
import time

# --- 1. INITIAL SETUP ---
df = pd.read_csv("C:/Users/13647/OneDrive/Desktop/MiMundo/Projects/TrajectoryAnalysis/data/trajectory_adf_zoi.csv")
coords = df[['lon', 'lat']].values
adf_values = df['ADF'].values
tree = cKDTree(coords)

sigma_values = [0.002, 0.005, 0.01, 0.015, 0.02]
sensitivity_results = []

print(f"{'Sigma':<10} | {'Edges':<12} | {'Communities':<12} | {'Time (s)':<10}")
print("-" * 50)

# --- 2. THE SENSITIVITY LOOP ---
for s_val in sigma_values:
    t_start = time.perf_counter()
    
    # A. Neighbor Search
    max_dist = s_val * 5
    pairs = tree.query_pairs(r=max_dist)
    
    # B. Graph Construction
    G = nx.Graph()
    G.add_nodes_from(range(len(df)))
    
    edge_list = []
    for i, j in pairs:
        dist = np.linalg.norm(coords[i] - coords[j])
        weight = min(adf_values[i], adf_values[j]) * np.exp(-dist / s_val)
        if weight > 0:
            edge_list.append((i, j, weight))
    
    G.add_weighted_edges_from(edge_list)
    
    # C. Infomap Community Detection
    infomap_wrapper = Infomap("--two-level --silent")
    for u, v, w in G.edges(data='weight'):
        infomap_wrapper.add_link(u, v, w)
    
    infomap_wrapper.run()
    
    # D. Record Metrics
    communities = [node.module_id for node in infomap_wrapper.nodes]
    unique_comms, counts = np.unique(communities, return_counts=True)
    
    # We only care about "significant" communities
    valid_communities = sum(1 for count in counts if count >= 4)
    
    t_end = time.perf_counter()
    duration = t_end - t_start
    
    print(f"{s_val:<10.4f} | {len(edge_list):<12} | {valid_communities:<12} | {duration:<10.2f}")
    
    sensitivity_results.append({
        "sigma": s_val,
        "num_edges": len(edge_list),
        "num_communities": valid_communities,
        "compute_time": duration
    })

# --- 3. EXPORT RESULTS ---
results_df = pd.DataFrame(sensitivity_results)
results_df.to_csv("sigma_sensitivity_report.csv", index=False)
print("\nSensitivity analysis saved to 'sigma_sensitivity_report.csv'")