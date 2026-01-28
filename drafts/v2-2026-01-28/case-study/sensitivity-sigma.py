import pandas as pd
import numpy as np
from infomap import Infomap
from scipy.spatial import cKDTree
import time
import geopandas as gpd # Added for projection

# --- 1. INITIAL SETUP ---
df = pd.read_csv("C:/Users/13647/OneDrive/Desktop/MiMundo/Projects/TrajectoryAnalysis/data/trajectory_adf_zoi.csv")

df = df[df["ZOI"] == 1]

# CHANGE 1: Project to Meters (UTM 48N)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
gdf = gdf.to_crs(epsg=32648)
coords_m = np.vstack([gdf.geometry.x, gdf.geometry.y]).T # These are now in meters

adf_values = df['ADF'].values

# CHANGE 2: Build tree using meter-based coordinates
tree = cKDTree(coords_m)

# CHANGE 3: Use Metric Sigma values (Equivalent to your degree range)
sigma_values = [200, 500, 1000, 1500, 2000] 
sensitivity_results = []

print(f"{'Sigma (m)':<10} | {'Edges':<12} | {'Communities':<12} | {'Time (s)':<10}")
print("-" * 50)

# --- 2. THE SENSITIVITY LOOP (Logic remains exactly the same) ---
for s_val in sigma_values:
    t_start = time.perf_counter()
    
    # A. Neighbor Search (Now using meters)
    max_dist = s_val * 5
    pairs = tree.query_pairs(r=max_dist)
    
    # B. Vectorized Edge Computation
    if pairs:
        pair_array = np.array(list(pairs))
        i_arr, j_arr = pair_array.T
        
        # Using coords_m for Euclidean distance calculation
        coords_i = coords_m[i_arr]
        coords_j = coords_m[j_arr]
        
        dists = np.linalg.norm(coords_i - coords_j, axis=1)
        adf_mins = np.minimum(adf_values[i_arr], adf_values[j_arr])
        weights = adf_mins * np.exp(-dists / s_val)
        
        mask = weights > 0
        edge_list = list(zip(i_arr[mask], j_arr[mask], weights[mask]))
    else:
        edge_list = []
    
    # C. Infomap Community Detection
    infomap_wrapper = Infomap("--two-level --silent")
    for u, v, w in edge_list:
        infomap_wrapper.add_link(u, v, w)
    
    infomap_wrapper.run()
    
    # D. Record Metrics
    communities = [node.module_id for node in infomap_wrapper.nodes]
    unique_comms, counts = np.unique(communities, return_counts=True)
    
    valid_communities = sum(1 for count in counts if count >= 4)
    
    t_end = time.perf_counter()
    duration = t_end - t_start
    
    print(f"{s_val:<10.0f} | {len(edge_list):<12} | {valid_communities:<12} | {duration:<10.2f}")
    
    sensitivity_results.append({
        "sigma": s_val,
        "num_edges": len(edge_list),
        "num_communities": valid_communities,
        "compute_time": duration
    })

# --- 3. EXPORT RESULTS ---
results_df = pd.DataFrame(sensitivity_results)
results_df.to_csv("sigma_sensitivity_report_meters.csv", index=False)
print("\nSensitivity analysis saved to 'sigma_sensitivity_report_meters.csv'")