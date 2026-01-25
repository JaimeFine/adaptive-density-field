import numpy as np
import pandas as pd
import faiss
from infomap import Infomap
from scipy.spatial import cKDTree
import time
import geopandas as gpd
import networkx as nx

# -------------- Define the Functions ---------------- #

axis = 6378137.0
flattening = 1 / 298.257223563
eccentricity2 = flattening * (2 - flattening)

def geodetic2ecef(lon, lat, hei):
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    N = axis / np.sqrt(1 - eccentricity2 * np.sin(lat)**2)

    x = (N + hei) * np.cos(lat) * np.cos(lon)
    y = (N + hei) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - eccentricity2) + hei) * np.sin(lat)

    return np.array([x, y, z])

def adf(x, k=100, sigma0=500.0):
    _, idx = index.search(x.reshape(1, 3), k)
    neighbors = pos[idx[0]]
    scores = s[idx[0]]

    # Vectorized kernel function:
    diff = neighbors - x

    sigma = sigma0 / (scores + 1e-6)
    inverse = 1.0 / (sigma ** 2)

    quadratic = np.sum(diff ** 2 * inverse[:, None], axis=1)
    Func = np.sum(scores * np.exp(-0.5 * quadratic))

    return Func

# ------------- Importing Data -------------- #

df = pd.read_csv("D:/ADataBase/china_poi.csv")

pos = np.vstack([
    geodetic2ecef(lon, lat, alt)
    for lon, lat, alt in df[["lon", "lat", "alt"]].to_numpy()
]).astype("float32")

s = df["poi_score"].to_numpy()
n = len(pos)

# ------------- Creating ADF --------------- #

quantizer = faiss.IndexFlatL2(3)
# 4096 is the 12 power of two, interesting!
index = faiss.IndexIVFFlat(quantizer, 3, 4096)

index.train(pos)
index.add(pos)

index.nprobe = 16

# ------------------------------------------ #
#               ZOI Extraction               #
# ------------------------------------------ #

import json

with open("D:/ADataBase/flights_data_geojson/2024-12-16/2024-12-16-CTU_processed.geojson") as flight:
    track = json.load(flight)

track_coords = np.array([
    f["geometry"]["coordinates"] 
    for f in track["features"] 
    if f["geometry"]["type"] == "Point"
])

def trajectory2ecef(track):
    return np.vstack([
        geodetic2ecef(lon, lat, alt)
        for lon, lat, alt in track
    ]).astype("float32")

def get_adf_value(track):
    return np.array([adf(p) for p in track])

def zoi_masking(track, alpha):
    baseline = np.median(track)
    return track >= alpha * baseline

track_converted = trajectory2ecef(track_coords)
track_adf = get_adf_value(track_converted)
baseline_adf = np.median(track_adf)

df_track = pd.DataFrame({
    "lon": track_coords[:, 0],
    "lat": track_coords[:, 1],
    "alt": track_coords[:, 2],
    "ADF": track_adf
})

# --- PROJECT TO METERS ONCE (full trajectory) ---
gdf = gpd.GeoDataFrame(df_track, geometry=gpd.points_from_xy(df_track.lon, df_track.lat), crs="EPSG:4326")
gdf = gdf.to_crs(epsg=32648)
coords_m = np.vstack([gdf.geometry.x, gdf.geometry.y]).T

# --- RUN GRAPH + INFOMAP ONCE ON FULL TRAJECTORY (fixed sigma=1000m) ---
sigma_fixed = 1000.0

t_graph_start = time.perf_counter()

tree = cKDTree(coords_m)
pairs = tree.query_pairs(r=sigma_fixed * 5)

edge_list = []
if len(pairs) > 0:
    pair_arr = np.array(list(pairs))
    i, j = pair_arr.T
    dists = np.linalg.norm(coords_m[i] - coords_m[j], axis=1)
    weights = np.minimum(track_adf[i], track_adf[j]) * np.exp(-dists / sigma_fixed)
    mask = weights > 0
    edge_list = list(zip(i[mask], j[mask], weights[mask]))

im = Infomap("--two-level --silent")
for u, v, w in edge_list:
    im.add_link(int(u), int(v), float(w))
im.run()

# Assign communities to full dataframe (isolates get -1 later if needed)
communities_full = [node.module_id for node in im.nodes]
# Map back (Infomap only includes connected nodes; isolates are missing)
community_map = {node.node_id: node.module_id for node in im.nodes}
df_track['community'] = pd.Series(community_map).reindex(range(len(df_track))).fillna(-1).astype(int)

# Metrics for the full run (this should exactly match your sigma=1000m sensitivity run)
_, counts = np.unique(df_track['community'][df_track['community'] != -1], return_counts=True)
valid_communities_full = sum(1 for count in counts if count >= 4)

print(f"Full trajectory graph complete in {time.perf_counter() - t_graph_start:.2f}s")
print(f"Full: Edges={len(edge_list)}, Valid Communities={valid_communities_full}")

# --- SENSITIVITY LOOP OVER ALPHA (mask only; communities fixed) ---
alphas_to_test = [1.1, 1.2, 1.3, 1.4, 1.5]
sigma_fixed = 1000.0  # Must match the '1000' in your sigma test
sensitivity_results = []

print(f"{'Alpha':<10} | {'ZOI Pts':<10} | {'Edges':<12} | {'Communities':<12}")
print("-" * 50)

# Project the full trajectory to meters once to save time
gdf_full = gpd.GeoDataFrame(
    df_track, 
    geometry=gpd.points_from_xy(df_track.lon, df_track.lat), 
    crs="EPSG:4326"
).to_crs(epsg=32648)
coords_full_m = np.vstack([gdf_full.geometry.x, gdf_full.geometry.y]).T

for a in alphas_to_test:
    t_start = time.perf_counter()
    
    # STEP A: Apply the Alpha Mask (Just like your ZOI Extraction code)
    is_zoi = track_adf >= (a * baseline_adf)
    zoi_indices = np.where(is_zoi)[0]
    
    if len(zoi_indices) < 4:
        continue
        
    # Get coordinates and ADF for ONLY the points that pass the alpha threshold
    coords_zoi = coords_full_m[zoi_indices]
    adf_zoi = track_adf[zoi_indices]
    
    # STEP B: Build Graph (Same logic as your Sigma sensitivity script)
    tree = cKDTree(coords_zoi)
    max_dist = sigma_fixed * 5
    pairs = tree.query_pairs(r=max_dist)
    
    edge_list = []
    if pairs:
        pair_array = np.array(list(pairs))
        i_idx, j_idx = pair_array.T
        
        # Calculate weights using the FIXED sigma
        dists = np.linalg.norm(coords_zoi[i_idx] - coords_zoi[j_idx], axis=1)
        weights = np.minimum(adf_zoi[i_idx], adf_zoi[j_idx]) * np.exp(-dists / sigma_fixed)
        
        mask = weights > 0
        edge_list = list(zip(i_idx[mask], j_idx[mask], weights[mask]))

    # STEP C: Run Infomap
    im = Infomap("--two-level --silent")
    for u, v, w in edge_list:
        im.add_link(int(u), int(v), float(w))
    im.run()
    
    # STEP D: Record Valid Communities (count >= 4)
    communities = [node.module_id for node in im.nodes]
    unique_comms, counts = np.unique(communities, return_counts=True)
    valid_communities = sum(1 for count in counts if count >= 4)
    
    duration = time.perf_counter() - t_start
    print(f"{a:<10.1f} | {len(zoi_indices):<10} | {len(edge_list):<12} | {valid_communities:<12}")
    
    sensitivity_results.append({
        "alpha": a,
        "zoi_pts": len(zoi_indices),
        "num_communities": valid_communities
    })

# Export
pd.DataFrame(sensitivity_results).to_csv("alpha_sensitivity_final.csv", index=False)