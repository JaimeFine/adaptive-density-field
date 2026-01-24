import numpy as np
import pandas as pd
import faiss
import json
import time
from scipy.spatial import cKDTree
import networkx as nx
from infomap import Infomap

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

df = pd.read_csv("D:/ADataBase/china_poi.csv")

pos = np.vstack([
    geodetic2ecef(lon, lat, alt)
    for lon, lat, alt in df[["lon", "lat", "alt"]].to_numpy()
]).astype("float32")

s = df["poi_score"].to_numpy()
n = len(pos)

quantizer = faiss.IndexFlatL2(3)
# 4096 is the 12 power of two, interesting!
index = faiss.IndexIVFFlat(quantizer, 3, 4096)

index.train(pos)
index.add(pos)

index.nprobe = 16

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

# --- CALCULATE ADF FOR THE TRACK --- #
with open("D:/ADataBase/flights_data_geojson/2024-12-16/2024-12-16-CTU_processed.geojson") as flight:
    track = json.load(flight)

track_coords = np.array([
    f["geometry"]["coordinates"] 
    for f in track["features"] 
    if f["geometry"]["type"] == "Point"
])

track_converted = trajectory2ecef(track_coords)
track_adf = get_adf_value(track_converted)

# --- THE SENSITIVITY SWEEP LOOP --- #
alphas_to_test = [1.1, 1.2, 1.3, 1.4, 1.5]
all_results = []

print(f"Starting Sensitivity Analysis for {len(alphas_to_test)} alpha values...")

for a in alphas_to_test:
    t_a = time.time()
    
    # Apply Masking
    baseline = np.median(track_adf)
    is_zoi = track_adf >= (a * baseline)
    
    # Extract only ZOI points for this alpha to build the graph
    zoi_indices = np.where(is_zoi)[0]
    zoi_coords = track_coords[zoi_indices]
    zoi_adf = track_adf[zoi_indices]
    
    if len(zoi_indices) > 0:
        # Build local graph for these ZOI points
        tree = cKDTree(zoi_coords[:, :2]) # Use lon/lat for graph distance
        sigma = 0.01
        pairs = tree.query_pairs(r=sigma * 5)
        
        G = nx.Graph()
        for i, j in pairs:
            dist = np.linalg.norm(zoi_coords[i, :2] - zoi_coords[j, :2])
            weight = min(zoi_adf[i], zoi_adf[j]) * np.exp(-dist / sigma)
            if weight > 0:
                G.add_edge(i, j, weight=weight)
        
        # Run Infomap
        im = Infomap("--two-level --silent")
        for u, v, data in G.edges(data=True):
            im.add_link(u, v, data['weight'])
        im.run()
        
        communities = {node.node_id: node.module_id for node in im.nodes}
    else:
        communities = {}

    # Store results for this alpha
    for idx in range(len(track_coords)):
        # If point was a ZOI, get its community, otherwise -1
        comm_id = communities.get(np.where(zoi_indices == idx)[0][0], -1) if is_zoi[idx] else -1
        
        all_results.append({
            "lon": track_coords[idx, 0],
            "lat": track_coords[idx, 1],
            "ADF": track_adf[idx],
            "alpha_val": a,
            "is_zoi": int(is_zoi[idx]),
            "community_id": comm_id
        })
    
    print(f"Alpha {a} processed in {time.time() - t_a:.2f}s | Communities found: {len(set(communities.values()))}")

# --- EXPORT TO COMPREHENSIVE CSV --- #
sensitivity_df = pd.DataFrame(all_results)
sensitivity_df.to_csv("alpha_sensitivity_results.csv", index=False)
print("Sensitivity analysis saved to alpha_sensitivity_results.csv")