import numpy as np
import pandas as pd
import faiss
from infomap import Infomap
from scipy.spatial import cKDTree
import time
import geopandas as gpd

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

df_raw = pd.read_csv("D:/ADataBase/china_poi.csv")

pos = np.vstack([
    geodetic2ecef(lon, lat, alt)
    for lon, lat, alt in df_raw[["lon", "lat", "alt"]].to_numpy()
]).astype("float32")

s = df_raw["poi_score"].to_numpy()
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

alphas = [1.1, 1.2, 1.3, 1.4, 1.5]
results = []

for a in alphas:
    zoi_mask = zoi_masking(track_adf, a)

    df = pd.DataFrame({
        "lon": track_coords[:, 0],
        "lat": track_coords[:, 1],
        "alt": track_coords[:, 2],
        "ADF": track_adf,
        "ZOI": zoi_mask.astype(int)
    })

    t_start = time.perf_counter()

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
        pair_array = np.fromiter(
            (p for pair in pairs for p in pair),
            dtype=np.int32
        ).reshape(-1, 2)

        i_idx = pair_array[:, 0]
        j_idx = pair_array[:, 1]

        # distances (vectorized)
        diffs = coords_m[i_idx] - coords_m[j_idx]
        dists = np.linalg.norm(diffs, axis=1)

        # weights (vectorized)
        adf_min = np.minimum(adf_values[i_idx], adf_values[j_idx])
        weights = adf_min * np.exp(-dists / sigma_m)

        # keep positive weights only
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
        infomap_wrapper.add_link(u, v, w)
    infomap_wrapper.run()

    communities_map = {node.node_id: node.module_id for node in infomap_wrapper.nodes}
    df['community'] = df.index.map(communities_map).fillna(-1)

    # 2. Filter communities by size (The "Scientific" Count)
    communities = [node.module_id for node in infomap_wrapper.nodes]
    unique_comms, counts = np.unique(communities, return_counts=True)
    
    valid_communities = sum(1 for count in counts if count >= 4)

    # 3. Count ZOI points (where ZOI masking was True)
    num_zoi_points = df['ZOI'].sum()

    # 4. Store the results for this alpha value
    results.append({
        "alpha": a,
        "Community numbers": valid_communities,
        "ZOI numbers": num_zoi_points
    })
    
    print(f"Alpha {a}: Found {valid_communities} communities and {num_zoi_points} ZOI points.")

# --- AFTER THE LOOP: Export to CSV ---
results_df = pd.DataFrame(results)
results_df.to_csv("alpha_sensitivity_report.csv", index=False)
print("\nSensitivity report saved to 'alpha_sensitivity_report.csv'")