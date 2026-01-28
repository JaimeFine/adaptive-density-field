import numpy as np
import pandas as pd
import faiss
import json
import time
import matplotlib.pyplot as plt

# -------------- 1. Coordinate Conversion ---------------- #
axis = 6378137.0
flattening = 1 / 298.257223563
eccentricity2 = flattening * (2 - flattening)

def geodetic2ecef(lon, lat, hei):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    N = axis / np.sqrt(1 - eccentricity2 * np.sin(lat)**2)
    x = (N + hei) * np.cos(lat) * np.cos(lon)
    y = (N + hei) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - eccentricity2) + hei) * np.sin(lat)
    return np.array([x, y, z])

# -------------- 2. Data Loading & Indexing ---------------- #
df = pd.read_csv("D:/ADataBase/china_poi.csv")
pos = np.vstack([
    geodetic2ecef(lon, lat, alt)
    for lon, lat, alt in df[["lon", "lat", "alt"]].to_numpy()
]).astype("float32")

s = df["poi_score"].to_numpy()

# Initialize Faiss
quantizer = faiss.IndexFlatL2(3)
index = faiss.IndexIVFFlat(quantizer, 3, 4096)
index.train(pos)
index.add(pos)
index.nprobe = 16

# -------------- 3. VBKDE Bandwidth Pre-calculation ---------------- #
print("Calculating Pilot Densities for VBKDE...")
# Use a fixed pilot bandwidth (e.g., 1000m) to estimate local density
h_pilot = 1000.0
k_pilot = 50 

# Search neighbors for all points to get pilot density
dist, _ = index.search(pos, k_pilot)
# Pilot density is inversely proportional to the distance to neighbors
pilot_density = 1.0 / (np.mean(dist, axis=1) + 1e-6)

# Calculate local bandwidths (Abramson's law)
g = np.exp(np.mean(np.log(pilot_density))) # Geometric mean
lambda_i = np.sqrt(g / pilot_density)
h0 = 500.0  # Global smoothing parameter
sigma_local = h0 * lambda_i

# -------------- 4. Core VBKDE Function ---------------- #
def vbkde(x, k=100):
    _, idx = index.search(x.reshape(1, 3), k)
    neighbors = pos[idx[0]]
    bandwidths = sigma_local[idx[0]]
    scores = s[idx[0]]

    # Compute Euclidean distances
    diff = neighbors - x
    dist_sq = np.sum(diff**2, axis=1)
    
    # Gaussian Kernel with variable bandwidth
    # weight = exp(-0.5 * (d / sigma)^2)
    weights = np.exp(-0.5 * dist_sq / (bandwidths**2))
    
    # Result is weighted sum of scores
    return np.sum(scores * weights)

# -------------- 5. Trajectory Processing ---------------- #
with open("D:/ADataBase/flights_data_geojson/2024-12-16/2024-12-16-CTU_processed.geojson") as f:
    track_data = json.load(f)

track_coords = np.array([
    feat["geometry"]["coordinates"] 
    for feat in track_data["features"] 
    if feat["geometry"]["type"] == "Point"
])

def trajectory2ecef(coords):
    return np.vstack([geodetic2ecef(p[0], p[1], p[2]) for p in coords]).astype("float32")

track_ecef = trajectory2ecef(track_coords)

print("Calculating VBKDE for trajectory...")
t_start = time.time()
track_vbkde = np.array([vbkde(p) for p in track_ecef])
print(f"VBKDE processing took: {time.time() - t_start:.2f}s")

# -------------- 6. ZOI Masking & Output ---------------- #
alpha = 1.3
baseline = np.median(track_vbkde)
zoi_mask = track_vbkde >= (alpha * baseline)

track_df = pd.DataFrame({
    "lon": track_coords[:, 0],
    "lat": track_coords[:, 1],
    "alt": track_coords[:, 2],
    "VBKDE": track_vbkde,
    "ZOI": zoi_mask.astype(int)
})

track_df.to_csv("trajectory_vbkde_zoi.csv", index=False)

# -------------- 7. Visualization ---------------- #

plt.figure(figsize=(10, 6))
plt.scatter(df["lon"], df["lat"], s=0.5, alpha=0.01, color="black", label="POIs")
sc = plt.scatter(track_df["lon"], track_df["lat"], c=track_df["VBKDE"], cmap="magma", s=5)
plt.colorbar(sc, label="VBKDE Intensity")

zoi_pts = track_df[track_df["ZOI"] == 1]
plt.scatter(zoi_pts["lon"], zoi_pts["lat"], color="cyan", s=8, label="VBKDE ZOI")

plt.title("Variable Bandwidth KDE (VBKDE) ZOI Extraction")
plt.legend()
plt.show()