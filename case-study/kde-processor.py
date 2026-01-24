import numpy as np
import pandas as pd
import faiss
import json
import time
import matplotlib.pyplot as plt

# -------------- 1. Coordinate Conversion (ECEF Meters) ---------------- #
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

# -------------- 2. Classical KDE Function ---------------- #
def classical_kde(x, index, pos, k=100, sigma_fixed=500.0):
    # Search for nearest neighbors
    _, idx = index.search(x.reshape(1, 3), k)
    neighbors = pos[idx[0]]
    
    # Euclidean distance in ECEF meters
    diff = neighbors - x
    dist_sq = np.sum(diff**2, axis=1)
    
    # Gaussian Kernel with FIXED sigma
    # Note: We still sum the scores (weighted KDE) or just counts. 
    # For a fair comparison with ADF, we keep the scores but fix sigma.
    weights = np.exp(-0.5 * dist_sq / (sigma_fixed**2))
    
    # result = sum of (Score * Kernel)
    return np.sum(s[idx[0]] * weights)

# ------------- 3. Importing Data -------------- #
df = pd.read_csv("D:/ADataBase/china_poi.csv")
pos = np.vstack([
    geodetic2ecef(lon, lat, alt)
    for lon, lat, alt in df[["lon", "lat", "alt"]].to_numpy()
]).astype("float32")

s = df["poi_score"].to_numpy()

# Faiss Indexing
quantizer = faiss.IndexFlatL2(3)
index = faiss.IndexIVFFlat(quantizer, 3, 4096)
index.train(pos)
index.add(pos)
index.nprobe = 16

# ------------- 4. ZOI Extraction ---------------- #
with open("D:/ADataBase/flights_data_geojson/2024-12-16/2024-12-16-CTU_processed.geojson") as flight:
    track = json.load(flight)

track_coords = np.array([
    f["geometry"]["coordinates"] 
    for f in track["features"] 
    if f["geometry"]["type"] == "Point"
])

def trajectory2ecef(track_coords):
    return np.vstack([
        geodetic2ecef(lon, lat, alt)
        for lon, lat, alt in track_coords
    ]).astype("float32")

track_converted = trajectory2ecef(track_coords)

print("Calculating Classical KDE...")
t0 = time.time()
# Using a fixed sigma of 500m
track_kde = np.array([classical_kde(p, index, pos, sigma_fixed=500.0) for p in track_converted])
print(f"KDE processing took: {time.time() - t0:.2f} seconds")

# ZOI Masking
alpha = 1.3
baseline = np.median(track_kde)
zoi_mask = track_kde >= (alpha * baseline)

track_df = pd.DataFrame({
    "lon": track_coords[:, 0],
    "lat": track_coords[:, 1],
    "alt": track_coords[:, 2],
    "KDE": track_kde,
    "ZOI": zoi_mask.astype(int)
})

# ------------- 5. Visualization -------------- #

plt.figure(figsize=(9, 6))
plt.scatter(df["lon"], df["lat"], s=1, alpha=0.03, color="gray", label="POIs")
plt.scatter(track_df["lon"], track_df["lat"], c=track_df["KDE"], cmap="viridis", s=4)

zoi_df = track_df[track_df["ZOI"] == 1]
plt.scatter(zoi_df["lon"], zoi_df["lat"], color="blue", s=6, label="Classical KDE ZOI")

plt.title("Classical KDE (Fixed Bandwidth) ZOI Extraction")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

track_df.to_csv("trajectory_kde_zoi.csv", index=False)