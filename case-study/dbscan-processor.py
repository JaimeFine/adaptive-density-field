import numpy as np
import pandas as pd
import faiss
import json
import time
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

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

# -------------- 2. Data Loading ---------------- #
df_poi = pd.read_csv("D:/ADataBase/china_poi.csv")
poi_pos = np.vstack([
    geodetic2ecef(lon, lat, alt)
    for lon, lat, alt in df_poi[["lon", "lat", "alt"]].to_numpy()
]).astype("float32")

with open("D:/ADataBase/flights_data_geojson/2024-12-16/2024-12-16-CTU_processed.geojson") as f:
    track_data = json.load(f)

track_coords = np.array([
    feat["geometry"]["coordinates"] 
    for feat in track_data["features"] 
    if feat["geometry"]["type"] == "Point"
])

def trajectory2ecef(coords):
    return np.vstack([geodetic2ecef(p[0], p[1], p[2]) for p in coords]).astype("float32")

track_pos = trajectory2ecef(track_coords)

# -------------- 3. DBSCAN Comparison Logic ---------------- #
# Note: DBSCAN usually clusters the data itself. 
# For ZOI comparison, we label a trajectory point as "ZOI" if it 
# falls within a dense cluster of POIs.

print("Running DBSCAN-based ZOI extraction...")
t_start = time.time()

# We define a "Dense POI Zone" using DBSCAN
# eps: 1000 meters, min_samples: 15 POIs
db = DBSCAN(eps=1000, min_samples=15, metric='euclidean', n_jobs=-1).fit(poi_pos)
poi_labels = db.labels_

# Extract only the "Core/Cluster" POIs (ignore noise -1)
core_poi_pos = poi_pos[poi_labels != -1]

# Use Faiss to check which trajectory points are near these core clusters
res = faiss.StandardGpuResources() # Optional: if you have GPU
quantizer = faiss.IndexFlatL2(3)
index_core = faiss.IndexFlatL2(3) 
index_core.add(core_poi_pos)

# If a trajectory point is within 1000m of a core POI cluster, it's ZOI
dists, _ = index_core.search(track_pos, 1)
zoi_mask_db = np.sqrt(dists).flatten() <= 1000.0

print(f"DBSCAN processing took: {time.time() - t_start:.2f}s")

# -------------- 4. Output & Formatting ---------------- #
track_df = pd.DataFrame({
    "lon": track_coords[:, 0],
    "lat": track_coords[:, 1],
    "alt": track_coords[:, 2],
    "ZOI_DBSCAN": zoi_mask_db.astype(int)
})

track_df.to_csv("trajectory_dbscan_zoi.csv", index=False)

# -------------- 5. Comparison Visualization ---------------- #

plt.figure(figsize=(12, 6))

# Background POIs
plt.scatter(df_poi["lon"], df_poi["lat"], s=0.5, alpha=0.01, color="black", label="POIs")

# DBSCAN ZOI (The comparison)
dbscan_zoi = track_df[track_df["ZOI_DBSCAN"] == 1]
plt.scatter(dbscan_zoi["lon"], dbscan_zoi["lat"], color="blue", s=10, label="DBSCAN ZOI", alpha=0.6)

# Trajectory line
plt.plot(track_df["lon"], track_df["lat"], color="red", linewidth=1, label="Flight Path", alpha=0.5)

plt.title("DBSCAN-based ZOI Extraction (Baseline Comparison)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()