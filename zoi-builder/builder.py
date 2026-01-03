import numpy as np
import pandas as pd
import faiss

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

# ----------------- Benchmarking --------------- #

import time
Q = 500
t0 = time.time()
for p in pos[:Q]:
    _ = adf(p)

t1 = time.time()
print("Time per query:", (t1 - t0) / Q, "seconds")

# ------------------------------------------ #
#               ZOI Extraction               #
# ------------------------------------------ #

from gensim.models import Word2Vec
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

zoi_mask = zoi_masking(track_adf, 1.3)

track_df = pd.DataFrame({
    "lon": track_coords[:, 0],
    "lat": track_coords[:, 1],
    "alt": track_coords[:, 2],
    "ADF": track_adf,
    "ZOI": zoi_mask.astype(int)
})

# Sample visualization in 2D (3D is not ideal for visualization)
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))

plt.scatter(df["lon"], df["lat"], s=1, alpha=0.03, color="gray")

plt.scatter(
    track_df["lon"],
    track_df["lat"],
    c=track_df["ADF"],
    cmap="viridis",
    s=4
)

zoi_df = track_df[track_df["ZOI"] == 1]
plt.scatter(
    zoi_df["lon"],
    zoi_df["lat"],
    color="red",
    s=6,
    label="ZOI"
)

plt.plot(track_df["lon"], track_df["lat"], color="white", linewidth=1)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Trajectory-Conditioned ZOI via Adaptive Density Field")
plt.legend()
plt.show()

# Save for R's leaflet:
df[["lon", "lat"]].to_csv(
    "poi_background.csv", index=False
)

track_df.to_csv("trajectory_adf_zoi.csv", index=False)