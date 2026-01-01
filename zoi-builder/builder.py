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

"""
Note:
    I switched from Direct FlatL2 to IVF-Flat as the suggestion from AI,
    and the Time per query reduced from 0.00703 to 0.000107!!!
"""

# ---------------- Real Stuff ------------------ #

# Replace this code with real data:
Fvalues = np.array([
    adf(p)
    for p in pos
])

df["ADF"] = Fvalues