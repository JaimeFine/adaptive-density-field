import numpy as np
from numpy.linalg import inv
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal
import pandas as pd
import hdbscan
import faiss

# -------------- Define the Functions ---------------- #

axis = 6378137.0
flattening = 1 / 298.257223563
eccentrity2 = flattening * (2 - flattening)

def geodetic2ecef(lon, lat, hei):
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    N = axis / np.sqrt(1 - eccentrity2 * np.sin(lat)**2)

    x = (N + hei) * np.cos(lat) * np.cos(lon)
    y = (N + hei) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - eccentrity2) + hei) * np.sin(lat)

    return np.array([x, y, z])

def gaussian_kernel(x, xi, sigma_inv):
    diff = x - xi
    return np.exp(-0.5 * diff @ sigma_inv @ diff)

def adf(x, k=100, sigma0=500.0):
    _, idx = index.search(x.reshape(1, 3), k)

    Func = 0.0
    for i in idx[0]:
        sigma = sigma0 / (s[i] + 1e-6)
        sigma_inv = np.eye(3) / (sigma ** 2)
        Func += s[i] * gaussian_kernel(x, pos[i], sigma_inv)
    return Func

# ------------- Importing Data -------------- #

df = pd.read_csv("D:/ADataBase/china_poi.csv")

pos = np.vstack([
    geodetic2ecef(lon, lat, alt)
    for lon, lat, alt in df[["lon", "lat", "alt"]].to_numpy()
]).astype("float32")
s = df["poi_score"]
n = len(pos)

index = faiss.IndexFlatL2(3)
index.add(pos)

# ------------- Form the Clusters --------------- #

Fvalues = np.array([
    adf(p)
    for p in pos
])

df["ADF"] = Fvalues

