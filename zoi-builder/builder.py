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

def ecef2enu(xyz, ref_lon, ref_lat, ref_hei):
    ref_xyz = geodetic2ecef(ref_lon, ref_lat, ref_hei)
    lon = np.deg2rad(ref_lon)
    lat = np.deg2rad(ref_lat)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
        [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
    ])

    return (xyz - ref_xyz) @ R.T

def flight_conversion(coords):
    ref_lon = coords["lon"].iloc[0]
    ref_lat = coords["lat"].iloc[0]
    ref_hei = coords["alt"].iloc[0]

    arr = coords[["lon", "lat", "alt"]].to_numpy(dtype=float)
    ecef = np.array([
        geodetic2ecef(lon, lat, hei)
        for lon, lat, hei in arr
    ])
    enu = ecef2enu(ecef, ref_lon, ref_lat, ref_hei)

    return enu

def anisotropic_gaussian_kernel(x, xi, Sigma_inv):
    diff = x - xi
    return np.exp(-0.5 * diff.T @ Sigma_inv @ diff)

def attention_distribution_function(x, poi_positions, poi_scores, Sigma_inv):
    Func = 0.0
    for i, s in zip(poi_positions, poi_scores):
        Func += s * anisotropic_gaussian_kernel(x, i, Sigma_inv)
    return Func

# ------------- Importing Data -------------- #

df = pd.read_csv("D:/ADataBase/china_poi.csv")
x_raw = df[["lon", "lat", "alt"]]
x = flight_conversion(x_raw)
x = x.astype('float32')
s = df["poi_score"]
n = x.shape[0]

# ------------- Form the Clusters --------------- #

cluster_baby = hdbscan.HDBSCAN(
    min_cluster_size=500,
    min_samples=50,
    metric='euclidean',
    core_dist_n_jobs=1,
    algorithm='best',
    approx_min_span_tree=True
)

labels = cluster_baby.fit_predict(x)

for i in labels:
    flight_conversion(i)

attention_distribution_function(labels, x, s)