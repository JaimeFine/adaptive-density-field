import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import faiss
import json
import time

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

with open("data/2024-12-16-CTU_processed.geojson") as flight:
    track = json.load(flight)

# --------------- Background Data Processing -------------- #
df = pd.read_csv("data/poi_without_ctu.csv")

pos = np.vstack([
    geodetic2ecef(lon, lat, alt)
    for lon, lat, alt in df[["lon", "lat", "alt"]].to_numpy()
]).astype("float32")

pos_poi_label = np.ones(len(pos), dtype=np.int8)

flights = {}

for feature in track["features"]:
    fid = feature["properties"]["flight_id"]
    lon, lat, alt = feature["geometry"]["coordinates"]

    flights.setdefault(fid, []).append([lon, lat, alt])

# ------------ I-don't-know-initialization ----------- #
total_time = 0
avg_ms_per_point = []

# --------------- KNN Initialization -------------- #
k_vals = [25, 50, 75, 500]

for k in k_vals:
    total_query = 0
    all_rows = []

    start_bench = time.perf_counter()
    print(f"{'Flight_ID':<15} | {'Points':<8}")
    
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(pos)

    for fid, coords in flights.items():
        coords = np.array(coords)
        num_points_in_flight = len(coords)

        track_ecef = np.vstack([
            geodetic2ecef(lon, lat, alt)
            for lon, lat, alt in coords
        ]).astype("float32")

        track_knn = np.array([
            knn.kneighbors(p.reshape(1, -1), return_distance=True)[0].mean()
            for p in track_ecef
        ])

        percent = 75
        threshold = np.percentile(track_knn, percent)
        poi_mask = track_knn <= threshold

        for (lon, lat, alt), knn_val, poi in zip(coords, track_knn, poi_mask):
            if int(poi) == 1:
                all_rows.append([fid, lon, lat, alt, knn_val, 1])

        print(f"{fid:<15} | {len(coords):<8}")
        total_query += num_points_in_flight

    end_bench = time.perf_counter()

    total_time = end_bench - start_bench
    avg_time_per_query = total_time / total_query
    avg_ms_per_point.append(avg_time_per_query * 1000)

    output = pd.DataFrame(all_rows, columns=[
        "flight_id", "lon", "lat", "alt", "KNN", "POI"
    ])

    output.to_csv(f"data/ctu_knn_poi_{k}.csv", index=False)

for i, val in enumerate(avg_ms_per_point):
    print(f"k={k_vals[i]} → Average Latency: {val:.4f} ms/query")

"""
k=25 → Average Latency: 0.3671 ms/query
k=50 → Average Latency: 0.3361 ms/query
k=75 → Average Latency: 0.3499 ms/query
k=500 → Average Latency: 0.4753 ms/query
"""

"""
25:
  Threshold_m    TT    FP   FN Precision  Recall       F1
1         200 33401 28561 1612 0.5390562 0.95396 0.688858

50:
  Threshold_m    TT    FP   FN Precision    Recall        F1
1         200 33275 28687 1720 0.5370227 0.9508501 0.6863867

75:
  Threshold_m    TT    FP   FN Precision    Recall        F1
1         200 33122 28840 1812 0.5345534 0.9481308 0.6836608

500:
  Threshold_m    TT    FP   FN Precision Recall        F1
1         200 31791 30171 2682 0.5130725 0.9222 0.6593249
"""