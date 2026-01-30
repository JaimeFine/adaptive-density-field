import numpy as np
import pandas as pd
import faiss
import json
import time

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
    return np.sum(scores * np.exp(-0.5 * quadratic))

# ------------- Importing Data -------------- #

df = pd.read_csv("data/poi_without_ctu.csv")

pos = np.vstack([
    geodetic2ecef(lon, lat, alt)
    for lon, lat, alt in df[["lon", "lat", "alt"]].to_numpy()
]).astype("float32")

s = df["poi_score"].to_numpy()

# ------------- Creating ADF --------------- #

quantizer = faiss.IndexFlatL2(3)
# 4096 is the 12 power of two, interesting!
index = faiss.IndexIVFFlat(quantizer, 3, 4096)

index.train(pos)
index.add(pos)

index.nprobe = 16

# ------------------------------------------ #
#                POI Extraction              #
# ------------------------------------------ #

with open("data/2024-12-16-CTU_processed.geojson") as flight:
    track = json.load(flight)

flights = {}
for feature in track["features"]:
    fid = feature["properties"]["flight_id"]
    lon, lat, alt = feature["geometry"]["coordinates"]

    flights.setdefault(fid, []).append([lon, lat, alt])

def trajectory2ecef(track):
    return np.vstack([
        geodetic2ecef(lon, lat, alt)
        for lon, lat, alt in track
    ]).astype("float32")

def get_adf_value(track):
    return np.array([adf(p) for p in track])

def zoi_masking(track, alpha):
    threshold = np.percentile(track, alpha)
    return track >= threshold

k_vals = [50, 100, 150]

for k in k_vals:
    all_rows = []
    total_time = 0
    total_query = 0

    start_bench = time.perf_counter()
    print(f"{'Flight_ID':<15} | {'Points':<8}")

    for fid, coords in flights.items():
        coords = np.array(coords)
        num_points_in_flight = len(coords)

        track_ecef = np.vstack([
            geodetic2ecef(lon, lat, alt)
            for lon, lat, alt in coords
        ]).astype("float32")

        track_adf = np.array([adf(p, k=k) for p in track_ecef])

        percent = 75
        threshold = np.percentile(track_adf, percent)
        poi_mask = track_adf >= threshold

        for (lon, lat, alt), adf_val, poi in zip(coords, track_adf, poi_mask):
            all_rows.append([fid, lon, lat, alt, adf_val, int(poi)])

        print(f"{fid:<15} | {len(coords):<8}")

        total_query += num_points_in_flight

    end_bench = time.perf_counter()

    total_time = end_bench - start_bench
    avg_time_per_query = total_time / total_query
    avg_ms_per_point = avg_time_per_query * 1000

    print(f"Total Processing Time: {total_time:.4f} seconds")
    print(f"Total Points Processed: {total_query}")
    print(f"Average Latency: {avg_ms_per_point:.4f} ms/query")

    output = pd.DataFrame(all_rows, columns=[
        "flight_id", "lon", "lat", "alt", "ADF", "POI"
    ])

    output.to_csv(f"data/ctu_adf_poi_k={k}.csv", index=False)

"""
50:
Total Processing Time: 11.8644 seconds
Total Points Processed: 82651
Average Latency: 0.1435 ms/query

100:
Total Processing Time: 13.1911 seconds
Total Points Processed: 82651
Average Latency: 0.1596 ms/query

150:
Total Processing Time: 14.2526 seconds
Total Points Processed: 82651
Average Latency: 0.1724 ms/query
"""