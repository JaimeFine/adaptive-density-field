from collections import defaultdict
import numpy as np
from datetime import datetime
import json
# from structure_lib import Node, LinkedList

# -------------------- Block 1 ----------------- # 
#      Preprocessing with the flight data        #
# ---------------------------------------------- #

flights = defaultdict(lambda: {
    "coords": [],
    "vel": [],
    # "time": [],
    "dt": []
})

with open("D:/ADataBase/flights_data_geojson/2024-11-10/2024-11-10-CAN_processed.geojson") as raw:
    geojson = json.load(raw)

for feature in geojson["features"]:
    props = feature["properties"]
    geom = feature["geometry"]
    
    f_id = props["flight_id"]
    dt = props["dt"]

    lon, lat, alt = geom["coordinates"]
    vx, vy, vz = map(float, props["velocity"].split())

    # timestamp = datetime.fromisoformat(props["timestamp"])
    
    flights[f_id]["coords"].append([lon, lat, alt])
    flights[f_id]["vel"].append([vx, vy, vz])
    flights[f_id]["dt"].append(dt)
    # flights[f_id]["time"].append(timestamp)

for f_id in flights:
    flights[f_id]["coords"] = np.array(flights[f_id]["coords"])
    flights[f_id]["vel"] = np.array(flights[f_id]["vel"])
    flights[f_id]["dt"] = np.array(flights[f_id]["dt"])

# Converting the coordinate systems:
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
    ref_lon, ref_lat, ref_hei = coords[0]

    ecef = np.array([
        geodetic2ecef(lon, lat, hei)
        for lon, lat, hei in coords
    ])
    enu = ecef2enu(ecef, ref_lon, ref_lat, ref_hei)

    return enu
    
# ------------------ Block 2 ----------------- # 
#            Predicting the position           #
# -------------------------------------------- #

# Pure physics-based model ----- advanced:
from scipy.interpolate import CubicHermiteSpline

flight_alpha = {}
physic_better = {}

for f_id in flights:
    coords_raw = flights[f_id]["coords"]
    vel = flights[f_id]["vel"]
    dt = flights[f_id]["dt"]

    coords = flight_conversion(coords_raw)
    size = len(coords)

    curvatures = []
    a = np.zeros((size, 3))

    for i in range(1, size-1):
        a[i] = (vel[i+1] - vel[i-1]) / ((dt[i-1] + dt[i]))

        speed = np.linalg.norm(vel[i])
        if speed > 1e-7:
            k = np.linalg.norm(np.cross(vel[i], a[i])) / speed**3
            curvatures.append(k)

    curvatures = np.array(curvatures)
    k95_raw = np.percentile(curvatures, 95)
    k95 = np.maximum(k95_raw, 1e-12)

    flight_alpha[f_id] = np.log(5) / k95

    physic_better[f_id] = np.zeros((size, 3), dtype = float)

    for i in range(2, size-2):
        idx = [i-2, i-1, i+1, i+2]

        t0 = 0.0
        t1 = dt[i-2]
        tm = (dt[i-2] + dt[i-1])
        t2 = (dt[i-2] + dt[i-1] + dt[i])
        t3 = (dt[i-2] + dt[i-1] + dt[i] + dt[i+1])

        t = np.array([t0, t1, t2, t3])
        p = coords[idx]
        v = vel[idx]

        spline_x = CubicHermiteSpline(t, p[:,0], v[:,0])
        spline_y = CubicHermiteSpline(t, p[:,1], v[:,1])
        spline_z = CubicHermiteSpline(t, p[:,2], v[:,2])

        # Get the Spline prediction:
        spline = np.array([
            spline_x(tm), spline_y(tm), spline_z(tm)
        ], dtype=float)

        # Calculate the Constant-Acceleration prediction:
        dt_sec = dt[i-1]
        ca = coords[i-1] + vel[i-1] * dt_sec + 0.5 * a[i] * dt_sec**2

        # Local curvature:
        speed = np.linalg.norm(vel[i])
        if speed < 1e-7 or speed > 1e+7:
            k = 0.0
        else:
            k = np.linalg.norm(np.cross(vel[i], a[i])) / (speed**3)

        w = np.exp(-flight_alpha[f_id] * k)
        pred = w * ca + (1 - w) * spline

        physic_better[f_id][i] = pred

# ------------------ Block 3 ----------------- #
#          Computation for the loss            #
# -------------------------------------------- #

losses_mahalanobis = {}

for f_id in flights:
    coords_raw = flights[f_id]["coords"]
    preds_raw = physic_better[f_id]
    dt = flights[f_id]["dt"]
    
    coords = flight_conversion(coords_raw)
    size = len(coords)

    preds = preds_raw[2:size-2]
    actuals = coords[2:size-2]
    residuals = preds - actuals

    mean_res = np.mean(residuals, axis=0)
    centered = residuals - mean_res

    mean_dt = float(np.mean(dt[:-1]))
    pred_dt = np.asarray(dt[1:-3], dtype=np.float64)
    rel_dt = pred_dt / mean_dt
    t_factor = np.sqrt(rel_dt + 1e-12)

    cov = np.cov(centered, rowvar=False)
    cov_inv = np.linalg.inv(cov)

    # Compute the mahalanobis loss:
    mahalanobis_raw = np.einsum('ij,ij->i', centered @ cov_inv, centered)
    mahalanobis = np.sqrt(mahalanobis_raw)
    relative_mahala = mahalanobis / t_factor

    losses_mahalanobis[f_id] = relative_mahala

# ---------------- Block 4 ---------------- #
#            Export losses to CSV           #
# ----------------------------------------- #

import csv

rows = []
for f_id, losses in losses_mahalanobis.items():
    for idx, loss in enumerate(losses):
        rows.append([f_id, idx, loss])

csv_path = "C:/Users/13647/OneDrive/Desktop/flight_loss.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["flight_id", "point_index", "mahalanobis_loss"])
    writer.writerows(rows)

print(f"CSV file saved to {csv_path}")


# ---------------- Block 5 ---------------- # 
#              POI Detection                #
# ----------------------------------------- #

# I am thinking about putting all this in RStudio
# For a better visualization and a faster prototyping

# Physics-ML model:
# Sparse attention?
