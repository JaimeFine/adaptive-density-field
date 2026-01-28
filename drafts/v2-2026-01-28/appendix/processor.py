from collections import defaultdict
import numpy as np
import json
import os

# -------------------- Block 1 ----------------- # 
#      Preprocessing with the flight data        #
# ---------------------------------------------- #

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

folder_path = "D:/ADataBase/flights_data_geojson/2024-12-16/"
files = [f for f in os.listdir(folder_path) if f.endswith(".geojson")]
output_folder = "D:/ADataBase/poi_data_csv/"

for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    print(f"Processing with {file_path} !!!")

    poi_csv_path = os.path.join(
        output_folder,
        os.path.splitext(file_name)[0] + "_poi.csv"
    )

    if os.path.exists(poi_csv_path):
        print(f"\033[92mSkipping {file_name}, already processed.\033[0m")
        continue

    flights = defaultdict(lambda: {
        "coords": [],
        "vel": [],
        "dt": []
    })

    with open(file_path, "r", encoding="utf-8") as raw:
        geojson = json.load(raw)

    for feature in geojson["features"]:
        props = feature["properties"]
        geom = feature["geometry"]
        
        f_id = props["flight_id"]
        dt = props["dt"]

        lon, lat, alt = geom["coordinates"]
        vx, vy, vz = map(float, props["velocity"].split())
        
        flights[f_id]["coords"].append([lon, lat, alt])
        flights[f_id]["vel"].append([vx, vy, vz])
        flights[f_id]["dt"].append(dt)

    for f_id in flights:
        flights[f_id]["coords"] = np.array(flights[f_id]["coords"])
        flights[f_id]["vel"] = np.array(flights[f_id]["vel"])
        flights[f_id]["dt"] = np.array(flights[f_id]["dt"][:-1], dtype=float)
        
    # ------------------ Block 2 ----------------- # 
    #            Predicting the position           #
    # -------------------------------------------- #

    # Pure physics-based model ----- advanced:
    from scipy.interpolate import CubicHermiteSpline

    flight_alpha = {}
    physic_better = {}
    coords = []
    print("Predicting the position...")

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
        if len(curvatures) == 0:
            print(f"\033[91mSkipping flight {f_id}: no valid curvatures\033[0m")
            continue
        k95_raw = np.percentile(curvatures, 95)
        k95 = np.maximum(k95_raw, 1e-12)

        flight_alpha[f_id] = np.log(5) / k95

        physic_better[f_id] = np.zeros((size, 3), dtype = float)
        if np.any(dt <= 0):
            print(f"\033[91mSkipping flight {f_id}: non-monotonic timestamps\033[0m")
            continue

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
    print("Computing the loss...")

    for f_id in flights:
        if f_id not in physic_better:
            continue
        preds_raw = physic_better[f_id]
        dt = flights[f_id]["dt"]
        coords = flight_conversion(flights[f_id]["coords"])
        
        size = len(coords)

        preds = preds_raw[2:size-2]
        if len(preds) == 0:
            print(f"\033[91mSkipping flight {f_id}: not enough points for prediction.\033[0m")
            continue
        actuals = coords[2:size-2]
        residuals = preds - actuals

        mean_res = np.mean(residuals, axis=0)
        centered = residuals - mean_res

        mean_dt = float(np.mean(dt))
        pred_dt = np.asarray(dt[1:-2], dtype=np.float64)
        rel_dt = pred_dt / (mean_dt + 1e-12)
        t_factor = np.sqrt(rel_dt + 1e-12)

        cov = np.cov(centered, rowvar=False) + np.eye(3) * 1e-5 # Tikhonov
        cov_inv = np.linalg.inv(cov)

        # Compute the mahalanobis loss:
        mahalanobis_raw = np.einsum('ij,ij->i', centered @ cov_inv, centered)
        mahalanobis = np.sqrt(mahalanobis_raw)
        relative_mahala = mahalanobis / t_factor

        losses_mahalanobis[f_id] = relative_mahala

    # ---------------- Block 4 ---------------- #
    #            Export losses to CSV           #
    # ----------------------------------------- #

    # This is only for inspection of the current progress

    """
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
    """

    # ---------------- Block 5 ---------------- # 
    #              POI Detection                #
    # ----------------------------------------- #

    import pandas as pd

    pois = []
    print("Detecting POIs...")

    for f_id, losses in losses_mahalanobis.items():
        if len(losses) == 0:
            print(f"\033[91mSkipping POI detection for flight {f_id}: no losses computed.\033[0m")
            continue

        coords = flights[f_id]["coords"]

        score_norm = (losses - losses.min()) / (losses.max() - losses.min() + 1e-12)

        threshold = 0.75
        poi_id = np.where(score_norm >= threshold)[0]

        for idx in poi_id:
            lon, lat, hei = coords[idx + 2]
            score = score_norm[idx]
            pois.append([f_id, idx, lon, lat, hei, score])

    poi_df = pd.DataFrame(pois, columns = [
        "flight_id", "point_index", "lon",
        "lat", "alt", "poi_score"
    ])
    poi_df.to_csv(poi_csv_path, index=False)

    print(f"\033[92mPOI CSV file saved to: {poi_csv_path}\033[0m")

print("\033[92mFinished!!!\033[0m")

# Physics-ML model:
# Sparse attention?
