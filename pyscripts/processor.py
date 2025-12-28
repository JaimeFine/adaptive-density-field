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

def velocity_conversion(vel, lat):
    lat_rad = np.deg2rad(lat)
    
    ve = vel[:,0] * (
        1111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    )
    vn = vel[:,1] * 111132.92
    vu = vel[:,2]

    return np.stack([ve, vn, vu], axis=1)
    
# ------------------ Block 2 ----------------- # 
#            Computation heavy zone            #
# -------------------------------------------- #

# Pure physics-based model ----- basic:
physic_normal = {}
for f_id in flights:
    coords_raw = flights[f_id]["coords"]
    vel = flights[f_id]["vel"]
    dt = flights[f_id]["dt"]

    coords = flight_conversion(coords_raw)
    size = len(coords) - 1
    physic_normal[f_id] = np.zeros((size, 3), dtype = float)

    for i in range(size):
        # The phsic_matrix's first column is the prediction for 2nd position!!!
        dx = coords[i] + vel[i] * dt[i]
        physic_normal[f_id][i] = dx

"""
Obviously, the output:
...
[ 3.93346485e+04, -9.70215903e+04,  3.61000000e+04],
[ 1.97690042e+04, -4.85948999e+04,  3.61000000e+04],
[ 3.93348216e+04, -9.70219568e+04,  3.61000000e+04],
[-5.31731053e+04, -7.33233705e+04,  3.61000000e+04],
[-3.95573973e+04, -1.84714838e+04,  3.99400000e+04],
[-3.95575754e+04, -1.84716463e+04,  3.61000000e+04],
[-3.42670276e+04, -1.60046179e+04,  3.61000000e+04],
[-1.35313544e+04, -6.92548368e+03,  3.61000000e+04],
...
is stupid!
"""

# Pure physics-based model ----- advanced:
from scipy.interpolate import CubicHermiteSpline

flight_alpha = {}
physic_better = {}

for f_id in flights:
    coords_raw = flights[f_id]["coords"]
    vel_raw = flights[f_id]["vel"]
    dt_raw = flights[f_id]["dt"]

    dt = dt_raw * 60
    vel = velocity_conversion(vel_raw, coords_raw[:,1])
    coords = flight_conversion(coords_raw)
    size = len(coords)

    curvatures = []
    a = np.zeros((size, 3))

    for i in range(1, size-1):
        a[i] = (vel[i+1] - vel[i-1]) / (dt[i-1] + dt[i])

        speed = np.linalg.norm(vel[i])
        if speed > 1e-7:
            k = np.linalg.norm(np.cross(vel[i], a)) / speed**3
            curvatures.append(k)

    curvatures = np.array(curvatures)
    k95 = np.percentile(curvatures, 95)

    flight_alpha[f_id] = np.log(5) / k95

    physic_better[f_id] = np.zeros((size, 3), dtype = float)

    for i in range(2, size-2):
        idx = [i-2, i-1, i+1, i+2]

        t0 = 0.0
        t1 = dt[i-2]
        tm = dt[i-2] + dt[i-1]
        t2 = dt[i-2] + dt[i-1] + dt[i]
        t3 = dt[i-2] + dt[i-1] + dt[i] + dt[i+1]

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
        ca = coords[i-1] + vel[i-1] * dt[i-1] + a[i]/2 * dt[i-1]**2

        # Local curvature:
        speed = np.linalg.norm(vel[i])
        if speed < 1e-7 or speed > 1e+7:
            k = 0.0
        else:
            k = np.linalg.norm(np.cross(vel[i], a[i])) / (speed**3)

        w = np.exp(-flight_alpha[f_id] * k)
        pred = w * ca + (1 - w) * spline

        physic_better[f_id][i] = pred

"""
Predictions:
       [ 1.92588495e+04, -1.67535715e+04,  8.33351098e+04],
       [ 2.39458001e+04, -2.09093802e+04,  9.65352975e+04],
       [ 1.82348173e+04, -1.61653065e+04,  5.76233957e+04],
       [ 4.94297431e+04, -6.95793030e+04,  1.34560978e+05],
       [ 3.18711824e+04, -3.72383145e+04,  2.26000000e+04],
       [ 1.21706109e+04, -4.83397406e+04,  5.14742306e+04],
       [ 1.04023143e+04, -4.98191322e+04,  2.32394576e+04],
       [ 4.85419442e+04, -9.10575205e+04,  2.34990668e+05],
look better, at least
"""

# Physics-ML model:


# Sparse attention?

