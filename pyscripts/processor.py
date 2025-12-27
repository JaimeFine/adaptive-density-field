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
    
# ------------------ Block 2 ----------------- # 
#            Computation heavy zone            #
# -------------------------------------------- #

# Pure physics-based model ----- basic:
physic_normal = {}
for f_id in flights:
    coords = flights[f_id]["coords"]
    vel = flights[f_id]["vel"]
    dt = flights[f_id]["dt"]

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
from julia import Julia
jl = Julia(compiled_modules=False)
from julia import Interpolations

physic_better = {}
for f_id in flights:
    coords = flights[f_id]["coords"]
    vel = flights[f_id]["vel"]
    dt = flights[f_id]["dt"]

    size = len(coords) - 1
    physic_better[f_id] = np.zeros((size, 3), dtype = float)

    for i in range(1, size - 3):
        x = coords[[i-2, i-1, i+1, i+2], 0]
        y = coords[[i-2, i-1, i+1, i+2], 1]
        z = coords[[i-2, i-1, i+1, i+2], 2]

        vx = vel[[i-2, i-1, i+1, i+2], 0]
        vy = vel[[i-2, i-1, i+1, i+2], 1]
        vz = vel[[i-2, i-1, i+1, i+2], 2]

        interpol_x = jl.CubicHermiteInterpolation(x, y, vy/vx)
        interpol_y = jl.CubicHermiteInterpolation(y, z, vz/vy)
        interpol_z = jl.CubicHermiteInterpolation(x, z, vz/vx)

        # Get the Spline prediction:
        t = dt[i-2] + dt[i-1] 
        spline = [
            interpol_x(t), interpol_y(t), interpol_z(t)
        ]

        # Calculate the Constant-Acceleration prediction:
        a = (vel[i+1] - vel[i-1]) / (dt[i-1] + dt[i])
        ca = coords[i] + vel[i] * dt[i] + 0.5 * a * dt[i] * dt[i]


# Physics-ML model:


# Sparse attention?

