import pandas as pd
import numpy as np
from infomap import Infomap
from KDEpy import FFTKDE
from skimage import measure
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import LineString
import time

t_start = time.perf_counter()

print(f"[{time.perf_counter()-t_start:.2f}s] Loading data...")
df = pd.read_csv("C:/Users/13647/OneDrive/Desktop/MiMundo/Projects/TrajectoryAnalysis/data/trajectory_adf_zoi.csv")

df = df[df["ZOI"] == 1]

gdf_points = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs="EPSG:4326"
)

gdf_points = gdf_points.to_crs(epsg=32648)
coords_m = np.vstack([gdf_points.geometry.x.values, gdf_points.geometry.y.values]).T
adf_values = df['ADF'].values

t0 = time.perf_counter()
print(f"[{time.perf_counter()-t_start:.2f}s] Building KDTree...")
tree = cKDTree(coords_m)

sigma_m = 1000.0
max_dist = sigma_m * 5
pairs = tree.query_pairs(r=max_dist)
print(f"[{time.perf_counter()-t_start:.2f}s] Found {len(pairs)} potential edges.")

if pairs:
    pair_array = np.array(list(pairs))
    i, j = pair_array.T

    dist = np.linalg.norm(coords_m[i] - coords_m[j], axis=1)
    weight = np.minimum(adf_values[i], adf_values[j]) * np.exp(-dist / sigma_m)

    mask = weight > 0
    edge_list = list(zip(i[mask], j[mask], weight[mask]))

else:
    edge_list = []

print(f"Graph construction complete in {time.perf_counter() - t0:.2f}s")

# --- RUN INFOMAP ---
print(f"[{time.perf_counter()-t_start:.2f}s] Running Infomap...")
infomap_wrapper = Infomap("--two-level --silent")
for u, v, w in edge_list:
    infomap_wrapper.add_link(u, v, w)
infomap_wrapper.run()

communities_map = {node.node_id: node.module_id for node in infomap_wrapper.nodes}
df['community'] = df.index.map(communities_map).fillna(-1)

communities = df['community'].values
unique_comms, counts = np.unique(communities[communities >= 0], return_counts=True)

valid_communities = sum(1 for count in counts if count >= 4)

community_polygons = {}

for comm_id in unique_comms:
    print(f"Start processing {comm_id}")
    mask = (np.array(communities) == comm_id)
    X = coords_m[mask]
    weights = adf_values[mask]

    num_points = len(X)

    if num_points >= 4:
        kde = FFTKDE(bw=500).fit(X, weights=weights) # same to the ADF sigma
        grid, density = kde.evaluate(grid_points=1024)


        eps = 1e-12
        threshold = np.min(density) + eps

        n = int(np.sqrt(len(density)))
        dens_grid = density.reshape(n, n)

        gx = grid[:, 0].reshape(n, n)
        gy = grid[:, 1].reshape(n, n)

        contours = measure.find_contours(
            dens_grid,
            level=threshold
        )

        community_polygons[comm_id] = []

        for contour in contours:
            rows = contour[:, 0].astype(int)
            cols = contour[:, 1].astype(int)
            xs = gx[rows, cols]
            ys = gy[rows, cols]
            poly = np.column_stack([xs, ys])
            community_polygons[comm_id].append(poly)
    else:
        print(f"Community {comm_id} does not have enough points")

gdf_list = []

for comm_id, polys in community_polygons.items():
    t_comm = time.perf_counter()

    mask = (np.array(communities) == comm_id)
    X = coords_m[mask]
    xmin, ymin = X.min(axis=0)
    xmax, ymax = X.max(axis=0)
    diag = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
    extension_distance = 0.05 * diag    # In this case, select 5% of extension

    for coords in polys:
        line = LineString(coords).buffer(extension_distance)
        gdf_list.append(
            gpd.GeoDataFrame(
                {'community': [comm_id]},
                geometry=[line],
                crs="EPSG:32648"
            )
        )
    print(f"Community {comm_id} processed in {time.perf_counter() - t_comm:.4f} seconds")

if len(gdf_list) == 0:
    print("No contours generated.")
else:
    gdf = pd.concat(gdf_list, ignore_index=True)
    # Convert back to lat/lon for Leaflet
    gdf = gdf.to_crs(epsg=4326)
    gdf.to_file("zoi_contour_meters.geojson", driver="GeoJSON")
    print("Saved KDE contour lines to 'zoi_contour_meters.geojson'")