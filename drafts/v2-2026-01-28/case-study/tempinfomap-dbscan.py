import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN  # For DBSCAN clustering
import time
import alphashape
import shapely.geometry as geom
import geopandas as gpd

# --- INITIAL STEPS ---
t_start = time.perf_counter()

print(f"[{time.perf_counter()-t_start:.2f}s] Loading data...")
df = pd.read_csv("C:/Users/13647/OneDrive/Desktop/MiMundo/Projects/TrajectoryAnalysis/case-study/trajectory_dbscan_zoi.csv")

# --- CONVERT TO GEOPANDAS + PROJECT TO METERS ---
gdf_points = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs="EPSG:4326"  # WGS84
)

# Use a local UTM zone for meters (Chengdu ~ EPSG:32648)
gdf_points = gdf_points.to_crs(epsg=32648)
coords_m = np.vstack([gdf_points.geometry.x.values, gdf_points.geometry.y.values]).T

# --- RUN DBSCAN (assuming ZOI_DBSCAN is binary mask; cluster only on ZOI=1 points) ---
# Filter to ZOI points
zoi_df = df[df['ZOI_DBSCAN'] == 1]
if len(zoi_df) == 0:
    print("No ZOI points found. Skipping clustering.")
    df['community'] = -1
else:
    zoi_indices = zoi_df.index
    coords_m_zoi = coords_m[zoi_indices.values]  # Subset meters

    t_db_start = time.perf_counter()
    print(f"[{time.perf_counter()-t_start:.2f}s] Running DBSCAN on ZOI points...")

    eps_m = 1000.0  # DBSCAN eps in meters (tune as needed)
    min_samples = 5  # minPts (tune as needed)

    db = DBSCAN(eps=eps_m, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(coords_m_zoi)

    # Map labels back to original df (noise -1, clusters >=0)
    df['community'] = -1
    df.loc[zoi_indices, 'community'] = labels

print(f"DBSCAN complete in {time.perf_counter() - t_db_start:.2f}s")

# --- ALPHA-SHAPE HULLS (METERS) ---
unique_comms = [c for c in df['community'].unique() if c != -1 and c >= 0]  # Exclude noise

alpha_m = 0.0002  # Example alpha (tune; but in meters, use sensible like 200m)
community_polygons = {}
print("\nBuilding α-shape polygons in meters...")
for comm_id in unique_comms:
    subset = df[df['community'] == comm_id]
    num_points = len(subset)
    
    if num_points < 4:
        continue
    
    # Subset meters
    comm_indices = subset.index
    points_m = coords_m[comm_indices.values]
    
    poly = alphashape.alphashape(points_m, alpha_m)
    community_polygons[comm_id] = poly

# --- CREATE GEOJSON FOR LEAFLET ---
gdf_list = []
for comm_id, poly in community_polygons.items():
    gdf_list.append(gpd.GeoDataFrame(
        {'community':[comm_id]},
        geometry=[poly],
        crs="EPSG:32648"
    ))

gdf = pd.concat(gdf_list, ignore_index=True)
# Convert back to lat/lon for Leaflet
gdf = gdf.to_crs(epsg=4326)
gdf.to_file("dbscan_polygon.geojson", driver="GeoJSON")

print(f"Saved α-shape polygons in meters to 'dbscan_polygon.geojson'")