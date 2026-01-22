import pandas as pd
import numpy as np
import networkx as nx
from infomap import Infomap
from scipy.spatial import cKDTree, ConvexHull
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/13647/OneDrive/Desktop/MiMundo/Projects/TrajectoryAnalysis/data/trajectory_adf_zoi.csv")

coords = df[['lon', 'lat']].values
tree = cKDTree(coords)

sigma = 0.01    # A spatial decay parameter
max_dist = sigma * 5
G = nx.Graph()

G.add_nodes_from(range(len(df)))
pairs = tree.query_pairs(r=max_dist)

for i, j in pairs:
    dist = np.linalg.norm(coords[i] - coords[j])
    weight = min(df.at[i, 'ADF'], df.at[j, 'ADF']) * np.exp(-dist / sigma)
    if weight > 0:
        G.add_edge(i, j, weight=weight)

infomap_wrapper = Infomap("--two-level --silent")

for u, v, data in G.edges(data=True):
    infomap_wrapper.add_link(u, v, data['weight'])

infomap_wrapper.run()

communities = {}
for node in infomap_wrapper.nodes:
    communities[node.node_id] = node.module_id

df['community'] = df.index.map(communities)

community_hulls = {}
for comm_id in df['community'].unique():
    points = df[df['community'] == comm_id][['lon', 'lat']].values
    if len(points) >= 3:
        hull = ConvexHull(points)
        community_hulls[comm_id] = points[hull.vertices]
    else:
        community_hulls[comm_id] = points

plt.figure(figsize=(10,8))
for comm_id, hull_points in community_hulls.items():
    plt.fill(hull_points[:,0], hull_points[:,1], alpha=0.3, label=f'ZOI {comm_id}')
plt.scatter(df['lon'], df['lat'], c='k', s=10)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Trajectory-conditioned ZOIs via Infomap")
plt.legend()
plt.show()