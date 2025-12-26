import networkx as nx
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import json

# Getting the frequency:
with open("data/flight_routes.geojson") as flights:
    track = json.load(flights)

edges = []

for feature in track["features"]:
    coords = feature["geometry"]["coordinates"]
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        x = f"{lon1:.3f}_{lat1:.3f}"
        y = f"{lon2:.3f}_{lat2:.3f}"
        edges.append([x, y])

df = pd.DataFrame(
    edges, columns = ["origin", "destination"]
).groupby(
    ["origin", "destination"]
).size().reset_index(name = "freq")

# Getting the similarity:
embed = KeyedVectors.load_word2vec_format(
    "outputs/track_vectors.txt", binary = False
)

def similarity(x, y):
    v1, v2 = embed[x], embed[y]
    similarity = float(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )
    return (similarity + 1) / 2

similarities = []
for idx, row in df.iterrows():
    similarities.append(similarity(row['origin'], row['destination']))

df['similarity'] = similarities
df['weight'] = df['freq'] * df['similarity']

# Create the graph, ha, discreate mathematics! Cool!!!
G = nx.DiGraph()
for idx, row in df.iterrows():
    G.add_edge(
        row['origin'],
        row['destination'],
        weight = row['weight'],
        freq = row['freq']
    )

df.to_csv("outputs/graph_edges.csv", index=False)