from gensim.models import Word2Vec
import json

with open("data/flight_routes.geojson") as flight:
    track = json.load(flight)

sequences = []
for feature in track["features"]:
    coords = feature["geometry"]["coordinates"]
    tokens = [f"{lon:.3f}_{lat:.3f}" for lon, lat in coords]
    sequences.append(tokens)

model = Word2Vec(
    sentences = sequences,
    vector_size = 125,
    window = 10,
    min_count = 1,
    sg = 1,
    epochs = 25,
    workers = 5, # CPU threads.
    compute_loss = True
)

model.wv.save_word2vec_format("outputs/track_vectors.txt", binary=False)
model.save("outputs/w2v.model")