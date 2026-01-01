# Physics-informed Semantic Airspace Infrastructure Discovery From Trajectory Data

> This project is still in progress
> 
> The current version is only a sketch

## Project Goal

Use aircraft trajectory data to detect spatial communities of flight trajectories, analyze patterns, and visualize them on maps.

-----

## Structure
```txt
project/
  data/                     # Raw and processed data
    flight_loss.csv
  data-pipeline/            # Programs for data proessing and data visualization
    geojson_generator.R
    loss_viewer.R
    poi_sample_viewer.R
  poi-detector/             # Python program for POI detection
    processor.py
    technical-report.md
    technical-report-format1.pdf
    technical-report-format2.pdf
  notebooks/                # Recording all thoughts and brainstormed idea
    poi_detection.ipynb
    motion_prediction.ipynb
    base_map_building.ipynb
  plots/                  # All generated outputs
    kdeplot.png
    lossplot.png
    poiplot.png
  drafts/                   # Deprecated or experimental stuff, or previous version
```

---

<!--
## How It Works

1. **Data Cleaning & Preprocessing**
   Used `geojson_generator.R` to clean raw CSV data and convert to GeoJSON.

2. **Sequence Generation & Embedding**
   `embedder.py` generates sequences of trajectory points and trains Word2Vec (`w2v.model` / `track_vectors.txt`) to embed points in a vector space.

3. **Graph Construction**
   `edgebuilder.py` builds a weighted graph from trajectory points. Each node represents a trajectory point; edges represent transitions between points, weighted by frequency and similarity.

4. **Community Detection**
   `infomapper.py` runs Infomap on the trajectory graph and produces `communities.csv` and `infomap_output.txt`. Each node (trajectory point) is assigned a community/module.

5. **Visualization**
   `visualizer.py` and `viewer.R` uses trajectory coordinates and community assignments to create maps and plots:

   * `communities_map.html`: interactive map of communities (**Ignored**)
   * `maps/*.png`: static plots (heatmap, clutter plot, trajectories, etc.)

---

## Outputs

* `track_vectors.txt`, `w2v.model`: embeddings of trajectory points
* `graph_edges.csv`: weighted graph edges
* `communities.csv`: node → module assignments
* `communities_map.html`: interactive community map (**Ignored**)
* `maps/`: PNG plots (by `viewer.R`)

---

## Notes

* The project focuses on **trajectory patterns**, which can later be interpreted in terms of **urban function**.
* Drafts and experiments are in `drafts/` and can be skipped.
* Scripts are modular and can be run independently in the order above.

---
