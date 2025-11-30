# Trajectory-Based Urban Community Detection Project

## Project Goal

Use aircraft trajectory data around Chengdu to detect spatial communities of flight trajectories, analyze patterns, and visualize them on maps.

-----

## Structure
```txt
project/
  data/                     # Raw and processed data
    cleaned.csv
    chengdu.geo.json
    flight_routes.geojson
    shuangliu.csv
  pyscripts/                # Python scripts
    edgebuilder.py          # Build graph edges from trajectory points
    embedder.py             # Train Word2Vec on trajectory sequences
    infomapper.py           # Run Infomap community detection
    visualizer.py           # Visualize trajectories and communities
  rscripts/                 # R scripts for data cleaning or plotting
    cleaner.R
    turner.R
    viewer.R
  outputs/                  # All generated outputs
    w2v.model
    track_vectors.txt
    graph_edges.csv
    communities.csv
    communities_map.html    # Ignored due to file size
    infomap_output.txt
    maps/                   # PNG plots: heatmaps, clutterplots, trajectories
      clutterplot.png
      conciseplot.png
      heatmap.png
      trajectory.png
      chengdu.png
  drafts/                   # Deprecated or experimental code/outputs
  notebooks/                # Optional Jupyter notebooks (currently empty)
```

---

## How It Works

1. **Data Cleaning**
   Used `cleaner.R` and `turner.R` to clean raw CSV data and convert to GeoJSON.

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
