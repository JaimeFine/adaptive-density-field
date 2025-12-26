from infomap import Infomap
import pandas as pd

edges = pd.read_csv("outputs/graph_edges.csv")

nodes = pd.concat(
    [edges["origin"], edges["destination"]]
).unique()
map_node = {
    node: i for i, node in enumerate(nodes)
}

im = Infomap()
for _, r in edges.iterrows():
    origin = map_node[r["origin"]]
    destination = map_node[r["destination"]]
    weight = float(r["weight"])
    im.addLink(origin, destination, weight)

im.run()

comms = []
for node in im.nodes:
    comms.append({
        'position': nodes[node.node_id],
        'module': node.module_id
    })

pd.DataFrame(comms).to_csv("outputs/communities.csv", index=False)