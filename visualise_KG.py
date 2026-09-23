import pipmaster as pm

if not pm.is_installed("pyvis"):
    pm.install("pyvis")
if not pm.is_installed("networkx"):
    pm.install("networkx")

import networkx as nx
from pyvis.network import Network
import random

G = nx.read_graphml("./lightrag_storage/graph_chunk_entity_relation.graphml")

net = Network(height="100vh", notebook=True)

net.from_nx(G)

for node in net.nodes:
    node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    if "description" in node:
        node["title"] = node["description"]

for edge in net.edges:
    if "description" in edge:
        edge["title"] = edge["description"]

net.show("knowledge_graph.html")
