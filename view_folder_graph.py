import json
import os
from context_analysis import ContextAnalyzer
import config


def build_graph_html(output_path: str = "context_graph.html"):
    analyzer = ContextAnalyzer()
    analyzer.build()
    G = analyzer.graph
    nodes_data = []
    edges_data = []
    for n in G.nodes():
        ntype = G.nodes[n].get("type", "?")
        label = n.split("/")[-1] or "/"
        ext = G.nodes[n].get("ext", "")
        kw = analyzer.file_keywords.get(n.lstrip("/"), [])
        title = f"{label}\nType: {ntype}"
        if kw:
            title += f"\nKeywords: {', '.join(kw[:5])}"
        color = "#87CEEB" if ntype == "dir" else "#90EE90"
        nodes_data.append({
            "id": n,
            "label": label or "/",
            "title": title,
            "color": color,
        })
    for u, v in G.edges():
        edges_data.append({"from": u, "to": v})
    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges_data)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Context Graph — {G.number_of_nodes()} nodes</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ margin: 0; font-family: system-ui, sans-serif; }}
        #main {{ width: 100vw; height: 100vh; }}
        #info {{ position: fixed; top: 10px; left: 10px; background: rgba(255,255,255,0.9);
            padding: 8px 12px; border-radius: 6px; font-size: 13px; z-index: 100; }}
    </style>
</head>
<body>
    <div id="info">Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}</div>
    <div id="main"></div>
    <script>
        const nodes = new vis.DataSet({nodes_json});
        const edges = new vis.DataSet({edges_json});
        const container = document.getElementById('main');
        const options = {{
            nodes: {{ shape: 'box', font: {{ size: 12 }}, margin: 4 }},
            edges: {{ arrows: 'to' }},
            layout: {{ hierarchical: {{ direction: 'UD', sortMethod: 'directed' }} }},
            physics: {{ enabled: false }}
        }};
        const network = new vis.Network(container, {{ nodes, edges }}, options);
    </script>
</body>
</html>"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    abs_path = os.path.abspath(output_path)
    print(f"Graph saved to {abs_path}")
    return abs_path


if __name__ == "__main__":
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "context_graph.html")
    build_graph_html(output)
