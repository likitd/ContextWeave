import os
from pathlib import Path
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import config
from document_loader import load_file_as_text, discover_files


class ContextAnalyzer:
    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.file_paths: list[str] = []
        self.file_keywords: dict[str, list[str]] = {}

    def build(self):
        self._build_directory_graph()
        self._build_tfidf_index()
        print(f"ContextAnalyzer ready — {len(self.file_paths)} files indexed, "
              f"{self.graph.number_of_nodes()} graph nodes")

    def _build_directory_graph(self):
        root = config.CONTEXT_FOLDER
        self.graph.clear()
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root)
            if rel_dir == ".":
                rel_dir = "/"
            else:
                rel_dir = "/" + rel_dir.replace("\\", "/")
            self.graph.add_node(rel_dir, type="dir")
            for d in sorted(dirnames):
                child = rel_dir.rstrip("/") + "/" + d
                self.graph.add_node(child, type="dir")
                self.graph.add_edge(rel_dir, child)
            for f in sorted(filenames):
                ext = Path(f).suffix.lower()
                if ext not in config.SUPPORTED_EXTENSIONS:
                    continue
                child = rel_dir.rstrip("/") + "/" + f
                self.graph.add_node(child, type="file", ext=ext)
                self.graph.add_edge(rel_dir, child)

    def _build_tfidf_index(self):
        self.file_paths = discover_files(config.CONTEXT_FOLDER)
        if not self.file_paths:
            return
        corpus = []
        valid_paths = []
        for fp in self.file_paths:
            text = load_file_as_text(fp)
            if text and text.strip():
                corpus.append(text)
                valid_paths.append(fp)
        self.file_paths = valid_paths
        if not corpus:
            return
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=200,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        for i, fp in enumerate(self.file_paths):
            rel = os.path.relpath(fp, config.CONTEXT_FOLDER)
            row = self.tfidf_matrix[i].toarray().flatten()
            top_indices = row.argsort()[-10:][::-1]
            top_kw = [feature_names[j] for j in top_indices if row[j] > 0]
            self.file_keywords[rel] = top_kw

    def get_graph_overview(self) -> str:
        lines = ["Directory structure:"]
        root = config.CONTEXT_FOLDER

        def _walk(node, indent=0):
            ntype = self.graph.nodes[node].get("type", "?")
            rel_path = node.lstrip("/") or "/"
            label = node.split("/")[-1] or "/"
            prefix = "  " * indent
            if ntype == "dir":
                lines.append(f"{prefix}[DIR] {label}/")
            else:
                kw = self.file_keywords.get(rel_path, self.file_keywords.get(
                    label, []))
                kw_str = f"  (keywords: {', '.join(kw[:5])})" if kw else ""
                lines.append(f"{prefix}[FILE] {rel_path}{kw_str}")
            for child in sorted(self.graph.successors(node)):
                _walk(child, indent + 1)

        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        for r in sorted(roots):
            _walk(r)
        return "\n".join(lines)

    def find_relevant_files(self, query: str, top_k: int = 5) -> list[dict]:
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        query_vec = self.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked[:top_k]:
            rel = os.path.relpath(self.file_paths[idx], config.CONTEXT_FOLDER)
            results.append({
                "file": rel,
                "score": round(float(scores[idx]), 4),
                "keywords": self.file_keywords.get(rel, [])[:5],
            })
        return results

    def get_file_count(self) -> int:
        return len(self.file_paths)
