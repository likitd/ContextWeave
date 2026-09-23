import os
import re
import time
import json
from pathlib import Path
from dataclasses import dataclass, field

import config
from document_loader import load_file_as_text


@dataclass
class ToolStat:
    name: str
    total_calls: int = 0
    successful: int = 0
    failed: int = 0
    total_duration_ms: float = 0.0

    @property
    def avg_duration_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_duration_ms / self.total_calls


@dataclass
class ToolCall:
    id: int
    name: str
    parameters: dict
    start_time: float


class AgentFSManager:
    def __init__(self):
        self.agent = None
        self.context_root = config.CONTEXT_FOLDER
        self._kv_store: dict = {}
        self._tool_stats: dict[str, ToolStat] = {}
        self._active_calls: dict[int, ToolCall] = {}
        self._call_counter = 0
        self._data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agentfs_local")

    async def initialize(self):
        os.makedirs(self._data_dir, exist_ok=True)
        kv_path = os.path.join(self._data_dir, "kv_store.json")
        if os.path.exists(kv_path):
            with open(kv_path, "r", encoding="utf-8") as f:
                self._kv_store = json.load(f)
        print(f"AgentFS local manager initialized")
        print(f"Context folder: {self.context_root}")

    async def close(self):
        self._save_kv()

    def _save_kv(self):
        kv_path = os.path.join(self._data_dir, "kv_store.json")
        with open(kv_path, "w", encoding="utf-8") as f:
            json.dump(self._kv_store, f, indent=2, default=str)

    def _safe_path(self, rel_path: str) -> str | None:
        cleaned = rel_path.strip("/").strip("\\")
        full = os.path.normpath(os.path.join(self.context_root, cleaned))
        if not full.startswith(os.path.normpath(self.context_root)):
            return None
        return full

    def list_dir(self, rel_path: str = "") -> list[dict]:
        full = self._safe_path(rel_path)
        if not full or not os.path.isdir(full):
            return []
        entries = []
        for name in sorted(os.listdir(full)):
            child = os.path.join(full, name)
            if os.path.isdir(child):
                entries.append({"name": name, "type": "dir", "size": 0})
            else:
                ext = Path(name).suffix.lower()
                if ext in config.SUPPORTED_EXTENSIONS:
                    entries.append({"name": name, "type": "file", "size": os.path.getsize(child)})
        return entries

    def _read_file(self, rel_path: str) -> str | None:
        full = self._safe_path(rel_path)
        if not full or not os.path.isfile(full):
            return None
        return load_file_as_text(full)

    def search(self, pattern: str, max_results: int = 10) -> list[dict]:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        results = []
        for root, _, files in os.walk(self.context_root):
            for fname in sorted(files):
                ext = Path(fname).suffix.lower()
                if ext not in config.SUPPORTED_EXTENSIONS:
                    continue
                fpath = os.path.join(root, fname)
                content = load_file_as_text(fpath)
                if not content:
                    continue
                matches = []
                for line_num, line in enumerate(content.split("\n"), 1):
                    if regex.search(line):
                        matches.append({"line": line_num, "text": line.strip()})
                        if len(matches) >= 3:
                            break
                if matches:
                    rel = os.path.relpath(fpath, self.context_root)
                    results.append({"file": rel, "matches": matches})
                    if len(results) >= max_results:
                        return results
        return results

    async def ingest_to_agentfs(self, documents):
        docs_dir = os.path.join(self._data_dir, "documents")
        os.makedirs(docs_dir, exist_ok=True)
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            safe_name = source.replace("/", "_").replace("\\", "_")
            with open(os.path.join(docs_dir, safe_name), "w", encoding="utf-8") as f:
                f.write(doc.page_content)
        await self.kv_set("documents:count", len(documents))
        sources = [doc.metadata.get("source", "unknown") for doc in documents]
        await self.kv_set("documents:sources", sources)
        print(f"Ingested {len(documents)} documents into local storage")

    async def tool_start(self, name: str, parameters=None) -> int:
        print(f"Step/Process {name} started...")
        self._call_counter += 1
        call_id = self._call_counter
        self._active_calls[call_id] = ToolCall(
            id=call_id, name=name, parameters=parameters or {}, start_time=time.time()
        )
        if name not in self._tool_stats:
            self._tool_stats[name] = ToolStat(name=name)
        self._tool_stats[name].total_calls += 1
        return call_id

    async def tool_success(self, call_id: int, result=None):
        call = self._active_calls.pop(call_id, None)
        if call:
            elapsed = (time.time() - call.start_time) * 1000
            self._tool_stats[call.name].successful += 1
            self._tool_stats[call.name].total_duration_ms += elapsed

    async def tool_error(self, call_id: int, error: str):
        call = self._active_calls.pop(call_id, None)
        if call:
            elapsed = (time.time() - call.start_time) * 1000
            self._tool_stats[call.name].failed += 1
            self._tool_stats[call.name].total_duration_ms += elapsed

    async def get_stats(self):
        return list(self._tool_stats.values())

    async def kv_set(self, key: str, value):
        self._kv_store[key] = value

    async def kv_get(self, key: str):
        return self._kv_store.get(key)
