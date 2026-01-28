"""Persistent cache for knowledge graph and call graph."""

from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

try:
    from core.config import config
    from memory.index import scan_repo
    from memory.knowledge_graph import CodebaseGraph
    from memory.call_graph import CallGraphBuilder
except ImportError:
    config = None  # type: ignore
    scan_repo = None  # type: ignore
    CodebaseGraph = None  # type: ignore
    CallGraphBuilder = None  # type: ignore


def _repo_fingerprint(repo_root: Path) -> str:
    """Fingerprint Python files by path + mtime."""
    if scan_repo is None:
        return ""
    files = [f for f in scan_repo(str(repo_root)) if f.endswith(".py")]
    parts = []
    for f in files:
        try:
            path = repo_root / f
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        parts.append(f"{f}:{mtime}")
    digest = hashlib.md5("|".join(sorted(parts)).encode("utf-8", errors="replace")).hexdigest()
    return digest


class GraphCache:
    """Cache KG and call graph on disk with TTL and fingerprint checks."""

    def __init__(self, repo_root: Path, ttl_seconds: Optional[int] = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.cache_path = self.repo_root / ".lca" / "graph_cache.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds or (config.graph_cache_ttl if config else 300)

    def load(self) -> Optional[Tuple["CodebaseGraph", "CallGraphBuilder"]]:
        if not self.cache_path.exists():
            return None
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            created_at = data.get("created_at", 0.0)
            if self.ttl_seconds and (time.time() - created_at) > self.ttl_seconds:
                return None
            fingerprint = data.get("fingerprint")
            if fingerprint != _repo_fingerprint(self.repo_root):
                return None

            kg_data = data.get("knowledge_graph")
            cg_data = data.get("call_graph")
            if not kg_data or not cg_data:
                return None
            if CodebaseGraph is None or CallGraphBuilder is None:
                return None

            kg = CodebaseGraph.from_dict(self.repo_root, kg_data)
            cg = CallGraphBuilder.from_dict(self.repo_root, kg, cg_data)
            return kg, cg
        except Exception:
            return None

    def save(self, kg: "CodebaseGraph", cg: "CallGraphBuilder") -> None:
        try:
            data: Dict[str, Any] = {
                "version": 1,
                "created_at": time.time(),
                "fingerprint": _repo_fingerprint(self.repo_root),
                "knowledge_graph": kg.to_dict(),
                "call_graph": cg.to_dict(),
            }
            self.cache_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass
