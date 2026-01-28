"""Persistent summary cache for code context."""

from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class SummaryEntry:
    path: str
    content_hash: str
    mtime: float
    structure: Optional[str] = None
    llm: Optional[str] = None
    updated_at: float = 0.0


class SummaryCache:
    """Cache for per-file summaries stored in .lca/summary.json."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.cache_path = self.repo_root / ".lca" / "summary.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = {"version": 1, "entries": {}}
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                with self.cache_path.open("r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {"version": 1, "entries": {}}

    def _save(self) -> None:
        try:
            with self.cache_path.open("w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            pass

    def compute_hash(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()

    def get(self, file_path: str, content_hash: str) -> Optional[SummaryEntry]:
        entries = self._data.get("entries", {})
        entry = entries.get(file_path)
        if not entry:
            return None
        if entry.get("hash") != content_hash:
            return None
        return SummaryEntry(
            path=file_path,
            content_hash=entry.get("hash", ""),
            mtime=entry.get("mtime", 0.0),
            structure=entry.get("structure"),
            llm=entry.get("llm"),
            updated_at=entry.get("updated_at", 0.0),
        )

    def set(
        self,
        file_path: str,
        content_hash: str,
        mtime: float,
        structure: Optional[str] = None,
        llm: Optional[str] = None,
    ) -> None:
        entries = self._data.setdefault("entries", {})
        entry = entries.get(file_path, {})
        entry["hash"] = content_hash
        entry["mtime"] = mtime
        entry["updated_at"] = time.time()
        if structure is not None:
            entry["structure"] = structure
        if llm is not None:
            entry["llm"] = llm
        entries[file_path] = entry
        self._save()
