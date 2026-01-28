"""Performance optimizations - Lazy loading, caching, and incremental processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable
from collections import OrderedDict
from datetime import datetime, timedelta
import hashlib
import json
import time


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class LRUCache:
    """LRU cache with size limit and TTL."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = None):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries.
            ttl_seconds: Optional time-to-live in seconds.
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check TTL
        if self.ttl_seconds:
            age = (datetime.now() - entry.timestamp).total_seconds()
            if age > self.ttl_seconds:
                del self.cache[key]
                return None
        
        # Update access info
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        
        return entry.value
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self.cache.popitem(last=False)  # Remove oldest
        
        # Create or update entry
        if key in self.cache:
            entry = self.cache[key]
            entry.value = value
            entry.timestamp = datetime.now()
            self.cache.move_to_end(key)
        else:
            self.cache[key] = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
            )
            self.cache.move_to_end(key)
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {
                "size": 0,
                "max_size": self.max_size,
                "hit_rate": 0.0,
            }
        
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_accesses": total_accesses,
            "avg_accesses": total_accesses / len(self.cache) if self.cache else 0,
        }


class IncrementalIndexer:
    """Incremental indexing - only process changed files."""
    
    def __init__(self, repo_root: Path, index_file: Optional[Path] = None):
        """
        Initialize incremental indexer.
        
        Args:
            repo_root: Repository root directory.
            index_file: Optional path to store index metadata.
        """
        self.repo_root = Path(repo_root).resolve()
        self.index_file = index_file or (self.repo_root / ".lca" / "index.json")
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self.file_hashes: Dict[str, str] = {}
        self.file_timestamps: Dict[str, float] = {}
        self._load_index()
    
    def _load_index(self) -> None:
        """Load index from file."""
        if self.index_file.exists():
            try:
                with self.index_file.open("r") as f:
                    data = json.load(f)
                    self.file_hashes = data.get("hashes", {})
                    self.file_timestamps = data.get("timestamps", {})
            except Exception:
                self.file_hashes = {}
                self.file_timestamps = {}
    
    def _save_index(self) -> None:
        """Save index to file."""
        try:
            data = {
                "hashes": self.file_hashes,
                "timestamps": self.file_timestamps,
            }
            with self.index_file.open("w") as f:
                json.dump(data, f)
        except Exception:
            pass
    
    def _file_hash(self, file_path: Path) -> str:
        """Calculate hash of file."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return ""
    
    def get_changed_files(self, files: List[Path]) -> List[Path]:
        """
        Get list of files that have changed since last index.
        
        Args:
            files: List of files to check.
        
        Returns:
            List of changed files.
        """
        changed: List[Path] = []
        
        for file_path in files:
            rel_path = str(file_path.relative_to(self.repo_root))
            
            # Check if file exists
            if not file_path.exists():
                if rel_path in self.file_hashes:
                    # File was deleted
                    changed.append(file_path)
                    del self.file_hashes[rel_path]
                    if rel_path in self.file_timestamps:
                        del self.file_timestamps[rel_path]
                continue
            
            # Get current hash and timestamp
            current_hash = self._file_hash(file_path)
            current_timestamp = file_path.stat().st_mtime
            
            # Check if changed
            if rel_path not in self.file_hashes:
                # New file
                changed.append(file_path)
            elif self.file_hashes[rel_path] != current_hash:
                # File changed
                changed.append(file_path)
            elif rel_path in self.file_timestamps:
                if self.file_timestamps[rel_path] < current_timestamp:
                    # Timestamp changed (might be a change)
                    changed.append(file_path)
            
            # Update index
            self.file_hashes[rel_path] = current_hash
            self.file_timestamps[rel_path] = current_timestamp
        
        # Save index
        self._save_index()
        
        return changed
    
    def mark_processed(self, file_path: Path) -> None:
        """Mark file as processed."""
        rel_path = str(file_path.relative_to(self.repo_root))
        if file_path.exists():
            self.file_hashes[rel_path] = self._file_hash(file_path)
            self.file_timestamps[rel_path] = file_path.stat().st_mtime
            self._save_index()


class LazyFileLoader:
    """Lazy file loading - only load files when needed."""
    
    def __init__(self, repo_root: Path, cache: Optional[LRUCache] = None):
        """
        Initialize lazy file loader.
        
        Args:
            repo_root: Repository root directory.
            cache: Optional cache for file contents.
        """
        self.repo_root = Path(repo_root).resolve()
        self.cache = cache or LRUCache(max_size=500)
        self.loaded_files: Set[str] = set()
    
    def load_file(self, file_path: str, force: bool = False) -> Optional[str]:
        """
        Load file content (with caching).
        
        Args:
            file_path: Relative file path.
            force: Force reload even if cached.
        
        Returns:
            File content or None if not found.
        """
        # Check cache first
        if not force:
            cached = self.cache.get(file_path)
            if cached is not None:
                return cached
        
        # Load file
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return None
        
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            self.cache.set(file_path, content)
            self.loaded_files.add(file_path)
            return content
        except Exception:
            return None
    
    def load_files(self, file_paths: List[str], max_files: Optional[int] = None) -> Dict[str, str]:
        """
        Load multiple files lazily.
        
        Args:
            file_paths: List of file paths.
            max_files: Optional maximum files to load.
        
        Returns:
            Dictionary of file_path -> content.
        """
        if max_files:
            file_paths = file_paths[:max_files]
        
        result: Dict[str, str] = {}
        for file_path in file_paths:
            content = self.load_file(file_path)
            if content is not None:
                result[file_path] = content
        
        return result
    
    def clear_cache(self) -> None:
        """Clear file cache."""
        self.cache.clear()
        self.loaded_files.clear()


class ContextPruner:
    """Prune context based on relevance."""
    
    def __init__(self, max_size: int = 50000):
        """
        Initialize context pruner.
        
        Args:
            max_size: Maximum context size in bytes.
        """
        self.max_size = max_size
    
    def prune(
        self,
        snippets: List[Dict[str, Any]],
        max_bytes: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prune context snippets to fit size limit.
        
        Args:
            snippets: List of context snippets.
            max_bytes: Maximum bytes (uses self.max_size if None).
        
        Returns:
            Pruned list of snippets.
        """
        max_bytes = max_bytes or self.max_size
        
        # Calculate sizes
        snippet_sizes = []
        total_size = 0
        
        for snippet in snippets:
            content = snippet.get("snippet", "")
            size = len(content.encode("utf-8"))
            snippet_sizes.append((snippet, size))
            total_size += size
        
        # If fits, return as-is
        if total_size <= max_bytes:
            return snippets
        
        # Sort by relevance score if available
        snippet_sizes.sort(key=lambda x: x[0].get("score", 0.0), reverse=True)
        
        # Select snippets that fit
        pruned: List[Dict[str, Any]] = []
        current_size = 0
        
        for snippet, size in snippet_sizes:
            if current_size + size <= max_bytes:
                pruned.append(snippet)
                current_size += size
            else:
                # Try to truncate snippet
                remaining_bytes = max_bytes - current_size
                if remaining_bytes > 100:  # Only if meaningful space
                    truncated_snippet = snippet.copy()
                    content = snippet.get("snippet", "")
                    truncated_content = content[:remaining_bytes]
                    truncated_snippet["snippet"] = truncated_content + "\n... (truncated)"
                    pruned.append(truncated_snippet)
                break
        
        return pruned


class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}
    
    def start(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        del self.start_times[operation]
        
        return duration
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        stats: Dict[str, Dict[str, float]] = {}
        
        for operation, durations in self.metrics.items():
            if durations:
                stats[operation] = {
                    "count": len(durations),
                    "total": sum(durations),
                    "avg": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                }
        
        return stats
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.start_times.clear()


# Global instances
_global_cache: Optional[LRUCache] = None
_global_monitor: Optional[PerformanceMonitor] = None


def get_global_cache() -> LRUCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = LRUCache(max_size=1000, ttl_seconds=3600)
    return _global_cache


def get_global_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


if __name__ == "__main__":
    # Demo
    cache = LRUCache(max_size=10)
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    
    print(f"Get key1: {cache.get('key1')}")
    print(f"Stats: {cache.stats()}")
    
    monitor = PerformanceMonitor()
    monitor.start("test")
    time.sleep(0.1)
    duration = monitor.end("test")
    print(f"Test took {duration:.3f}s")
    print(f"Stats: {monitor.get_stats()}")
