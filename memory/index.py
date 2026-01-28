from __future__ import annotations

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Set, Optional, Dict
from functools import lru_cache

try:
    from core.config import config
except ImportError:
    # Allow running as a standalone script by adding repo root to sys.path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from core.config import config  # type: ignore

try:
    from agent.performance import IncrementalIndexer, get_global_cache
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False
    IncrementalIndexer = None  # type: ignore
    get_global_cache = None  # type: ignore

MAX_FILE_BYTES = 1_000_000  # 1MB

# Cache for repository scans
_scan_cache: Dict[str, tuple[List[str], float]] = {}
CACHE_TTL = 300.0  # 5 minutes cache TTL

# Global incremental indexer
_incremental_indexer: Optional[Any] = None


def scan_repo(root: str, use_cache: bool = True, incremental: bool = True) -> List[str]:
    """
    Return a sorted list of code file paths under root.

    Rules:
    1. Try using `git ls-files` to respect .gitignore if possible.
    2. Fallback to os.walk with denylisted directories (config.denylist_paths).
    3. Filter by allowed extensions (config.allowed_exts).
    4. Skip files larger than 1MB.
    5. Results are cached for 5 minutes to improve performance.
    6. Uses incremental indexing if available.

    Args:
        root: Repository root directory.
        use_cache: Whether to use cached results if available.
        incremental: Whether to use incremental indexing.

    Returns:
        Sorted list of relative file paths.
    """
    root_path = Path(root).resolve()
    root_str = str(root_path)
    
    # Check cache
    if use_cache and root_str in _scan_cache:
        cached_files, cached_time = _scan_cache[root_str]
        if time.time() - cached_time < CACHE_TTL:
            return cached_files.copy()
    
    allowed_exts = set(config.allowed_exts)
    
    # Try git first
    files = _scan_with_git(root_path)
    if files is None:
        # Fallback to manual walk
        files = _scan_with_walk(root_path, set(config.denylist_paths))
    
    # Filter by extension and size
    results: List[str] = []
    for rel_path in files:
        path = root_path / rel_path
        if path.suffix not in allowed_exts:
            continue
        try:
            if not path.is_file():
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        results.append(rel_path)
    
    sorted_results = sorted(results)
    
    # Update cache
    if use_cache:
        _scan_cache[root_str] = (sorted_results, time.time())
    
    return sorted_results


def clear_scan_cache() -> None:
    """Clear the repository scan cache."""
    global _scan_cache
    _scan_cache.clear()


def _scan_with_git(root: Path) -> List[str] | None:
    """Return list of relative paths using git, or None if failed."""
    if not (root / ".git").exists():
        return None
    
    try:
        # Get tracked files
        tracked = subprocess.run(
            ["git", "ls-files", "-c"], 
            cwd=str(root), capture_output=True, text=True, check=True
        ).stdout.splitlines()
        
        # Get untracked but not ignored files
        untracked = subprocess.run(
            ["git", "ls-files", "-o", "--exclude-standard"], 
            cwd=str(root), capture_output=True, text=True, check=True
        ).stdout.splitlines()
        
        return list(set(tracked + untracked))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _scan_with_walk(root: Path, deny_dirs: Set[str]) -> List[str]:
    """Fallback directory traversal."""
    results: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Remove denylisted directories from traversal
        dirnames[:] = [d for d in dirnames if d not in deny_dirs]

        for filename in filenames:
            path = Path(dirpath) / filename
            results.append(path.relative_to(root).as_posix())
    return results


if __name__ == "__main__":
    files = scan_repo(str(config.repo_root))
    print(f"Total files: {len(files)}")
    for file_path in files[:20]:
        print(file_path)
