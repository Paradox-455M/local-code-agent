"""File watching for incremental editing and context updates."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Set
from dataclasses import dataclass

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None


@dataclass
class FileChangeEvent:
    """Represents a file change event."""
    
    path: Path
    event_type: str  # 'created', 'modified', 'deleted'
    timestamp: float


class RepositoryFileHandler(FileSystemEventHandler):
    """Handler for file system events in the repository."""
    
    def __init__(self, callback: Callable[[FileChangeEvent], None], repo_root: Path):
        self.callback = callback
        self.repo_root = repo_root
        self.ignored_patterns = {
            '.git', '__pycache__', '.pytest_cache', '.mypy_cache',
            'node_modules', '.venv', 'venv', '.lca', 'logs'
        }
        self.last_modified: dict[str, float] = {}
    
    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        parts = path.parts
        for part in parts:
            if part.startswith('.') and part != '.':
                if part in self.ignored_patterns:
                    return True
            if part in self.ignored_patterns:
                return True
        return False
    
    def _handle_event(self, event_path: str, event_type: str) -> None:
        """Handle a file system event."""
        path = Path(event_path)
        
        # Normalize path relative to repo root
        try:
            rel_path = path.relative_to(self.repo_root)
        except ValueError:
            # Path is outside repo root
            return
        
        if self._should_ignore(path):
            return
        
        # Debounce rapid changes (within 0.5 seconds)
        path_str = str(rel_path)
        now = time.time()
        if path_str in self.last_modified:
            if now - self.last_modified[path_str] < 0.5:
                return
        
        self.last_modified[path_str] = now
        
        # Only watch Python files and common config files
        if path.suffix not in ('.py', '.md', '.txt', '.json', '.yaml', '.yml', '.toml'):
            return
        
        change_event = FileChangeEvent(
            path=rel_path,
            event_type=event_type,
            timestamp=now
        )
        
        self.callback(change_event)
    
    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification."""
        if not event.is_directory:
            self._handle_event(event.src_path, 'modified')
    
    def on_created(self, event: FileCreatedEvent) -> None:
        """Handle file creation."""
        if not event.is_directory:
            self._handle_event(event.src_path, 'created')
    
    def on_deleted(self, event: FileDeletedEvent) -> None:
        """Handle file deletion."""
        if not event.is_directory:
            self._handle_event(event.src_path, 'deleted')


class FileWatcher:
    """Watches repository files for changes."""
    
    def __init__(self, repo_root: Path):
        """
        Initialize file watcher.
        
        Args:
            repo_root: Root directory of the repository to watch.
        """
        if not WATCHDOG_AVAILABLE:
            raise ImportError(
                "watchdog library is required for file watching. "
                "Install it with: pip install watchdog"
            )
        
        self.repo_root = Path(repo_root).resolve()
        self.observer: Optional[Observer] = None
        self.handler: Optional[RepositoryFileHandler] = None
        self.callbacks: list[Callable[[FileChangeEvent], None]] = []
        self.is_watching = False
    
    def add_callback(self, callback: Callable[[FileChangeEvent], None]) -> None:
        """
        Add a callback to be called when files change.
        
        Args:
            callback: Function to call with FileChangeEvent when files change.
        """
        self.callbacks.append(callback)
    
    def _on_file_change(self, event: FileChangeEvent) -> None:
        """Internal handler that calls all registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception:
                # Don't let one callback failure stop others
                pass
    
    def start(self) -> None:
        """Start watching files."""
        if self.is_watching:
            return
        
        if not self.repo_root.exists():
            raise ValueError(f"Repository root does not exist: {self.repo_root}")
        
        self.handler = RepositoryFileHandler(self._on_file_change, self.repo_root)
        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.repo_root), recursive=True)
        self.observer.start()
        self.is_watching = True
    
    def stop(self) -> None:
        """Stop watching files."""
        if not self.is_watching:
            return
        
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=1.0)
            self.observer = None
        
        self.is_watching = False
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def watch_repository(
    repo_root: Path,
    callback: Callable[[FileChangeEvent], None],
    timeout: Optional[float] = None
) -> None:
    """
    Watch repository files and call callback on changes.
    
    Args:
        repo_root: Root directory to watch.
        callback: Function to call when files change.
        timeout: Optional timeout in seconds. If None, watches indefinitely.
    """
    watcher = FileWatcher(repo_root)
    watcher.add_callback(callback)
    
    try:
        watcher.start()
        
        if timeout:
            time.sleep(timeout)
        else:
            # Watch indefinitely
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    finally:
        watcher.stop()


if __name__ == "__main__":
    # Demo
    import sys
    
    def on_change(event: FileChangeEvent):
        print(f"[{event.event_type}] {event.path}")
    
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    
    print(f"Watching {repo_root}... (Press Ctrl+C to stop)")
    watch_repository(repo_root, on_change)
