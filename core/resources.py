"""Resource monitoring and management."""

from __future__ import annotations

import gc
import os
import psutil
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ResourceStats:
    """Resource usage statistics."""
    
    memory_mb: float
    cpu_percent: float
    disk_usage_mb: float
    open_files: int


class ResourceMonitor:
    """Monitor system resource usage."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self._temp_files: List[Path] = []
    
    def get_stats(self) -> ResourceStats:
        """
        Get current resource usage statistics.
        
        Returns:
            ResourceStats object with current usage.
        """
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        cpu_percent = self.process.cpu_percent(interval=0.1)
        
        # Get disk usage for current directory
        disk_usage = shutil.disk_usage(Path.cwd())
        disk_usage_mb = disk_usage.used / (1024 * 1024)
        
        open_files = len(self.process.open_files())
        
        return ResourceStats(
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            disk_usage_mb=disk_usage_mb,
            open_files=open_files,
        )
    
    def register_temp_file(self, path: Path) -> None:
        """Register a temporary file for cleanup."""
        self._temp_files.append(path)
    
    def cleanup_temp_files(self) -> int:
        """
        Clean up registered temporary files.
        
        Returns:
            Number of files cleaned up.
        """
        cleaned = 0
        for path in self._temp_files[:]:
            try:
                if path.exists():
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
                    cleaned += 1
                self._temp_files.remove(path)
            except Exception:
                pass
        return cleaned
    
    def force_gc(self) -> None:
        """Force garbage collection."""
        gc.collect()
    
    def check_memory_limit(self, limit_mb: float = 1024.0) -> tuple[bool, float]:
        """
        Check if memory usage exceeds limit.
        
        Args:
            limit_mb: Memory limit in MB.
        
        Returns:
            Tuple of (exceeds_limit, current_mb).
        """
        stats = self.get_stats()
        exceeds = stats.memory_mb > limit_mb
        return exceeds, stats.memory_mb


# Global resource monitor instance
_global_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor() -> ResourceMonitor:
    """Get the global resource monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ResourceMonitor()
    return _global_monitor


if __name__ == "__main__":
    # Demo
    monitor = ResourceMonitor()
    stats = monitor.get_stats()
    print(f"Memory: {stats.memory_mb:.1f} MB")
    print(f"CPU: {stats.cpu_percent:.1f}%")
    print(f"Disk: {stats.disk_usage_mb:.1f} MB")
    print(f"Open files: {stats.open_files}")
    
    # Test temp file cleanup
    temp_file = Path(tempfile.mktemp())
    temp_file.write_text("test")
    monitor.register_temp_file(temp_file)
    print(f"\nCleaned up {monitor.cleanup_temp_files()} temp files")
