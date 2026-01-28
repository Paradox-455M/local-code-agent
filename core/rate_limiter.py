"""Rate limiting for LLM API calls."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Optional


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    max_requests: int = 10  # Maximum requests per window
    window_seconds: float = 60.0  # Time window in seconds
    min_interval: float = 0.1  # Minimum interval between requests (seconds)


class RateLimiter:
    """Thread-safe rate limiter for API calls."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration. Uses defaults if None.
        """
        self.config = config or RateLimitConfig()
        self._request_times: deque[float] = deque()
        self._last_request_time: float = 0.0
        self._lock = Lock()
    
    def wait_if_needed(self) -> None:
        """
        Wait if necessary to respect rate limits.
        
        Raises:
            RuntimeError: If rate limit would be exceeded.
        """
        with self._lock:
            now = time.time()
            
            # Enforce minimum interval between requests
            time_since_last = now - self._last_request_time
            if time_since_last < self.config.min_interval:
                sleep_time = self.config.min_interval - time_since_last
                time.sleep(sleep_time)
                now = time.time()
            
            # Remove old requests outside the window
            cutoff_time = now - self.config.window_seconds
            while self._request_times and self._request_times[0] < cutoff_time:
                self._request_times.popleft()
            
            # Check if we're at the limit
            if len(self._request_times) >= self.config.max_requests:
                # Calculate wait time until oldest request expires
                oldest_time = self._request_times[0]
                wait_time = (oldest_time + self.config.window_seconds) - now
                if wait_time > 0:
                    raise RuntimeError(
                        f"Rate limit exceeded: {len(self._request_times)} requests in "
                        f"{self.config.window_seconds}s. Wait {wait_time:.1f}s"
                    )
            
            # Record this request
            self._request_times.append(now)
            self._last_request_time = now
    
    def record_request(self) -> None:
        """Record a request (call after successful request)."""
        with self._lock:
            now = time.time()
            self._request_times.append(now)
            self._last_request_time = now
            
            # Clean up old requests
            cutoff_time = now - self.config.window_seconds
            while self._request_times and self._request_times[0] < cutoff_time:
                self._request_times.popleft()
    
    def get_stats(self) -> dict:
        """
        Get current rate limit statistics.
        
        Returns:
            Dictionary with stats.
        """
        with self._lock:
            now = time.time()
            cutoff_time = now - self.config.window_seconds
            
            # Clean up old requests
            while self._request_times and self._request_times[0] < cutoff_time:
                self._request_times.popleft()
            
            return {
                "requests_in_window": len(self._request_times),
                "max_requests": self.config.max_requests,
                "window_seconds": self.config.window_seconds,
                "time_until_reset": (
                    (self._request_times[0] + self.config.window_seconds) - now
                    if self._request_times else 0.0
                ),
            }


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter


def set_rate_limiter(limiter: RateLimiter) -> None:
    """Set the global rate limiter instance."""
    global _global_rate_limiter
    _global_rate_limiter = limiter


if __name__ == "__main__":
    # Demo
    limiter = RateLimiter(RateLimitConfig(max_requests=3, window_seconds=5.0))
    
    print("Making 3 requests (should succeed)...")
    for i in range(3):
        limiter.wait_if_needed()
        limiter.record_request()
        print(f"  Request {i+1} OK")
    
    print("\nMaking 4th request (should fail)...")
    try:
        limiter.wait_if_needed()
        print("  Request 4 OK (unexpected)")
    except RuntimeError as e:
        print(f"  Rate limited: {e}")
    
    print(f"\nStats: {limiter.get_stats()}")
