"""
Rolling Window Rate Limiter
============================
Thread-safe rate limiter using sliding window algorithm.
Ensures all API calls respect rate limits across the entire application.
"""

import time
import threading
from collections import deque
import logging

logger = logging.getLogger(__name__)


class RollingWindowRateLimiter:
    """
    Thread-safe rolling window rate limiter.
    
    Instead of fixed intervals, this tracks timestamps of recent calls
    and ensures we never exceed max_calls within any window_seconds period.
    
    Example:
        limiter = RollingWindowRateLimiter(max_calls=2, window_seconds=60)
        limiter.wait()  # Blocks if necessary to respect rate limit
        # ... make API call ...
    """
    
    def __init__(self, max_calls: int, window_seconds: int, name: str = ""):
        """
        Initialize the rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the window
            window_seconds: Size of the sliding window in seconds
            name: Optional name for logging
        """
        self.max_calls = max_calls
        self.window = window_seconds
        self.name = name or "limiter"
        self.calls = deque()
        self.lock = threading.Lock()
        
        logger.debug(f"Rate limiter '{self.name}' initialized: {max_calls} calls per {window_seconds}s")
    
    def wait(self) -> float:
        """
        Wait until it's safe to make a call, then record the call.
        
        Returns:
            The time spent waiting (0 if no wait was needed)
        """
        total_wait = 0.0
        
        while True:
            with self.lock:
                now = time.time()
                
                # Remove expired timestamps (older than window)
                while self.calls and now - self.calls[0] > self.window:
                    self.calls.popleft()
                
                # Check if we can make a call
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    if total_wait > 0:
                        logger.debug(f"[{self.name}] Waited {total_wait:.1f}s for rate limit")
                    return total_wait
                
                # Calculate how long to wait
                oldest_call = self.calls[0]
                sleep_time = self.window - (now - oldest_call) + 0.1  # Small buffer
            
            if sleep_time > 0:
                logger.info(f"[{self.name}] Rate limit reached, waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                total_wait += sleep_time
    
    def can_call_now(self) -> bool:
        """Check if we can make a call right now without waiting."""
        with self.lock:
            now = time.time()
            
            # Remove expired timestamps
            while self.calls and now - self.calls[0] > self.window:
                self.calls.popleft()
            
            return len(self.calls) < self.max_calls
    
    def time_until_available(self) -> float:
        """Get the time in seconds until the next call can be made."""
        with self.lock:
            now = time.time()
            
            # Remove expired timestamps
            while self.calls and now - self.calls[0] > self.window:
                self.calls.popleft()
            
            if len(self.calls) < self.max_calls:
                return 0.0
            
            oldest_call = self.calls[0]
            return max(0, self.window - (now - oldest_call))
    
    @property
    def current_usage(self) -> int:
        """Get the current number of calls in the window."""
        with self.lock:
            now = time.time()
            while self.calls and now - self.calls[0] > self.window:
                self.calls.popleft()
            return len(self.calls)
    
    def reset(self):
        """Reset the limiter (clear all recorded calls)."""
        with self.lock:
            self.calls.clear()
