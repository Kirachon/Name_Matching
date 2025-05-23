"""
Cache manager for coordinating different caching strategies.
"""

import logging
from typing import Optional, Dict, Any
import time

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages multiple caching layers and strategies."""
    
    def __init__(
        self,
        enable_memory: bool = True,
        memory_size: int = 1000
    ):
        """Initialize cache manager.
        
        Args:
            enable_memory: Enable in-memory caching
            memory_size: Maximum size of memory cache
        """
        self.enable_memory = enable_memory
        self.memory_size = memory_size
        
        # Initialize memory cache (simple LRU)
        self.memory_cache = {}
        self.memory_access_order = []
        
        # Statistics
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "total_requests": 0
        }
    
    def _memory_get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get from memory cache."""
        if not self.enable_memory or key not in self.memory_cache:
            self.stats["memory_misses"] += 1
            return None
        
        # Update access order (move to end)
        self.memory_access_order.remove(key)
        self.memory_access_order.append(key)
        
        self.stats["memory_hits"] += 1
        return self.memory_cache[key]
    
    def _memory_set(self, key: str, value: Dict[str, Any]):
        """Set in memory cache."""
        if not self.enable_memory:
            return
        
        # Remove oldest if at capacity
        while len(self.memory_cache) >= self.memory_size:
            oldest_key = self.memory_access_order.pop(0)
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = value
        if key in self.memory_access_order:
            self.memory_access_order.remove(key)
        self.memory_access_order.append(key)
    
    def _generate_cache_key(self, name1: str, name2: str, additional_data: Optional[Dict] = None) -> str:
        """Generate cache key."""
        names = sorted([name1.lower().strip(), name2.lower().strip()])
        key = f"match:{':'.join(names)}"
        if additional_data:
            key += f":{hash(str(sorted(additional_data.items())))}"
        return key
    
    def get(self, name1: str, name2: str, additional_data: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Get cached result."""
        self.stats["total_requests"] += 1
        key = self._generate_cache_key(name1, name2, additional_data)
        
        # Try memory cache
        result = self._memory_get(key)
        if result is not None:
            logger.debug(f"Memory cache hit for {key}")
            return result["result"]
        
        logger.debug(f"Cache miss for {key}")
        return None
    
    def set(
        self,
        name1: str,
        name2: str,
        result: Dict[str, Any],
        additional_data: Optional[Dict] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Set cached result."""
        key = self._generate_cache_key(name1, name2, additional_data)
        
        # Prepare cache data
        cache_data = {
            "result": result,
            "cached_at": time.time(),
            "names": [name1, name2],
            "additional_data": additional_data
        }
        
        # Set in memory cache
        if self.enable_memory:
            self._memory_set(key, cache_data)
            logger.debug(f"Stored in memory cache: {key}")
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.copy()
        
        # Calculate hit rates
        total_memory = stats["memory_hits"] + stats["memory_misses"]
        stats["memory_hit_rate"] = (stats["memory_hits"] / max(total_memory, 1)) * 100
        stats["memory_cache_size"] = len(self.memory_cache)
        stats["memory_cache_max_size"] = self.memory_size
        
        return stats


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(**kwargs) -> CacheManager:
    """Get or create global cache manager."""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager(**kwargs)
    
    return _cache_manager
