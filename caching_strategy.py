"""
Advanced caching strategies for name matching performance.
"""

import hashlib
import pickle
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache, wraps
import time

class SmartCache:
    """
    Multi-level caching system for name matching operations.
    """
    
    def __init__(self, max_memory_cache: int = 10000, enable_disk_cache: bool = False):
        self.memory_cache = {}
        self.max_memory_cache = max_memory_cache
        self.enable_disk_cache = enable_disk_cache
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'memory_size': 0,
            'disk_size': 0
        }
        
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate consistent cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then disk)."""
        # Check memory cache first
        if key in self.memory_cache:
            self.cache_stats['hits'] += 1
            return self.memory_cache[key]
        
        # Check disk cache if enabled
        if self.enable_disk_cache:
            try:
                with open(f"cache/{key}.pkl", 'rb') as f:
                    value = pickle.load(f)
                    # Promote to memory cache
                    self._add_to_memory_cache(key, value)
                    self.cache_stats['hits'] += 1
                    return value
            except FileNotFoundError:
                pass
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        self._add_to_memory_cache(key, value)
        
        if self.enable_disk_cache:
            import os
            os.makedirs("cache", exist_ok=True)
            with open(f"cache/{key}.pkl", 'wb') as f:
                pickle.dump(value, f)
    
    def _add_to_memory_cache(self, key: str, value: Any):
        """Add to memory cache with LRU eviction."""
        if len(self.memory_cache) >= self.max_memory_cache:
            # Remove oldest item (simple FIFO, could be improved to LRU)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = value
        self.cache_stats['memory_size'] = len(self.memory_cache)

def cached_similarity(cache: SmartCache):
    """Decorator for caching similarity function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(s1: str, s2: str, *args, **kwargs):
            # Normalize inputs for consistent caching
            s1_norm = s1.strip().lower()
            s2_norm = s2.strip().lower()
            
            # Create cache key
            cache_key = cache._generate_key(func.__name__, s1_norm, s2_norm, *args, **kwargs)
            
            # Check cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(s1, s2, *args, **kwargs)
            cache.set(cache_key, result)
            return result
        
        return wrapper
    return decorator

class PrecomputedSimilarityMatrix:
    """
    Precompute similarity matrices for common name combinations.
    """
    
    def __init__(self):
        self.first_name_matrix = {}
        self.last_name_matrix = {}
        self.common_names = {
            'first_names': ['Juan', 'Maria', 'Jose', 'Ana', 'Carlos', 'Rosa', 'Antonio', 'Carmen'],
            'last_names': ['Santos', 'Cruz', 'Garcia', 'Lopez', 'Reyes', 'Ramos', 'Mendoza', 'Torres']
        }
        self._precompute_matrices()
    
    def _precompute_matrices(self):
        """Precompute similarity matrices for common names."""
        import jellyfish
        
        # Precompute first name similarities
        first_names = self.common_names['first_names']
        for i, name1 in enumerate(first_names):
            for j, name2 in enumerate(first_names):
                key = (name1.lower(), name2.lower())
                self.first_name_matrix[key] = jellyfish.jaro_winkler_similarity(name1, name2)
        
        # Precompute last name similarities
        last_names = self.common_names['last_names']
        for i, name1 in enumerate(last_names):
            for j, name2 in enumerate(last_names):
                key = (name1.lower(), name2.lower())
                self.last_name_matrix[key] = jellyfish.jaro_winkler_similarity(name1, name2)
    
    def get_similarity(self, name1: str, name2: str, name_type: str) -> Optional[float]:
        """Get precomputed similarity if available."""
        key = (name1.lower().strip(), name2.lower().strip())
        
        if name_type == 'first_name':
            return self.first_name_matrix.get(key)
        elif name_type == 'last_name':
            return self.last_name_matrix.get(key)
        
        return None

class AdaptiveCache:
    """
    Cache that adapts its strategy based on usage patterns.
    """
    
    def __init__(self):
        self.access_counts = {}
        self.last_access_time = {}
        self.cache = {}
        self.max_size = 5000
        
    def get(self, key: str) -> Optional[Any]:
        """Get with access tracking."""
        if key in self.cache:
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.last_access_time[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set with intelligent eviction."""
        if len(self.cache) >= self.max_size:
            self._evict_least_valuable()
        
        self.cache[key] = value
        self.access_counts[key] = 1
        self.last_access_time[key] = time.time()
    
    def _evict_least_valuable(self):
        """Evict based on access frequency and recency."""
        current_time = time.time()
        
        # Calculate value score for each item
        scores = {}
        for key in self.cache:
            access_count = self.access_counts.get(key, 1)
            time_since_access = current_time - self.last_access_time.get(key, current_time)
            
            # Higher score = more valuable (more accesses, more recent)
            scores[key] = access_count / (1 + time_since_access / 3600)  # Decay over hours
        
        # Remove lowest scoring item
        least_valuable = min(scores.keys(), key=lambda k: scores[k])
        del self.cache[least_valuable]
        del self.access_counts[least_valuable]
        del self.last_access_time[least_valuable]

# Usage example with optimized matcher
class CachedNameMatcher:
    """
    Name matcher with comprehensive caching.
    """
    
    def __init__(self):
        self.similarity_cache = SmartCache(max_memory_cache=10000, enable_disk_cache=True)
        self.component_cache = AdaptiveCache()
        self.precomputed_matrix = PrecomputedSimilarityMatrix()
        
    @cached_similarity
    def jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """Cached Jaro-Winkler similarity."""
        import jellyfish
        return jellyfish.jaro_winkler_similarity(s1, s2)
    
    def optimized_component_comparison(self, components1: Dict, components2: Dict) -> Dict[str, float]:
        """Component comparison with multi-level caching."""
        # Create cache key
        cache_key = (
            components1.get('first_name', '').lower().strip(),
            components1.get('middle_name', '').lower().strip(),
            components1.get('last_name', '').lower().strip(),
            components2.get('first_name', '').lower().strip(),
            components2.get('middle_name', '').lower().strip(),
            components2.get('last_name', '').lower().strip()
        )
        
        # Check component cache
        cached_result = self.component_cache.get(str(cache_key))
        if cached_result is not None:
            return cached_result
        
        # Extract components
        fn1, mn1, ln1, fn2, mn2, ln2 = cache_key
        
        scores = {}
        
        # Try precomputed matrix first, then cached similarity
        for name_type, (n1, n2) in [('first_name', (fn1, fn2)), ('last_name', (ln1, ln2))]:
            precomputed = self.precomputed_matrix.get_similarity(n1, n2, name_type)
            if precomputed is not None:
                scores[name_type] = precomputed
            else:
                scores[name_type] = self.jaro_winkler_similarity(n1, n2)
        
        # Middle name (not in precomputed matrix)
        scores['middle_name'] = self.jaro_winkler_similarity(mn1, mn2)
        
        # Full name sorted
        full1 = ' '.join(filter(None, [fn1, mn1, ln1]))
        full2 = ' '.join(filter(None, [fn2, mn2, ln2]))
        
        # Fast token sort
        tokens1 = sorted(full1.split())
        tokens2 = sorted(full2.split())
        sorted1 = ' '.join(tokens1)
        sorted2 = ' '.join(tokens2)
        
        scores['full_name_sorted'] = self.jaro_winkler_similarity(sorted1, sorted2)
        
        # Cache the result
        self.component_cache.set(str(cache_key), scores)
        
        return scores
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'similarity_cache': self.similarity_cache.cache_stats,
            'component_cache_size': len(self.component_cache.cache),
            'precomputed_first_names': len(self.precomputed_matrix.first_name_matrix),
            'precomputed_last_names': len(self.precomputed_matrix.last_name_matrix)
        }
