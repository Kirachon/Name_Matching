"""
Performance optimization implementations for name matching.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import hashlib

class OptimizedNameMatcher:
    """
    High-performance name matcher with blocking and caching.
    """
    
    def __init__(self):
        self.similarity_cache = {}
        self.soundex_cache = {}
        self.component_cache = {}
        
    def create_blocking_index(self, df: pd.DataFrame, blocking_fields: List[str]) -> Dict[str, List[int]]:
        """
        Create blocking index to reduce comparison space.
        
        Args:
            df: DataFrame to index
            blocking_fields: Fields to use for blocking (e.g., ['first_name_first_char', 'soundex_last_name'])
            
        Returns:
            Dictionary mapping block keys to record indices
        """
        blocks = defaultdict(list)
        
        for idx, row in df.iterrows():
            # Create composite blocking key
            block_key_parts = []
            for field in blocking_fields:
                if field == 'first_name_first_char':
                    value = str(row.get('first_name', '')).strip()[:1].upper()
                elif field == 'soundex_last_name':
                    last_name = str(row.get('last_name', '')).strip()
                    value = self._fast_soundex(last_name)
                elif field == 'year_of_birth':
                    birthdate = str(row.get('birthdate', ''))
                    value = birthdate[:4] if len(birthdate) >= 4 else 'UNKNOWN'
                else:
                    value = str(row.get(field, '')).strip().upper()
                
                block_key_parts.append(value)
            
            block_key = '|'.join(block_key_parts)
            blocks[block_key].append(idx)
            
        return dict(blocks)
    
    def _fast_soundex(self, name: str) -> str:
        """Optimized Soundex implementation with caching."""
        if not name:
            return "0000"
            
        if name in self.soundex_cache:
            return self.soundex_cache[name]
            
        # Fast soundex implementation
        name = name.upper()
        if not name.isalpha():
            result = "0000"
        else:
            first_letter = name[0]
            # Use lookup table instead of multiple if-elif
            soundex_map = str.maketrans('BFPVCGJKQSXZDTLMNR', '111122222222334556')
            coded = name.translate(soundex_map)
            
            # Remove consecutive duplicates and vowels
            result = first_letter
            prev_char = first_letter
            for char in coded[1:]:
                if char.isdigit() and char != prev_char:
                    result += char
                    prev_char = char
                    if len(result) == 4:
                        break
            
            result = (result + "000")[:4]
        
        self.soundex_cache[name] = result
        return result

    def vectorized_jaro_winkler(self, names1: List[str], names2: List[str]) -> np.ndarray:
        """
        Vectorized Jaro-Winkler calculation for batch processing.
        """
        # Pre-allocate result array
        results = np.zeros((len(names1), len(names2)), dtype=np.float32)
        
        # Use jellyfish for batch processing (faster than custom implementation)
        import jellyfish
        
        for i, name1 in enumerate(names1):
            for j, name2 in enumerate(names2):
                # Check cache first
                cache_key = f"{name1}|{name2}"
                if cache_key in self.similarity_cache:
                    results[i, j] = self.similarity_cache[cache_key]
                else:
                    score = jellyfish.jaro_winkler_similarity(name1, name2)
                    results[i, j] = score
                    self.similarity_cache[cache_key] = score
                    
        return results

    def optimized_component_comparison(self, components1: Dict, components2: Dict) -> Dict[str, float]:
        """
        Optimized component comparison with minimal key lookups.
        """
        # Use tuple as cache key for better performance
        cache_key = (
            components1.get('first_name', ''),
            components1.get('middle_name', ''),
            components1.get('last_name', ''),
            components2.get('first_name', ''),
            components2.get('middle_name', ''),
            components2.get('last_name', '')
        )
        
        if cache_key in self.component_cache:
            return self.component_cache[cache_key]
        
        # Pre-extract values to avoid repeated dict lookups
        fn1, mn1, ln1 = cache_key[:3]
        fn2, mn2, ln2 = cache_key[3:]
        
        # Use jellyfish directly (faster than our wrapper)
        import jellyfish
        
        scores = {
            'first_name': jellyfish.jaro_winkler_similarity(fn1, fn2),
            'middle_name': jellyfish.jaro_winkler_similarity(mn1, mn2),
            'last_name': jellyfish.jaro_winkler_similarity(ln1, ln2),
        }
        
        # Optimized full name sorted comparison
        full1 = ' '.join(filter(None, [fn1, mn1, ln1]))
        full2 = ' '.join(filter(None, [fn2, mn2, ln2]))
        
        # Fast token sort without creating intermediate lists
        tokens1 = sorted(full1.split())
        tokens2 = sorted(full2.split())
        sorted1 = ' '.join(tokens1)
        sorted2 = ' '.join(tokens2)
        
        scores['full_name_sorted'] = jellyfish.jaro_winkler_similarity(sorted1, sorted2)
        
        self.component_cache[cache_key] = scores
        return scores

# Optimized data structures
class CompactNameRecord:
    """
    Memory-efficient name record using __slots__ and numeric IDs.
    """
    __slots__ = ['id', 'first_name', 'middle_name', 'last_name', 'birthdate', 'location_id']
    
    def __init__(self, id: int, first_name: str, middle_name: str, last_name: str, 
                 birthdate: str = None, location_id: int = None):
        self.id = id
        self.first_name = first_name
        self.middle_name = middle_name
        self.last_name = last_name
        self.birthdate = birthdate
        self.location_id = location_id

class LocationLookup:
    """
    Efficient location lookup using numeric IDs instead of strings.
    """
    def __init__(self):
        self.location_to_id = {}
        self.id_to_location = {}
        self.next_id = 1
    
    def get_location_id(self, province: str, city: str, barangay: str = None) -> int:
        """Convert location strings to numeric ID."""
        location_key = f"{province}|{city}|{barangay or ''}"
        
        if location_key not in self.location_to_id:
            self.location_to_id[location_key] = self.next_id
            self.id_to_location[self.next_id] = (province, city, barangay)
            self.next_id += 1
            
        return self.location_to_id[location_key]

# Parallel processing utilities
def parallel_name_matching(df1: pd.DataFrame, df2: pd.DataFrame, 
                          num_processes: int = 4) -> pd.DataFrame:
    """
    Parallel name matching using multiprocessing.
    """
    import multiprocessing as mp
    from functools import partial
    
    # Split df1 into chunks
    chunk_size = len(df1) // num_processes
    chunks = [df1[i:i + chunk_size] for i in range(0, len(df1), chunk_size)]
    
    # Create worker function
    def match_chunk(chunk_df1, df2_data):
        matcher = OptimizedNameMatcher()
        results = []
        
        for _, row1 in chunk_df1.iterrows():
            for _, row2 in df2_data.iterrows():
                # Perform matching logic here
                pass
                
        return results
    
    # Use multiprocessing
    with mp.Pool(num_processes) as pool:
        worker_func = partial(match_chunk, df2_data=df2)
        chunk_results = pool.map(worker_func, chunks)
    
    # Combine results
    all_results = []
    for chunk_result in chunk_results:
        all_results.extend(chunk_result)
    
    return pd.DataFrame(all_results)

# Memory-efficient batch processing
class BatchProcessor:
    """
    Process large datasets in memory-efficient batches.
    """
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        
    def process_large_dataset(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Process large datasets in batches to control memory usage.
        """
        results = []
        matcher = OptimizedNameMatcher()
        
        # Process df1 in batches
        for i in range(0, len(df1), self.batch_size):
            batch1 = df1[i:i + self.batch_size]
            
            # For each batch of df1, process against all of df2 in batches
            for j in range(0, len(df2), self.batch_size):
                batch2 = df2[j:j + self.batch_size]
                
                # Process this batch combination
                batch_results = self._process_batch_pair(batch1, batch2, matcher)
                results.extend(batch_results)
                
                # Optional: Clear caches periodically to prevent memory growth
                if len(matcher.similarity_cache) > 10000:
                    matcher.similarity_cache.clear()
                    matcher.component_cache.clear()
        
        return pd.DataFrame(results)
    
    def _process_batch_pair(self, batch1: pd.DataFrame, batch2: pd.DataFrame, 
                           matcher: OptimizedNameMatcher) -> List[Dict]:
        """Process a pair of batches."""
        results = []
        
        # Create blocking indices for this batch
        blocks1 = matcher.create_blocking_index(batch1, ['first_name_first_char', 'soundex_last_name'])
        blocks2 = matcher.create_blocking_index(batch2, ['first_name_first_char', 'soundex_last_name'])
        
        # Only compare records in matching blocks
        for block_key in blocks1:
            if block_key in blocks2:
                indices1 = blocks1[block_key]
                indices2 = blocks2[block_key]
                
                for idx1 in indices1:
                    for idx2 in indices2:
                        row1 = batch1.iloc[idx1]
                        row2 = batch2.iloc[idx2]
                        
                        # Perform comparison
                        components1 = {
                            'first_name': str(row1.get('first_name', '')),
                            'middle_name': str(row1.get('middle_name', '')),
                            'last_name': str(row1.get('last_name', ''))
                        }
                        components2 = {
                            'first_name': str(row2.get('first_name', '')),
                            'middle_name': str(row2.get('middle_name', '')),
                            'last_name': str(row2.get('last_name', ''))
                        }
                        
                        scores = matcher.optimized_component_comparison(components1, components2)
                        
                        # Calculate overall score (simplified)
                        overall_score = (
                            scores['first_name'] * 0.4 +
                            scores['middle_name'] * 0.2 +
                            scores['last_name'] * 0.3 +
                            scores['full_name_sorted'] * 0.1
                        )
                        
                        if overall_score > 0.55:  # Only keep potential matches
                            results.append({
                                'id1': row1.get('hh_id'),
                                'id2': row2.get('hh_id'),
                                'score': overall_score,
                                **{f'score_{k}': v for k, v in scores.items()}
                            })
        
        return results
