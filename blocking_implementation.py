"""
High-priority blocking implementation for immediate performance gains.
"""

from src.name_matcher import NameMatcher
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
import time

class BlockingNameMatcher(NameMatcher):
    """
    Enhanced NameMatcher with blocking for O(n²) → O(n×k) performance improvement.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocking_stats = {'blocks_created': 0, 'comparisons_avoided': 0}

    def match_dataframes_with_blocking(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        blocking_fields: List[str] = None,
        threshold: float = 0.55
    ) -> pd.DataFrame:
        """
        Match DataFrames using blocking strategy for massive performance improvement.

        Args:
            df1, df2: DataFrames to match
            blocking_fields: Fields to use for blocking (default: smart blocking)
            threshold: Minimum score threshold for results

        Returns:
            DataFrame with match results
        """
        if blocking_fields is None:
            blocking_fields = ['first_char_first_name', 'soundex_last_name']

        print(f"Starting blocked matching: {len(df1)} × {len(df2)} = {len(df1) * len(df2):,} potential comparisons")

        # Create blocking indices
        start_time = time.time()
        blocks1 = self._create_blocking_index(df1, blocking_fields)
        blocks2 = self._create_blocking_index(df2, blocking_fields)
        blocking_time = time.time() - start_time

        print(f"Created {len(blocks1)} blocks for df1 and {len(blocks2)} blocks for df2 in {blocking_time:.2f}s")

        # Perform blocked matching
        start_time = time.time()
        results = []
        total_comparisons = 0

        for block_key in blocks1:
            if block_key in blocks2:
                indices1 = blocks1[block_key]
                indices2 = blocks2[block_key]

                # Match within this block
                for idx1 in indices1:
                    for idx2 in indices2:
                        row1 = df1.iloc[idx1]
                        row2 = df2.iloc[idx2]

                        # Perform name matching
                        # Convert row data to proper format
                        name1_data = {
                            'first_name': str(row1.get('first_name', '')),
                            'middle_name_last_name': str(row1.get('middle_name_last_name', '')),
                            'birthdate': str(row1.get('birthdate', '')),
                            'province_name': str(row1.get('province_name', ''))
                        }
                        name2_data = {
                            'first_name': str(row2.get('first_name', '')),
                            'middle_name_last_name': str(row2.get('middle_name_last_name', '')),
                            'birthdate': str(row2.get('birthdate', '')),
                            'province_name': str(row2.get('province_name', ''))
                        }

                        score, classification, component_scores = self.match_names(
                            name1_data, name2_data
                        )

                        total_comparisons += 1

                        if score >= threshold:
                            results.append({
                                'id1': row1.get('hh_id', idx1),
                                'id2': row2.get('hh_id', idx2),
                                'score': score,
                                'classification': classification.name if hasattr(classification, 'name') else str(classification),
                                **{f'score_{k}': v for k, v in component_scores.items()}
                            })

        matching_time = time.time() - start_time

        # Calculate performance metrics
        potential_comparisons = len(df1) * len(df2)
        comparisons_avoided = potential_comparisons - total_comparisons
        speedup = potential_comparisons / total_comparisons if total_comparisons > 0 else float('inf')

        print(f"Completed {total_comparisons:,} comparisons in {matching_time:.2f}s")
        print(f"Avoided {comparisons_avoided:,} comparisons ({comparisons_avoided/potential_comparisons*100:.1f}%)")
        print(f"Speedup: {speedup:.1f}x")
        print(f"Found {len(results)} matches above threshold {threshold}")

        self.blocking_stats.update({
            'blocks_created': len(set(blocks1.keys()).union(blocks2.keys())),
            'comparisons_avoided': comparisons_avoided,
            'speedup': speedup
        })

        return pd.DataFrame(results)

    def _create_blocking_index(self, df: pd.DataFrame, blocking_fields: List[str]) -> Dict[str, List[int]]:
        """Create blocking index for a DataFrame."""
        blocks = defaultdict(list)

        for idx, row in df.iterrows():
            block_key_parts = []

            for field in blocking_fields:
                if field == 'first_char_first_name':
                    first_name = str(row.get('first_name', '')).strip()
                    value = first_name[:1].upper() if first_name else 'UNKNOWN'
                elif field == 'soundex_last_name':
                    last_name = str(row.get('last_name', '')).strip()
                    if not last_name:
                        # Try middle_name_last_name field
                        middle_last = str(row.get('middle_name_last_name', '')).strip()
                        if middle_last:
                            # Extract last part as last name
                            parts = middle_last.split()
                            last_name = parts[-1] if parts else ''
                    value = self._fast_soundex(last_name)
                elif field == 'birth_year':
                    birthdate = str(row.get('birthdate', ''))
                    value = birthdate[:4] if len(birthdate) >= 4 else 'UNKNOWN'
                elif field == 'province':
                    value = str(row.get('province_name', 'UNKNOWN')).strip().upper()
                else:
                    value = str(row.get(field, 'UNKNOWN')).strip().upper()

                block_key_parts.append(value)

            block_key = '|'.join(block_key_parts)
            blocks[block_key].append(idx)

        return dict(blocks)

    def _fast_soundex(self, name: str) -> str:
        """Fast Soundex implementation for blocking."""
        if not name or not name.strip():
            return "0000"

        name = name.upper().strip()
        if not name.isalpha():
            return "0000"

        # Soundex algorithm
        first_letter = name[0]

        # Replace consonants with digits
        replacements = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }

        coded = first_letter
        for char in name[1:]:
            if char in replacements:
                digit = replacements[char]
                if not coded or coded[-1] != digit:
                    coded += digit
            # Remove vowels and other characters

        # Pad or truncate to 4 characters
        coded = (coded + "000")[:4]
        return coded

# Performance testing function
def test_blocking_performance():
    """Test the performance improvement from blocking."""

    # Create test data
    print("=== Blocking Performance Test ===")

    # Generate realistic test data
    first_names = ['Juan', 'Maria', 'Jose', 'Ana', 'Carlos', 'Rosa', 'Antonio', 'Carmen'] * 125
    last_names = ['Santos', 'Cruz', 'Garcia', 'Lopez', 'Reyes', 'Ramos', 'Mendoza', 'Torres'] * 125

    data1 = {
        'hh_id': range(1000),
        'first_name': first_names,
        'middle_name_last_name': [f"dela {ln}" for ln in last_names],
        'birthdate': ['1990-01-01'] * 1000,
        'province_name': ['Manila'] * 1000
    }

    data2 = {
        'hh_id': range(1000, 2000),
        'first_name': [name + ('o' if i % 3 == 0 else '') for i, name in enumerate(first_names)],  # Add some variation
        'middle_name_last_name': [f"de la {ln}" for ln in last_names],
        'birthdate': ['1990-01-01'] * 1000,
        'province_name': ['Manila'] * 1000
    }

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    print(f"Test data: {len(df1)} × {len(df2)} = {len(df1) * len(df2):,} potential comparisons")

    # Test with blocking
    blocking_matcher = BlockingNameMatcher()

    start_time = time.time()
    results_blocked = blocking_matcher.match_dataframes_with_blocking(df1, df2, threshold=0.7)
    blocked_time = time.time() - start_time

    print(f"\\nBlocking Results:")
    print(f"Time: {blocked_time:.2f}s")
    print(f"Matches found: {len(results_blocked)}")
    print(f"Blocking stats: {blocking_matcher.blocking_stats}")

    # Estimate time for traditional approach
    traditional_matcher = NameMatcher()

    # Test small sample to estimate
    small_df1 = df1.head(50)
    small_df2 = df2.head(50)

    start_time = time.time()
    small_results = traditional_matcher.match_dataframes(small_df1, small_df2)
    small_time = time.time() - start_time

    # Estimate full time
    estimated_full_time = (len(df1) * len(df2) * small_time) / (len(small_df1) * len(small_df2))

    print(f"\\nTraditional Approach (estimated):")
    print(f"Time: {estimated_full_time:.2f}s ({estimated_full_time/60:.1f} minutes)")
    print(f"Speedup from blocking: {estimated_full_time/blocked_time:.1f}x")

    return results_blocked

if __name__ == "__main__":
    test_blocking_performance()
