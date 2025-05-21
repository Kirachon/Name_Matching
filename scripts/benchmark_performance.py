import time
import pandas as pd
import logging
import os
import sys

# Adjust path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.name_matcher import NameMatcher
from src.matcher import jaro_winkler_similarity, damerau_levenshtein_similarity

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.logging.getLogger(__name__)

def load_benchmark_data(filepath="data/sample_benchmark_names.csv"):
    """Loads benchmark data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} name pairs from {filepath}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: Benchmark file not found at {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading benchmark data: {e}")
        return None

def run_benchmark(matcher_config_name: str, name_matcher_instance: NameMatcher, data: pd.DataFrame):
    """
    Runs and times the matching process for a given NameMatcher configuration.
    """
    if data is None or data.empty:
        logger.warning(f"No data to benchmark for {matcher_config_name}.")
        return float('nan')

    logger.info(f"Starting benchmark for: {matcher_config_name}")

    # Warm-up run (important for JIT compilers like Numba)
    if not data.empty:
        row = data.iloc[0]
        name_matcher_instance.match_names(str(row['name1']), str(row['name2']))
    
    start_time = time.time()
    for index, row in data.iterrows():
        name_matcher_instance.match_names(str(row['name1']), str(row['name2']))
    end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"Finished benchmark for {matcher_config_name}. Time taken: {duration:.4f} seconds.")
    return duration

def main():
    benchmark_data = load_benchmark_data()
    if benchmark_data is None:
        logger.error("Could not load benchmark data. Exiting.")
        return

    # Define NameMatcher configurations to test
    configurations = {
        "Jaro-Winkler (default)": NameMatcher(base_component_similarity_func=jaro_winkler_similarity),
        "Damerau-Levenshtein": NameMatcher(base_component_similarity_func=damerau_levenshtein_similarity),
        # Monge-Elkan is already calculated within match_names and uses a secondary similarity.
        # To test Monge-Elkan as a primary strategy would require changing how NameMatcher.score_name_match works
        # or creating a new NameMatcher variant. For this script, we focus on base_component_similarity_func.
    }

    results = []

    logger.info(f"Benchmarking with {len(benchmark_data)} name pairs.")

    for config_name, matcher_instance in configurations.items():
        duration = run_benchmark(config_name, matcher_instance, benchmark_data)
        results.append({"Configuration": config_name, "Time (seconds)": duration})

    results_df = pd.DataFrame(results)
    print("\n--- Performance Benchmark Results ---")
    print(results_df.to_string(index=False))
    print("-----------------------------------")

if __name__ == "__main__":
    # Ensure the script can find the 'src' module if run directly
    # This is already handled by sys.path.append at the top for imports
    main()
