import pandas as pd
import logging
import os
import sys

# Adjust path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.name_matcher import NameMatcher
from src.matcher import jaro_winkler_similarity, damerau_levenshtein_similarity
from src.config import load_config, get_matching_thresholds # To ensure config is loaded for thresholds

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def generate_review_file(matcher_config_name: str, name_matcher_instance: NameMatcher, data: pd.DataFrame, output_dir: str = "output"):
    """
    Generates a CSV file with detailed match results for qualitative review.
    """
    if data is None or data.empty:
        logger.warning(f"No data to process for {matcher_config_name}.")
        return

    logger.info(f"Generating review file for: {matcher_config_name}")
    
    results_list = []
    for index, row in data.iterrows():
        name1 = str(row['name1'])
        name2 = str(row['name2'])
        id1 = row.get('id1', index) # Use index if id1 is not present
        id2 = row.get('id2', index + len(data)) # Use offset index if id2 is not present
        
        score, classification, component_scores = name_matcher_instance.match_names(name1, name2)
        
        result_row = {
            "id1": id1,
            "name1": name1,
            "id2": id2,
            "name2": name2,
            "overall_score": score,
            "classification": classification.value,
        }
        # Add all component scores, prefixing with "score_"
        for k, v in component_scores.items():
            result_row[f"score_{k}"] = v
        results_list.append(result_row)

    results_df = pd.DataFrame(results_list)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_filename = os.path.join(output_dir, f"qualitative_review_results_{matcher_config_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv")
    try:
        results_df.to_csv(output_filename, index=False, encoding='utf-8')
        logger.info(f"Successfully wrote review file: {output_filename}")
    except Exception as e:
        logger.error(f"Error writing CSV file {output_filename}: {e}")


def main():
    # Ensure config is loaded if NameMatcher relies on it for default thresholds
    load_config() # Load default config.ini if present
    
    benchmark_data = load_benchmark_data()
    if benchmark_data is None:
        logger.error("Could not load benchmark data. Exiting.")
        return

    output_directory = "output/qualitative_reviews"
    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
            logger.info(f"Created output directory: {output_directory}")
        except OSError as e:
            logger.error(f"Could not create output directory {output_directory}: {e}")
            return


    # Define NameMatcher configurations to test
    # These will use thresholds from config if not overridden
    configurations = {
        "Jaro-Winkler_base": NameMatcher(base_component_similarity_func=jaro_winkler_similarity),
        "Damerau-Levenshtein_base": NameMatcher(base_component_similarity_func=damerau_levenshtein_similarity),
    }
    
    # Example of using specific thresholds if needed for a review run:
    # config_thresholds = get_matching_thresholds()
    # custom_matcher = NameMatcher(
    #     match_threshold=config_thresholds['match_threshold'], 
    #     non_match_threshold=config_thresholds['non_match_threshold'],
    #     base_component_similarity_func=jaro_winkler_similarity
    # )
    # configurations["JaroWinkler_ConfigThresholds"] = custom_matcher


    for config_name, matcher_instance in configurations.items():
        generate_review_file(config_name, matcher_instance, benchmark_data, output_dir=output_directory)

    logger.info("Qualitative review file generation complete.")

if __name__ == "__main__":
    main()
