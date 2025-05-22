# Name Matching

A Python library for name matching and comparison, with special focus on Filipino names.

## Features

- Parse and standardize Filipino names, handling complex `middle_name_last_name` fields.
- Multiple string similarity algorithms:
    - Jaro-Winkler (default for component-wise comparison)
    - Soundex
    - Jaccard Index
    - Damerau-Levenshtein (good for typos including transpositions)
    - Monge-Elkan (effective for multi-token names, uses a secondary similarity measure like Damerau-Levenshtein or Jaro-Winkler)
- Configurable base similarity function for component-wise matching in `NameMatcher` (e.g., Jaro-Winkler or Damerau-Levenshtein).
- Weighted scoring model for name components.
- Support for additional fields (e.g., birthdate, geography) in matching.
- CSV file handling for batch processing.
- MySQL database connectivity for storing and matching records.
- Configuration via `config.ini` for database, logging, and matching thresholds, with environment variable fallbacks.
- Numba JIT compilation for improved CPU performance of core similarity functions.
- Includes scripts for performance benchmarking and qualitative review of matching results.
- Basic evaluation framework (`src/evaluation.py`) for precision, recall, and F1-score.

## Installation

```bash
pip install -r requirements.txt
```

For optional GPU support (currently conceptual for some features, see Performance Considerations):
```bash
# Install base requirements first
pip install -r requirements.txt
# Then install GPU specific libraries (e.g., for RAPIDS)
# pip install -r requirements-gpu.txt 
# Note: requirements-gpu.txt is a placeholder and would need to be created with specific GPU library versions.
# Numba GPU support is included with the standard numba installation if CUDA toolkit is available.
```

## Configuration

The application uses a `config.ini` file for configuration, with fallbacks to environment variables for some settings. A `config.ini.sample` file is provided as a template.

Key configuration sections:

*   **`[database]`**:
    *   `host`: Database host (e.g., `localhost`)
    *   `port`: Database port (e.g., `3306` for MySQL)
    *   `user`: Database user
    *   `password`: Database password
    *   `database`: Database name
    *   `use_ssl`: `true` or `false` for SSL connection.
    *   Environment variables like `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_USE_SSL` can also be used as fallbacks.

*   **`[logging]`**:
    *   `level`: Logging level (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`). Defaults to `INFO`. Can also be set by `LOG_LEVEL` environment variable.

*   **`[matching]`**:
    *   `match_threshold`: Score above which a pair is considered a "match" (default: 0.75).
    *   `non_match_threshold`: Score below which a pair is considered a "non-match" (default: 0.55). Scores between these are "uncertain".

## Usage

### Basic Name Matching

```python
from src import NameMatcher

# Create a matcher with default settings
matcher = NameMatcher()

# Match two names
score, classification, component_scores = matcher.match_names(
    "Juan dela Cruz",
    "Juan de la Cruz"
)

print(f"Match score: {score:.4f}")
print(f"Classification: {classification.value}") # .value to get the string representation
print(f"Component scores: {component_scores}")
# Example output might include:
# Component scores: {'first_name': 1.0, 'middle_name': 0.0, 'last_name': 1.0, 'full_name_sorted': 1.0, 
# 'monge_elkan_dl': 0.9..., 'monge_elkan_jw': 0.9...} 
# (Actual Monge-Elkan scores depend on tokenization and secondary similarity)

```

### Using Different Base Similarity Functions

You can specify the base similarity function for component-wise matching when initializing `NameMatcher`:

```python
from src import NameMatcher
from src.matcher import damerau_levenshtein_similarity

# Use Damerau-Levenshtein for component scores
matcher_dl = NameMatcher(base_component_similarity_func=damerau_levenshtein_similarity)

score, classification, component_scores = matcher_dl.match_names(
    "Johnathan Smith", 
    "Jonathan Smith" 
)
print(f"Match score (D-L base): {score:.4f}")
print(f"Component scores (D-L base): {component_scores}")
```


### Matching with Additional Fields

```python
from src import NameMatcher

# Create a matcher with custom weights
matcher = NameMatcher(
    match_threshold=0.8,
    non_match_threshold=0.6,
    additional_field_weights={
        "birthdate": 0.4,
        "geography": 0.2
    }
)

# Match two names with additional fields
score, classification, component_scores = matcher.match_names(
    "Juan Cruz Santos",
    "Juan Crux Santos",
    {"birthdate": "1990-01-01", "province_name": "Manila", "city_name": "Quezon City"},
    {"birthdate": "1990-01-01", "province_name": "Manila", "city_name": "Quezon City"}
)
```

### CSV File Matching

```python
from src import NameMatcher

# Create a matcher
matcher = NameMatcher()

# Match records between two CSV files
results = matcher.match_csv_files(
    "file1.csv",
    "file2.csv",
    column_mapping={
        "ID": "hh_id",
        "Name": "first_name",
        "Surname": "middle_name_last_name",
        "DOB": "birthdate",
        "Province": "province_name",
        "City": "city_name",
        "Barangay": "barangay_name"
    }
)

# Save results to CSV
results.to_csv("matches.csv", index=False)
```

### Database Connectivity

#### Configuration

Refer to the `config.ini.sample` file for setting up your `config.ini` with database credentials. Environment variables can also be used as fallbacks (e.g., `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_PORT`, `DB_USE_SSL`).

#### Connecting to the Database
The application will automatically attempt to load `config.ini` upon import of the `src.config` module.

```python
from src import get_engine, init_db

# Initialize the database connection
engine = get_engine()

# Create tables if they don't exist
init_db(engine, create_tables=True)
```

#### Matching Records in Database Tables

```python
from src import NameMatcher

# Create a matcher
matcher = NameMatcher()

# Match records between two database tables
results = matcher.match_db_tables(
    "table1",  # First table name
    "table2",  # Second table name
    use_blocking=True,  # Use blocking to reduce comparisons
    blocking_fields=["province_name", "city_name"],  # Fields to use for blocking
    save_results=True  # Save match results to the database
)

# Match specific records by ID
score, classification, component_scores = matcher.match_db_records(
    record1_id=1,
    record2_id=2,
    save_result=True  # Save match result to the database
)
```

## Development

To run tests:

```bash
python -m pytest tests/
```

To run tests with coverage:

```bash
python -m pytest --cov=src tests/
```

## Performance Considerations

*   **CPU Acceleration with Numba:** Core similarity functions like Jaro-Winkler and a fallback Damerau-Levenshtein implementation in `src/matcher.py` are JIT-compiled using Numba for improved CPU performance. This typically happens automatically on the first call to these functions.
*   **GPU Acceleration (Experimental/Future):**
    *   The potential for GPU acceleration using Numba CUDA (for custom kernels) and RAPIDS (cuDF for DataFrame operations, nvtext for text processing) has been explored.
    *   Currently, full GPU acceleration for the entire matching pipeline is not implemented but represents a future development path for significantly speeding up large-scale matching tasks.
    *   Users interested in experimenting with GPU libraries like RAPIDS can set up an environment using `requirements-gpu.txt` (note: this file is a placeholder and would need to be created and populated with specific versions compatible with your CUDA environment). Numba's GPU capabilities are available if a CUDA toolkit is correctly installed and detected.

## Evaluation and Benchmarking

The project includes tools to help evaluate and benchmark the performance and accuracy of the name matching algorithms:

*   **Evaluation Framework (`src/evaluation.py`):**
    *   Provides functions to calculate precision, recall, and F1-score.
    *   Includes `calculate_metrics` function to process labeled match results and compute these metrics.
    *   Contains documentation within the file on conceptual approaches for optimizing match thresholds.

*   **Performance Benchmarking (`scripts/benchmark_performance.py`):**
    *   A script to measure the execution time of the `NameMatcher` with different configurations (e.g., Jaro-Winkler vs. Damerau-Levenshtein as the base component similarity).
    *   Uses the `data/sample_benchmark_names.csv` file for testing.
    *   Example usage: `python scripts/benchmark_performance.py`

*   **Qualitative Review (`scripts/qualitative_review.py`):**
    *   Generates detailed CSV output of matching results for a given set of name pairs (from `data/sample_benchmark_names.csv`).
    *   Output includes individual component scores (first name, middle name, last name, full name sorted, Monge-Elkan with Damerau-Levenshtein, Monge-Elkan with Jaro-Winkler), the overall score, and classification.
    *   This helps in manually reviewing and understanding the matching behavior for different name pairs and algorithm configurations.
    *   Results are saved in the `output/qualitative_reviews/` directory.
    *   Example usage: `python scripts/qualitative_review.py`

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
