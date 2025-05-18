# Name Matching

A Python library for name matching and comparison, with special focus on Filipino names.

## Features

- Parse and standardize Filipino names, handling complex middle_name_last_name fields
- Multiple string similarity algorithms (Jaro-Winkler, Soundex, Jaccard)
- Weighted scoring model for name components
- Support for additional fields (birthdate, geography) in matching
- CSV file handling for batch processing
- MySQL database connectivity for storing and matching records
- Configurable match/non-match thresholds

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Name Matching

```python
from src import NameMatcher

# Create a matcher with default settings
matcher = NameMatcher()

# Match two names
score, classification, component_scores = matcher.match_names(
    "Juan Cruz Santos",
    "Juan Crux Santos"
)

print(f"Match score: {score}")
print(f"Classification: {classification}")
print(f"Component scores: {component_scores}")
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

Create a `.env` file with your database configuration (see `.env.sample` for an example):

```
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=password
MYSQL_DATABASE=name_matching
```

#### Connecting to the Database

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

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
