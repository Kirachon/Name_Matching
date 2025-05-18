# Running the Name Matching Application

This document explains how to run the Name Matching application.

## Prerequisites

1. Install Python 3.7 or higher
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

The Name Matching application can be run in several ways:

### 1. As a Library

You can import and use the Name Matching library in your own Python code:

```python
from src import NameMatcher

# Create a matcher
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

### 2. Using the Command-Line Interface (CLI)

The application provides a command-line interface for basic operations:

```bash
# Run the main script
python main.py
```

This will show the available commands. Here are some examples:

#### Match two names:

```bash
python main.py match-names "Juan Cruz Santos" "Juan Crux Santos"
```

With additional fields:

```bash
python main.py match-names "Juan Cruz Santos" "Juan Crux Santos" \
    --birthdate1 "1990-01-01" --birthdate2 "1990-01-01" \
    --province1 "Manila" --province2 "Manila"
```

#### Match CSV files:

```bash
python main.py match-csv file1.csv file2.csv \
    --column-mapping "Name=first_name" "Surname=middle_name_last_name" \
    --output matches.csv
```

#### Match database tables (if MySQL support is enabled):

```bash
python main.py match-db table1 table2 \
    --blocking-fields "province_name,city_name" \
    --output matches.csv
```

### 3. Database Configuration

If you're using the MySQL database functionality, you need to configure the database connection:

1. Copy the `.env.sample` file to `.env`:
   ```bash
   cp .env.sample .env
   ```

2. Edit the `.env` file with your database credentials:
   ```
   MYSQL_HOST=localhost
   MYSQL_PORT=3306
   MYSQL_USER=root
   MYSQL_PASSWORD=your_password
   MYSQL_DATABASE=name_matching
   ```

3. Create the database tables:
   ```bash
   python -c "from src import get_engine, init_db; init_db(get_engine(), create_tables=True)"
   ```

## Implementing a GUI

The repository includes an outline for a GUI implementation in `src/gui_outline.py`. This is not a complete implementation but provides a starting point for developing a graphical interface.

To create a working GUI:

1. Install the required dependencies:
   ```bash
   pip install tkinter
   ```

2. Complete the implementation based on the outline
3. Create a main script to launch the GUI

## Examples

### Basic Name Matching

```bash
python main.py match-names "Juan Cruz Santos" "Juan Crux Santos"
```

Output:
```
Match score: 0.9500
Classification: MatchClassification.MATCH
Component scores:
  first_name: 1.0000
  middle_name: 0.8000
  last_name: 1.0000
  full_name_sorted: 0.9500
  name_score: 0.9500
```

### CSV File Matching

```bash
python main.py match-csv data/file1.csv data/file2.csv --output matches.csv
```

This will match records between two CSV files and save the results to `matches.csv`.

### Database Table Matching

```bash
python main.py match-db table1 table2 --create-tables
```

This will create the necessary database tables (if they don't exist) and match records between the two tables.
