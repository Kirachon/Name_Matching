"""
Name Matching module.

A Python library for name matching and comparison, with special focus on Filipino names.
"""

__version__ = "0.1.0"

# Import core functionality
from .parser import parse_name, extract_name_components
from .standardizer import standardize_name, standardize_name_components
from .matcher import (
    jaro_winkler_similarity,
    soundex_similarity,
    jaccard_similarity,
    token_sort_similarity,
    compare_name_components,
)
from .scorer import score_name_match, classify_match, MatchClassification

# Import CSV handling
from .csv_handler import read_csv_to_dataframe, validate_dataframe, standardize_dataframe

# Import main interface
from .name_matcher import NameMatcher

# Import database module if available
try:
    from .db import get_engine, get_session, init_db
    from .db import PersonRecord, MatchResult
    HAS_DB_SUPPORT = True
except ImportError:
    HAS_DB_SUPPORT = False
