"""
Database module for the Name Matching application.

This module provides functionality for connecting to and interacting with MySQL databases.
"""

from .connection import get_engine, get_session, init_db
from .models import Base, PersonRecord, MatchResult
from .operations import (
    get_records_from_table,
    save_match_results,
    get_match_results,
    delete_match_results,
)
