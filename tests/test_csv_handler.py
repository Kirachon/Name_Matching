"""
Tests for the CSV handler module.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.csv_handler import (
    read_csv_to_dataframe,
    validate_dataframe,
    default_validators,
    _validate_date,
    standardize_dataframe,
)


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSV file
        csv_path = Path(temp_dir) / "test.csv"
        
        # Create test DataFrame
        df = pd.DataFrame({
            "ID": [1, 2, 3],
            "Name": ["Juan", "Maria", "Pedro"],
            "Surname": ["Santos", "Garcia", "Cruz"],
            "DOB": ["1990-01-01", "1985-05-15", "1978-12-25"],
            "Province": ["Manila", "Cebu", "Davao"],
            "City": ["Quezon City", "Cebu City", "Davao City"],
            "Barangay": ["Barangay 1", "Barangay 2", "Barangay 3"]
        })
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        
        yield str(csv_path)


def test_read_csv_to_dataframe(temp_csv_file):
    """Test reading a CSV file to a DataFrame."""
    # Read without column mapping
    df = read_csv_to_dataframe(temp_csv_file)
    assert len(df) == 3
    assert "ID" in df.columns
    assert "Name" in df.columns
    assert "Surname" in df.columns
    
    # Read with column mapping
    column_mapping = {
        "ID": "hh_id",
        "Name": "first_name",
        "Surname": "last_name",
        "DOB": "birthdate",
        "Province": "province_name",
        "City": "city_name",
        "Barangay": "barangay_name"
    }
    df = read_csv_to_dataframe(temp_csv_file, column_mapping)
    assert len(df) == 3
    assert "hh_id" in df.columns
    assert "first_name" in df.columns
    assert "last_name" in df.columns
    assert "birthdate" in df.columns


def test_validate_dataframe():
    """Test validating a DataFrame."""
    # Create test DataFrame
    df = pd.DataFrame({
        "first_name": ["Juan", "Maria", ""],
        "middle_name_last_name": ["Cruz Santos", "Reyes Garcia", "Dela Cruz"],
        "birthdate": ["1990-01-01", "invalid", "1978-12-25"],
        "province_name": ["Manila", "Cebu", "Davao"],
    })
    
    # Validate with required columns
    required_columns = ["first_name", "middle_name_last_name"]
    is_valid, errors = validate_dataframe(df, required_columns)
    assert is_valid is True
    assert len(errors) == 0
    
    # Validate with missing required column
    required_columns = ["first_name", "middle_name_last_name", "missing_column"]
    is_valid, errors = validate_dataframe(df, required_columns)
    assert is_valid is False
    assert len(errors) == 1
    assert "missing_column" in errors[0]
    
    # Validate with validators
    validators = {
        "first_name": lambda x: len(x) > 0,
        "birthdate": _validate_date,
    }
    is_valid, errors = validate_dataframe(df, required_columns=None, validators=validators)
    assert is_valid is False
    assert len(errors) == 2  # Both first_name and birthdate have invalid values


def test_default_validators():
    """Test the default validators."""
    validators = default_validators()
    
    # Test first_name validator
    assert validators["first_name"]("Juan") is True
    assert validators["first_name"]("") is False
    assert validators["first_name"](123) is False
    
    # Test birthdate validator
    assert validators["birthdate"]("1990-01-01") is True
    assert validators["birthdate"]("01/01/1990") is True
    assert validators["birthdate"]("January 1, 1990") is True
    assert validators["birthdate"]("invalid") is False
    assert validators["birthdate"](123) is False


def test_validate_date():
    """Test the _validate_date function."""
    assert _validate_date("1990-01-01") is True
    assert _validate_date("01/01/1990") is True
    assert _validate_date("January 1, 1990") is True
    assert _validate_date("1-Jan-1990") is True
    assert _validate_date("invalid") is False
    assert _validate_date(123) is False


def test_standardize_dataframe():
    """Test standardizing a DataFrame."""
    # Create test DataFrame
    df = pd.DataFrame({
        "first_name": ["  Juan  ", "MARIA", "pedro"],
        "middle_name_last_name": ["Cruz Santos", "Reyes Garcia", "Dela Cruz"],
        "birthdate": ["1990-01-01", "01/01/1990", "January 1, 1990"],
        "province_name": ["  Manila  ", "CEBU", "davao"],
    })
    
    # Standardize DataFrame
    result = standardize_dataframe(df)
    
    # Check string columns are stripped
    assert result["first_name"][0] == "Juan"
    assert result["province_name"][0] == "Manila"
    
    # Check date column is standardized
    assert result["birthdate"][0] == "1990-01-01"
    assert result["birthdate"][1] == "1990-01-01"  # Converted from 01/01/1990
    assert result["birthdate"][2] == "1990-01-01"  # Converted from January 1, 1990
