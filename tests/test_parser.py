"""
Tests for the name parser module.
"""

import pytest

from src.parser import parse_name, extract_name_components, _split_middle_name_last_name


def test_parse_name_basic():
    """Test basic name parsing."""
    result = parse_name("Juan", "Cruz Santos")
    assert result["first_name"] == "Juan"
    assert result["middle_name"] == "Cruz"
    assert result["last_name"] == "Santos"


def test_parse_name_empty_middle_last():
    """Test parsing with empty middle_name_last_name."""
    result = parse_name("Juan")
    assert result["first_name"] == "Juan"
    assert result["middle_name"] == ""
    assert result["last_name"] == ""


def test_parse_name_compound_surname():
    """Test parsing with compound surname."""
    result = parse_name("Juan", "Cruz Dela Rosa")
    assert result["first_name"] == "Juan"
    assert result["middle_name"] == "Cruz"
    assert result["last_name"] == "Dela Rosa"


def test_parse_name_multiple_middle_names():
    """Test parsing with multiple middle names."""
    result = parse_name("Juan", "Cruz Garcia Santos")
    assert result["first_name"] == "Juan"
    assert result["middle_name"] == "Cruz Garcia"
    assert result["last_name"] == "Santos"


def test_split_middle_name_last_name():
    """Test the _split_middle_name_last_name function."""
    # Basic case
    assert _split_middle_name_last_name("Cruz Santos") == ["Cruz", "Santos"]
    
    # Compound surname
    assert _split_middle_name_last_name("Cruz Dela Rosa") == ["Cruz", "Dela Rosa"]
    
    # Multiple middle names
    assert _split_middle_name_last_name("Cruz Garcia Santos") == ["Cruz", "Garcia", "Santos"]
    
    # Empty string
    assert _split_middle_name_last_name("") == []
    
    # Single name
    assert _split_middle_name_last_name("Santos") == ["Santos"]


def test_extract_name_components():
    """Test the extract_name_components function."""
    # Basic case
    result = extract_name_components("Juan Cruz Santos")
    assert result["first_name"] == "Juan"
    assert result["middle_name"] == "Cruz"
    assert result["last_name"] == "Santos"
    
    # Two names only
    result = extract_name_components("Juan Santos")
    assert result["first_name"] == "Juan"
    assert result["middle_name"] == ""
    assert result["last_name"] == "Santos"
    
    # Single name
    result = extract_name_components("Juan")
    assert result["first_name"] == "Juan"
    assert result["middle_name"] == ""
    assert result["last_name"] == ""
    
    # Empty string
    result = extract_name_components("")
    assert result["first_name"] == ""
    assert result["middle_name"] == ""
    assert result["last_name"] == ""
    
    # Compound surname
    result = extract_name_components("Juan Cruz Dela Rosa")
    assert result["first_name"] == "Juan"
    assert result["middle_name"] == "Cruz"
    assert result["last_name"] == "Dela Rosa"
