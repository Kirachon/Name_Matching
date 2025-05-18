"""
Tests for the name standardizer module.
"""

import pytest

from src.standardizer import (
    standardize_name,
    standardize_name_components,
    remove_name_prefixes,
    remove_name_suffixes,
)


def test_standardize_name_basic():
    """Test basic name standardization."""
    assert standardize_name("Juan") == "juan"
    assert standardize_name("SANTOS") == "santos"
    assert standardize_name("Cruz-Santos") == "cruz-santos"


def test_standardize_name_whitespace():
    """Test standardization with whitespace."""
    assert standardize_name("  Juan  ") == "juan"
    assert standardize_name("Juan\tCruz") == "juan cruz"
    assert standardize_name("Juan\nCruz") == "juan cruz"


def test_standardize_name_unicode():
    """Test standardization with Unicode characters."""
    assert standardize_name("Juañ") == "juan"
    assert standardize_name("Crúz") == "cruz"
    assert standardize_name("Sãntös") == "santos"


def test_standardize_name_special_chars():
    """Test standardization with special characters."""
    assert standardize_name("Juan.Cruz") == "juancruz"
    assert standardize_name("Juan,Cruz") == "juancruz"
    assert standardize_name("Juan-Cruz") == "juan-cruz"  # Hyphens preserved
    assert standardize_name("O'Brien") == "o'brien"  # Apostrophes preserved


def test_standardize_name_components():
    """Test standardization of name components."""
    components = {
        "first_name": "JUAN",
        "middle_name": "De la Cruz",
        "last_name": "SANTOS-REYES",
    }
    
    result = standardize_name_components(components)
    
    assert result["first_name"] == "juan"
    assert result["middle_name"] == "de la cruz"
    assert result["last_name"] == "santos-reyes"


def test_remove_name_prefixes():
    """Test removal of name prefixes."""
    assert remove_name_prefixes("Mr. Juan") == "Juan"
    assert remove_name_prefixes("Mrs. Santos") == "Santos"
    assert remove_name_prefixes("Dr. Juan Cruz") == "Juan Cruz"
    assert remove_name_prefixes("Prof. Santos") == "Santos"
    assert remove_name_prefixes("Juan") == "Juan"  # No prefix


def test_remove_name_suffixes():
    """Test removal of name suffixes."""
    assert remove_name_suffixes("Juan Jr.") == "Juan"
    assert remove_name_suffixes("Santos Sr") == "Santos"
    assert remove_name_suffixes("Juan Cruz III") == "Juan Cruz"
    assert remove_name_suffixes("Santos, PhD") == "Santos"
    assert remove_name_suffixes("Juan") == "Juan"  # No suffix
