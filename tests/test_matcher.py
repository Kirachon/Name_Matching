"""
Tests for the name matcher module.
"""

import pytest

from src.matcher import (
    jaro_winkler_similarity,
    _jaro_similarity,
    soundex,
    soundex_similarity,
    jaccard_similarity,
    token_sort_similarity,
    compare_name_components,
)


def test_jaro_similarity():
    """Test the Jaro similarity function."""
    assert _jaro_similarity("MARTHA", "MARHTA") == pytest.approx(0.944, abs=0.001)
    assert _jaro_similarity("DIXON", "DICKSONX") == pytest.approx(0.767, abs=0.001)
    assert _jaro_similarity("JELLYFISH", "SMELLYFISH") == pytest.approx(0.896, abs=0.001)
    
    # Edge cases
    assert _jaro_similarity("", "") == 1.0
    assert _jaro_similarity("A", "A") == 1.0
    assert _jaro_similarity("ABC", "ABC") == 1.0
    assert _jaro_similarity("ABC", "DEF") == 0.0


def test_jaro_winkler_similarity():
    """Test the Jaro-Winkler similarity function."""
    assert jaro_winkler_similarity("MARTHA", "MARHTA") == pytest.approx(0.961, abs=0.001)
    assert jaro_winkler_similarity("DIXON", "DICKSONX") == pytest.approx(0.813, abs=0.001)
    assert jaro_winkler_similarity("JELLYFISH", "SMELLYFISH") == pytest.approx(0.896, abs=0.001)
    
    # Edge cases
    assert jaro_winkler_similarity("", "") == 1.0
    assert jaro_winkler_similarity("A", "A") == 1.0
    assert jaro_winkler_similarity("ABC", "ABC") == 1.0
    assert jaro_winkler_similarity("ABC", "DEF") == 0.0


def test_soundex():
    """Test the Soundex function."""
    assert soundex("Robert") == "R163"
    assert soundex("Rupert") == "R163"
    assert soundex("Rubin") == "R150"
    assert soundex("Ashcraft") == "A261"
    assert soundex("Ashcroft") == "A261"
    assert soundex("Tymczak") == "T522"
    assert soundex("Pfister") == "P236"
    
    # Filipino names
    assert soundex("Santos") == "S532"
    assert soundex("Santoz") == "S532"  # Should match Santos
    assert soundex("Cruz") == "C620"
    assert soundex("Dela Cruz") == "D426"
    
    # Edge cases
    assert soundex("") == "0000"
    assert soundex("A") == "A000"


def test_soundex_similarity():
    """Test the Soundex similarity function."""
    assert soundex_similarity("Robert", "Rupert") == 1.0
    assert soundex_similarity("Ashcraft", "Ashcroft") == 1.0
    assert soundex_similarity("Santos", "Santoz") == 1.0
    assert soundex_similarity("Santos", "Cruz") == 0.0


def test_jaccard_similarity():
    """Test the Jaccard similarity function."""
    assert jaccard_similarity("Juan Cruz", "Juan Santos") == 0.5
    assert jaccard_similarity("Juan Cruz Santos", "Juan Cruz") == 0.5
    assert jaccard_similarity("Juan Cruz", "Juan Cruz") == 1.0
    assert jaccard_similarity("Juan", "Pedro") == 0.0
    
    # Edge cases
    assert jaccard_similarity("", "") == 0.0
    assert jaccard_similarity("Juan", "") == 0.0
    assert jaccard_similarity("", "Juan") == 0.0


def test_token_sort_similarity():
    """Test the token sort similarity function."""
    assert token_sort_similarity("Juan Cruz", "Cruz Juan") == 1.0
    assert token_sort_similarity("Juan Cruz Santos", "Santos Juan Cruz") == 1.0
    assert token_sort_similarity("Juan Cruz", "Juan Santos") < 1.0
    
    # Edge cases
    assert token_sort_similarity("", "") == 1.0
    assert token_sort_similarity("Juan", "") == 0.0
    assert token_sort_similarity("", "Juan") == 0.0


def test_compare_name_components():
    """Test the compare_name_components function."""
    name1 = {
        "first_name": "Juan",
        "middle_name": "Cruz",
        "last_name": "Santos",
    }
    
    name2 = {
        "first_name": "Juan",
        "middle_name": "Crux",  # Slight difference
        "last_name": "Santos",
    }
    
    result = compare_name_components(name1, name2)
    
    assert result["first_name"] == 1.0
    assert result["middle_name"] < 1.0
    assert result["last_name"] == 1.0
    assert result["full_name_sorted"] > 0.9  # Should be high despite middle name difference
