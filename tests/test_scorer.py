"""
Tests for the name scorer module.
"""

import pytest

from src.scorer import (
    score_name_match,
    classify_match,
    score_with_additional_fields,
    MatchClassification,
)


def test_score_name_match():
    """Test the score_name_match function."""
    # Perfect match
    component_scores = {
        "first_name": 1.0,
        "middle_name": 1.0,
        "last_name": 1.0,
        "full_name_sorted": 1.0,
    }
    assert score_name_match(component_scores) == 1.0
    
    # Partial match
    component_scores = {
        "first_name": 1.0,
        "middle_name": 0.5,
        "last_name": 1.0,
        "full_name_sorted": 0.9,
    }
    # Expected: (1.0*0.4 + 0.5*0.2 + 1.0*0.3 + 0.9*0.1) = 0.89
    assert score_name_match(component_scores) == pytest.approx(0.89, abs=0.001)
    
    # Custom weights
    weights = {
        "first_name": 0.5,
        "middle_name": 0.1,
        "last_name": 0.3,
        "full_name_sorted": 0.1,
    }
    # Expected: (1.0*0.5 + 0.5*0.1 + 1.0*0.3 + 0.9*0.1) = 0.94
    assert score_name_match(component_scores, weights) == pytest.approx(0.94, abs=0.001)
    
    # Missing component
    component_scores = {
        "first_name": 1.0,
        "last_name": 1.0,
    }
    # Expected: (1.0*0.4 + 0.0*0.2 + 1.0*0.3 + 0.0*0.1) / (0.4 + 0.2 + 0.3 + 0.1) = 0.7 / 1.0 = 0.7
    assert score_name_match(component_scores) == pytest.approx(0.7, abs=0.001)


def test_classify_match():
    """Test the classify_match function."""
    # Match
    assert classify_match(0.9) == MatchClassification.MATCH
    assert classify_match(0.85) == MatchClassification.MATCH
    
    # Manual review
    assert classify_match(0.8) == MatchClassification.MANUAL_REVIEW
    assert classify_match(0.7) == MatchClassification.MANUAL_REVIEW
    
    # Non-match
    assert classify_match(0.6) == MatchClassification.NON_MATCH
    assert classify_match(0.0) == MatchClassification.NON_MATCH
    
    # Custom thresholds
    assert classify_match(0.8, match_threshold=0.75) == MatchClassification.MATCH
    assert classify_match(0.6, non_match_threshold=0.5) == MatchClassification.MANUAL_REVIEW
    assert classify_match(0.4, non_match_threshold=0.5) == MatchClassification.NON_MATCH


def test_score_with_additional_fields():
    """Test the score_with_additional_fields function."""
    # Name score only
    assert score_with_additional_fields(0.8) == 0.8
    
    # With additional fields
    additional_scores = {
        "birthdate": 1.0,
        "geography": 0.5,
    }
    # Expected: 0.8*0.4 + 1.0*0.3 + 0.5*0.3 = 0.32 + 0.3 + 0.15 = 0.77
    assert score_with_additional_fields(0.8, additional_scores) == pytest.approx(0.77, abs=0.001)
    
    # Custom weights
    additional_weights = {
        "birthdate": 0.4,
        "geography": 0.2,
    }
    # Expected: 0.8*0.4 + 1.0*0.4 + 0.5*0.2 = 0.32 + 0.4 + 0.1 = 0.82
    assert score_with_additional_fields(0.8, additional_scores, additional_weights) == pytest.approx(0.82, abs=0.001)
