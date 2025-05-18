"""
Name match scoring module.

This module provides functions for scoring name matches based on component similarities
and classifying matches as Match, Non-Match, or Manual Review.
"""

from enum import Enum
from typing import Dict, List, Tuple, Union


class MatchClassification(str, Enum):
    """Classification of a name match."""
    MATCH = "match"
    NON_MATCH = "non_match"
    MANUAL_REVIEW = "manual_review"


def score_name_match(component_scores: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """
    Calculate an overall score for a name match based on component similarities.

    Args:
        component_scores: Dictionary with similarity scores for each name component
        weights: Dictionary with weights for each component (default weights if None)

    Returns:
        Overall match score between 0 and 1
    """
    if not weights:
        # Default weights
        weights = {
            "first_name": 0.4,
            "middle_name": 0.2,
            "last_name": 0.3,
            "full_name_sorted": 0.1
        }
    
    # Ensure all required components are present
    for component in weights:
        if component not in component_scores:
            component_scores[component] = 0.0
    
    # Calculate weighted sum
    weighted_sum = sum(
        component_scores[component] * weight
        for component, weight in weights.items()
        if component in component_scores
    )
    
    # Normalize by sum of weights
    total_weight = sum(weight for component, weight in weights.items() if component in component_scores)
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def classify_match(
    score: float,
    match_threshold: float = 0.85,
    non_match_threshold: float = 0.65
) -> MatchClassification:
    """
    Classify a match based on its score.

    Args:
        score: Overall match score
        match_threshold: Threshold for classifying as a match (default: 0.85)
        non_match_threshold: Threshold for classifying as a non-match (default: 0.65)

    Returns:
        MatchClassification enum value
    """
    if score >= match_threshold:
        return MatchClassification.MATCH
    elif score < non_match_threshold:
        return MatchClassification.NON_MATCH
    else:
        return MatchClassification.MANUAL_REVIEW


def score_with_additional_fields(
    name_score: float,
    additional_scores: Dict[str, float] = None,
    additional_weights: Dict[str, float] = None
) -> float:
    """
    Calculate an overall score incorporating additional fields like birthdate and geography.

    Args:
        name_score: The name match score
        additional_scores: Dictionary with similarity scores for additional fields
        additional_weights: Dictionary with weights for additional fields

    Returns:
        Overall match score between 0 and 1
    """
    if not additional_scores:
        return name_score
    
    if not additional_weights:
        # Default weights for additional fields
        additional_weights = {
            "birthdate": 0.3,
            "geography": 0.3,
            # Name score gets the remaining weight
        }
    
    # Calculate name weight (remaining weight)
    name_weight = 1.0 - sum(additional_weights.values())
    
    # Calculate weighted sum
    weighted_sum = name_score * name_weight
    for field, score in additional_scores.items():
        if field in additional_weights:
            weighted_sum += score * additional_weights[field]
    
    return weighted_sum
