import pytest
from src.evaluation import precision, recall, f1_score, calculate_metrics

def test_precision():
    assert precision(true_positives=0, false_positives=0) == 0.0
    assert precision(true_positives=10, false_positives=0) == 1.0
    assert precision(true_positives=0, false_positives=10) == 0.0
    assert precision(true_positives=10, false_positives=10) == 0.5
    assert precision(true_positives=5, false_positives=15) == 0.25

def test_recall():
    assert recall(true_positives=0, false_negatives=0) == 0.0
    assert recall(true_positives=10, false_negatives=0) == 1.0
    assert recall(true_positives=0, false_negatives=10) == 0.0
    assert recall(true_positives=10, false_negatives=10) == 0.5
    assert recall(true_positives=15, false_negatives=5) == 0.75

def test_f1_score():
    assert f1_score(precision_val=0.0, recall_val=0.0) == 0.0
    assert f1_score(precision_val=1.0, recall_val=1.0) == 1.0
    assert f1_score(precision_val=0.5, recall_val=0.5) == 0.5
    assert f1_score(precision_val=0.25, recall_val=0.75) == pytest.approx(0.375)
    assert f1_score(precision_val=0.0, recall_val=1.0) == 0.0
    assert f1_score(precision_val=1.0, recall_val=0.0) == 0.0

def test_calculate_metrics_empty_input():
    """Test calculate_metrics with empty labeled_results."""
    metrics = calculate_metrics([], match_threshold=0.8, non_match_threshold=0.6)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0

def test_calculate_metrics_perfect_match_scenario():
    """Test calculate_metrics with perfect matches."""
    labeled_results = [
        {"predicted_score": 0.9, "true_label": "match"},
        {"predicted_score": 0.85, "true_label": "match"},
    ]
    metrics = calculate_metrics(labeled_results, match_threshold=0.8, non_match_threshold=0.6)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0

def test_calculate_metrics_no_matches_found():
    """Test calculate_metrics when no predicted scores meet the match_threshold."""
    labeled_results = [
        {"predicted_score": 0.7, "true_label": "match"}, # FN
        {"predicted_score": 0.5, "true_label": "match"}, # FN
        {"predicted_score": 0.4, "true_label": "non-match"}, # TN (correctly not a match)
    ]
    metrics = calculate_metrics(labeled_results, match_threshold=0.8, non_match_threshold=0.6)
    assert metrics["precision"] == 0.0 # TP = 0
    assert metrics["recall"] == 0.0    # TP = 0
    assert metrics["f1_score"] == 0.0
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 2

def test_calculate_metrics_all_false_positives():
    """Test calculate_metrics with all predictions being false positives."""
    labeled_results = [
        {"predicted_score": 0.9, "true_label": "non-match"}, # FP
        {"predicted_score": 0.85, "true_label": "non-match"},# FP
    ]
    metrics = calculate_metrics(labeled_results, match_threshold=0.8, non_match_threshold=0.6)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0 # TP is 0
    assert metrics["f1_score"] == 0.0
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 2
    assert metrics["false_negatives"] == 0

def test_calculate_metrics_mixed_scenario():
    """Test calculate_metrics with a mix of TP, FP, FN."""
    labeled_results = [
        {"predicted_score": 0.95, "true_label": "match"},     # TP
        {"predicted_score": 0.80, "true_label": "match"},     # TP
        {"predicted_score": 0.70, "true_label": "non-match"}, # FP (score > non_match_threshold but < match_threshold, still predicted as match due to >= match_threshold)
        {"predicted_score": 0.85, "true_label": "non-match"}, # FP
        {"predicted_score": 0.60, "true_label": "match"},     # FN (score < match_threshold)
        {"predicted_score": 0.30, "true_label": "non-match"}, # TN
        {"predicted_score": 0.78, "true_label": "match"},     # TP
        {"predicted_score": 0.50, "true_label": "match"}      # FN
    ]
    # Using match_threshold = 0.75, non_match_threshold = 0.55 (from NameMatcher defaults)
    # TP: 0.95 (match), 0.80 (match), 0.78 (match) -> 3
    # FP: 0.85 (non-match) -> 1 (0.70 is not >= 0.75, so not FP)
    # FN: 0.70 (non-match but actually match), 0.60 (match), 0.50 (match)
    #   - 0.70 (non-match) is true_label="non-match", predicted_as_match is False (0.70 < 0.75). This is a TN.
    #   - Corrected understanding of calculate_metrics:
    #     - result1 (0.95, match): predicted_as_match=True, is_actually_match=True -> TP
    #     - result2 (0.80, match): predicted_as_match=True, is_actually_match=True -> TP
    #     - result3 (0.70, non-match): predicted_as_match=False (0.70 < 0.75), is_actually_match=False -> TN
    #     - result4 (0.85, non-match): predicted_as_match=True (0.85 >= 0.75), is_actually_match=False -> FP
    #     - result5 (0.60, match): predicted_as_match=False (0.60 < 0.75), is_actually_match=True -> FN
    #     - result6 (0.30, non-match): predicted_as_match=False (0.30 < 0.75), is_actually_match=False -> TN
    #     - result7 (0.78, match): predicted_as_match=True (0.78 >= 0.75), is_actually_match=True -> TP
    #     - result8 (0.50, match): predicted_as_match=False (0.50 < 0.75), is_actually_match=True -> FN
    # TP = 3 (0.95, 0.80, 0.78)
    # FP = 1 (0.85)
    # FN = 2 (0.60, 0.50)
    
    metrics = calculate_metrics(labeled_results, match_threshold=0.75, non_match_threshold=0.55)
    expected_precision = precision(3, 1) # 3 / (3+1) = 0.75
    expected_recall = recall(3, 2)    # 3 / (3+2) = 0.6
    expected_f1 = f1_score(expected_precision, expected_recall) # 2 * (0.75 * 0.6) / (0.75 + 0.6) = 2 * 0.45 / 1.35 = 0.9 / 1.35 = 0.666...

    assert metrics["true_positives"] == 3
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 2
    assert metrics["precision"] == pytest.approx(expected_precision)
    assert metrics["recall"] == pytest.approx(expected_recall)
    assert metrics["f1_score"] == pytest.approx(expected_f1)

def test_calculate_metrics_key_names():
    """Test calculate_metrics with custom key names for score and label."""
    labeled_results = [
        {"s": 0.9, "actual": "match"},
        {"s": 0.7, "actual": "non-match"},
    ]
    metrics = calculate_metrics(
        labeled_results, 
        match_threshold=0.8, 
        non_match_threshold=0.6,
        true_label_key="actual",
        predicted_score_key="s"
    )
    # TP = 1 (0.9, match)
    # FP = 0 (0.7 is not >= 0.8)
    # FN = 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0

def test_calculate_metrics_missing_keys(caplog):
    """Test calculate_metrics when score or label keys are missing."""
    import logging
    caplog.set_level(logging.WARNING)
    labeled_results = [
        {"predicted_score": 0.9}, # Missing true_label
        {"true_label": "match"},   # Missing predicted_score
        {"predicted_score": 0.8, "true_label": "match"} # Valid
    ]
    metrics = calculate_metrics(labeled_results, match_threshold=0.7, non_match_threshold=0.5)
    
    assert "Skipping result due to missing score or true_label" in caplog.text
    # Only one valid record
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0

def test_calculate_metrics_non_match_threshold_impact():
    """
    Test that non_match_threshold doesn't directly impact P/R/F1 as defined,
    but it's used for 'uncertain' classification which isn't directly measured here.
    The calculate_metrics function only cares about the match_threshold for TP/FP/FN.
    """
    labeled_results = [
        {"predicted_score": 0.7, "true_label": "match"},     # FN if match_threshold=0.8, TP if match_threshold=0.6
        {"predicted_score": 0.65, "true_label": "non-match"},# TN if match_threshold=0.8, FP if match_threshold=0.6
    ]
    
    # Scenario 1: match_threshold = 0.8
    # (0.7, match) -> FN
    # (0.65, non-match) -> TN
    metrics1 = calculate_metrics(labeled_results, match_threshold=0.8, non_match_threshold=0.5)
    assert metrics1["true_positives"] == 0
    assert metrics1["false_positives"] == 0
    assert metrics1["false_negatives"] == 1
    assert metrics1["precision"] == 0.0
    assert metrics1["recall"] == 0.0
    assert metrics1["f1_score"] == 0.0

    # Scenario 2: match_threshold = 0.6
    # (0.7, match) -> TP
    # (0.65, non-match) -> FP
    metrics2 = calculate_metrics(labeled_results, match_threshold=0.6, non_match_threshold=0.5)
    assert metrics2["true_positives"] == 1
    assert metrics2["false_positives"] == 1
    assert metrics2["false_negatives"] == 0
    assert metrics2["precision"] == 0.5
    assert metrics2["recall"] == 1.0
    assert metrics2["f1_score"] == pytest.approx(2 * (0.5 * 1.0) / (0.5 + 1.0))

```
