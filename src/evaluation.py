import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

def precision(true_positives: int, false_positives: int) -> float:
    """
    Calculates precision.
    Precision = TP / (TP + FP)
    Returns 0.0 if TP + FP is 0 to avoid division by zero.
    """
    if (true_positives + false_positives) == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)

def recall(true_positives: int, false_negatives: int) -> float:
    """
    Calculates recall.
    Recall = TP / (TP + FN)
    Returns 0.0 if TP + FN is 0 to avoid division by zero.
    """
    if (true_positives + false_negatives) == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)

def f1_score(precision_val: float, recall_val: float) -> float:
    """
    Calculates F1-score.
    F1 = 2 * (precision * recall) / (precision + recall)
    Returns 0.0 if precision + recall is 0 to avoid division by zero.
    """
    if (precision_val + recall_val) == 0:
        return 0.0
    return 2 * (precision_val * recall_val) / (precision_val + recall_val)

def calculate_metrics(
    labeled_results: List[Dict[str, Any]], 
    match_threshold: float, 
    non_match_threshold: float,
    true_label_key: str = "true_label", # 'match', 'non-match'
    predicted_score_key: str = "predicted_score"
) -> Dict[str, float]:
    """
    Calculates precision, recall, and F1-score based on labeled results and thresholds.

    Args:
        labeled_results: A list of dictionaries, where each dictionary contains
                         at least the predicted_score and the true_label.
                         Example: [{'predicted_score': 0.9, 'true_label': 'match'}, ...]
        match_threshold: The score above which a pair is classified as a 'match'.
        non_match_threshold: The score below which a pair is classified as a 'non-match'.
                             Scores between non_match_threshold and match_threshold are 'uncertain'.
        true_label_key: The key in the dictionaries for the true label.
        predicted_score_key: The key in the dictionaries for the predicted score.

    Returns:
        A dictionary with keys 'precision', 'recall', 'f1_score'.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    # True negatives are not directly used in these specific metrics but are important for overall accuracy.

    if not labeled_results:
        logger.warning("calculate_metrics received empty labeled_results. Returning all zeros.")
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "true_positives": 0, "false_positives": 0, "false_negatives": 0}

    for result in labeled_results:
        score = result.get(predicted_score_key)
        true_label = result.get(true_label_key)

        if score is None or true_label is None:
            logger.warning(f"Skipping result due to missing score or true_label: {result}")
            continue

        predicted_as_match = score >= match_threshold
        is_actually_match = true_label.lower() == "match"

        if predicted_as_match and is_actually_match:
            true_positives += 1
        elif predicted_as_match and not is_actually_match:
            false_positives += 1
        elif not predicted_as_match and is_actually_match:
            # This counts cases where the score is below match_threshold but should have been a match.
            # This includes scores that might fall into 'uncertain' or 'non-match' based on thresholds.
            false_negatives += 1
            
    prec = precision(true_positives, false_positives)
    rec = recall(true_positives, false_negatives)
    f1 = f1_score(prec, rec)

    logger.info(f"Calculated Metrics: TP={true_positives}, FP={false_positives}, FN={false_negatives}, Precision={prec:.4f}, Recall={rec:.4f}, F1-Score={f1:.4f}")

    return {
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

# Conceptual discussion for threshold optimization (as requested for documentation)
THRESHOLD_OPTIMIZATION_DOCSTRING = """
Conceptual Approaches for Threshold Optimization:

If a labeled dataset (ground truth) is available, the optimal `match_threshold` 
and `non_match_threshold` can be determined empirically using several methods:

1.  **Grid Search / Parameter Sweep:**
    *   Define a range of possible values for `match_threshold` and `non_match_threshold`.
    *   For each combination of thresholds:
        *   Classify the labeled dataset using the current `NameMatcher` configuration.
        *   Calculate evaluation metrics (e.g., F1-score, Precision, Recall) by comparing
          predicted classifications against the true labels.
    *   Select the threshold combination that maximizes the desired metric (often F1-score
      for a balance between precision and recall, or a custom metric based on business needs).
    *   This can be visualized by plotting the metric (e.g., F1-score) against different
      threshold values.

2.  **ROC Curves and Precision-Recall Curves:**
    *   **ROC (Receiver Operating Characteristic) Curve:** Plots True Positive Rate (Recall)
      against False Positive Rate (1 - Specificity) at various threshold settings for the
      `match_threshold`. The area under the ROC curve (AUC-ROC) provides a single measure
      of classifier performance across all thresholds. A threshold can be chosen based on
      the desired balance between TPR and FPR (e.g., the point closest to the top-left corner).
    *   **Precision-Recall Curve:** Plots Precision against Recall for different threshold
      settings of the `match_threshold`. This is particularly useful for imbalanced datasets
      (common in matching problems where true matches are rare). The area under the PR curve
      (AUC-PR) can also be used as a summary metric. A threshold can be chosen based on
      the desired precision/recall trade-off.
    *   These curves primarily help in selecting a single `match_threshold`. The
      `non_match_threshold` might be set relative to it or optimized separately, perhaps
      by analyzing the distribution of scores for true non-matches.

3.  **Algorithm-Specific Thresholds:**
    *   Different similarity algorithms (e.g., Jaro-Winkler, Levenshtein, Monge-Elkan, ML-based)
      produce scores with different distributions and sensitivities.
    *   Therefore, it's highly likely that optimal thresholds will vary depending on the
      underlying algorithm or combination of algorithms used by `NameMatcher`.
    *   If the `NameMatcher` allows configuring different similarity functions (as started
      with `base_component_similarity_func`), threshold optimization should ideally be
      performed for each configuration.
    *   Ensemble methods or weighted scoring systems (like the one in `score_name_match`)
      also mean that the final `predicted_score` distribution will be unique, requiring
      its own threshold optimization.

4.  **Cost-Benefit Analysis:**
    *   In some applications, the cost of a false positive (incorrectly identifying a non-match
      as a match) might be different from the cost of a false negative (missing a true match).
    *   Thresholds can be chosen to minimize an overall cost function that incorporates these
      business-specific costs.

**Practical Considerations:**
*   A dedicated validation dataset (separate from the training set if using ML, and separate
  from the final test set) should be used for threshold tuning to avoid overfitting the
  thresholds to the test data.
*   The distribution of scores for true matches and true non-matches should be analyzed to
  understand the separability and inform initial threshold guesses.
"""

if __name__ == '__main__':
    # Example Usage (requires labeled data)
    # This is a conceptual example as we don't have actual labeled_results here.
    
    print("This module provides evaluation metrics and notes on threshold optimization.")
    print("\nThreshold Optimization Approaches (Docstring):")
    print(THRESHOLD_OPTIMIZATION_DOCSTRING)

    # Dummy example for calculate_metrics
    dummy_labeled_results = [
        {"record1_id": 1, "record2_id": 101, "predicted_score": 0.95, "true_label": "match"},
        {"record1_id": 2, "record2_id": 102, "predicted_score": 0.80, "true_label": "match"},
        {"record1_id": 3, "record2_id": 103, "predicted_score": 0.60, "true_label": "non-match"}, # FP if match_threshold <= 0.6
        {"record1_id": 4, "record2_id": 104, "predicted_score": 0.40, "true_label": "match"},     # FN if match_threshold > 0.4
        {"record1_id": 5, "record2_id": 105, "predicted_score": 0.90, "true_label": "non-match"}, # FP if match_threshold <= 0.9
        {"record1_id": 6, "record2_id": 106, "predicted_score": 0.30, "true_label": "non-match"},
    ]

    print("\nExample metric calculation with dummy data:")
    metrics = calculate_metrics(dummy_labeled_results, match_threshold=0.75, non_match_threshold=0.55)
    print(f"Calculated Metrics: {metrics}")

    metrics_high_precision_thresh = calculate_metrics(dummy_labeled_results, match_threshold=0.92, non_match_threshold=0.55)
    print(f"Calculated Metrics (High Precision Threshold): {metrics_high_precision_thresh}")

    metrics_high_recall_thresh = calculate_metrics(dummy_labeled_results, match_threshold=0.50, non_match_threshold=0.30)
    print(f"Calculated Metrics (High Recall Threshold): {metrics_high_recall_thresh}")
```
