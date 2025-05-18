"""
Name matcher module.

This module provides the main interface for name matching functionality.
"""

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .csv_handler import read_csv_to_dataframe, standardize_dataframe
from .matcher import compare_name_components, jaro_winkler_similarity
from .parser import parse_name
from .scorer import MatchClassification, classify_match, score_name_match
from .standardizer import standardize_name, standardize_name_components


class NameMatcher:
    """
    Main class for name matching functionality.
    """

    def __init__(
        self,
        match_threshold: float = 0.75,  # Lowered from 0.85
        non_match_threshold: float = 0.55,  # Lowered from 0.65
        name_weights: Dict[str, float] = None,
        additional_field_weights: Dict[str, float] = None
    ):
        """
        Initialize the NameMatcher.

        Args:
            match_threshold: Threshold for classifying as a match
            non_match_threshold: Threshold for classifying as a non-match
            name_weights: Dictionary with weights for name components
            additional_field_weights: Dictionary with weights for additional fields
        """
        self.match_threshold = match_threshold
        self.non_match_threshold = non_match_threshold
        self.name_weights = name_weights or {
            "first_name": 0.4,
            "middle_name": 0.2,
            "last_name": 0.3,
            "full_name_sorted": 0.1
        }
        self.additional_field_weights = additional_field_weights or {
            "birthdate": 0.3,
            "geography": 0.3
        }

    def match_names(
        self,
        name1: Union[str, Dict[str, str]],
        name2: Union[str, Dict[str, str]],
        additional_fields1: Dict[str, str] = None,
        additional_fields2: Dict[str, str] = None
    ) -> Tuple[float, MatchClassification, Dict[str, float]]:
        """
        Match two names and return the score and classification.

        Args:
            name1: First name (string or pre-parsed components)
            name2: Second name (string or pre-parsed components)
            additional_fields1: Additional fields for the first record
            additional_fields2: Additional fields for the second record

        Returns:
            Tuple of (overall_score, classification, component_scores)
        """
        # Parse names if they are strings
        if isinstance(name1, str) and isinstance(name2, str):
            name1_components = parse_name(name1)
            name2_components = parse_name(name2)
        elif isinstance(name1, str):
            name1_components = parse_name(name1)
            name2_components = name2
        elif isinstance(name2, str):
            name1_components = name1
            name2_components = parse_name(name2)
        else:
            name1_components = name1
            name2_components = name2

        # Standardize name components
        name1_std = standardize_name_components(name1_components)
        name2_std = standardize_name_components(name2_components)

        # Compare name components
        component_scores = compare_name_components(name1_std, name2_std)

        # Calculate name score
        name_score = score_name_match(component_scores, self.name_weights)

        # Calculate additional field scores if provided
        additional_scores = {}
        if additional_fields1 and additional_fields2:
            # Compare birthdate (exact match)
            if "birthdate" in additional_fields1 and "birthdate" in additional_fields2:
                additional_scores["birthdate"] = 1.0 if additional_fields1["birthdate"] == additional_fields2["birthdate"] else 0.0

            # Compare geography (simple average of component similarities)
            geo_fields = ["province_name", "city_name", "barangay_name"]
            geo_scores = []
            for field in geo_fields:
                if field in additional_fields1 and field in additional_fields2:
                    field1_std = standardize_name(additional_fields1[field])
                    field2_std = standardize_name(additional_fields2[field])
                    geo_scores.append(jaro_winkler_similarity(field1_std, field2_std))

            if geo_scores:
                additional_scores["geography"] = sum(geo_scores) / len(geo_scores)

        # Calculate overall score
        if additional_scores:
            # Calculate name weight (remaining weight)
            name_weight = 1.0 - sum(self.additional_field_weights.values())

            # Calculate weighted sum
            overall_score = name_score * name_weight
            for field, score in additional_scores.items():
                if field in self.additional_field_weights:
                    overall_score += score * self.additional_field_weights[field]
        else:
            overall_score = name_score

        # Classify the match
        classification = classify_match(
            overall_score,
            self.match_threshold,
            self.non_match_threshold
        )

        # Add additional scores to component scores for detailed reporting
        component_scores.update(additional_scores)
        component_scores["name_score"] = name_score

        return overall_score, classification, component_scores

    def match_dataframes(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        name_columns: Dict[str, str] = None,
        additional_columns: List[str] = None,
        id_column: str = "hh_id",
        limit: int = None
    ) -> pd.DataFrame:
        """
        Match records between two DataFrames.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            name_columns: Dictionary mapping DataFrame columns to name components
                (e.g., {'first_name': 'first_name', 'middle_name_last_name': 'middle_name_last_name'})
            additional_columns: List of additional columns to use for matching
            id_column: Name of the ID column
            limit: Maximum number of matches to return per record in df1

        Returns:
            DataFrame with match results
        """
        if name_columns is None:
            # Default column mapping
            name_columns = {
                "first_name": "first_name",
                "middle_name_last_name": "middle_name_last_name"
            }

        if additional_columns is None:
            additional_columns = ["birthdate", "province_name", "city_name", "barangay_name"]

        # Standardize DataFrames
        df1_std = standardize_dataframe(df1)
        df2_std = standardize_dataframe(df2)

        # Prepare result DataFrame
        results = []

        # Process each record in df1
        for _, row1 in df1_std.iterrows():
            # Extract name components for row1
            name1 = {}
            for component, column in name_columns.items():
                if column in row1:
                    name1[component] = row1[column]

            # Extract additional fields for row1
            additional_fields1 = {}
            for column in additional_columns:
                if column in row1:
                    additional_fields1[column] = row1[column]

            # Match against each record in df2
            matches = []
            for _, row2 in df2_std.iterrows():
                # Extract name components for row2
                name2 = {}
                for component, column in name_columns.items():
                    if column in row2:
                        name2[component] = row2[column]

                # Extract additional fields for row2
                additional_fields2 = {}
                for column in additional_columns:
                    if column in row2:
                        additional_fields2[column] = row2[column]

                # Match the names
                score, classification, component_scores = self.match_names(
                    name1, name2, additional_fields1, additional_fields2
                )

                # Add to matches if not a non-match
                if classification != MatchClassification.NON_MATCH:
                    matches.append({
                        "id1": row1[id_column] if id_column in row1 else None,
                        "id2": row2[id_column] if id_column in row2 else None,
                        "score": score,
                        "classification": classification.value,
                        **{f"score_{k}": v for k, v in component_scores.items()}
                    })

            # Sort matches by score (descending) and limit if needed
            matches.sort(key=lambda x: x["score"], reverse=True)
            if limit:
                matches = matches[:limit]

            # Add to results
            results.extend(matches)

        # Convert results to DataFrame
        if results:
            return pd.DataFrame(results)
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "id1", "id2", "score", "classification",
                *[f"score_{k}" for k in self.name_weights.keys()],
                *[f"score_{k}" for k in self.additional_field_weights.keys()],
                "score_name_score"
            ])

    def match_csv_files(
        self,
        csv_file1: str,
        csv_file2: str,
        column_mapping: Dict[str, str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Match records between two CSV files.

        Args:
            csv_file1: Path to first CSV file
            csv_file2: Path to second CSV file
            column_mapping: Dictionary mapping CSV column names to standard names
            **kwargs: Additional arguments to pass to match_dataframes

        Returns:
            DataFrame with match results
        """
        # Read CSV files
        df1 = read_csv_to_dataframe(csv_file1, column_mapping)
        df2 = read_csv_to_dataframe(csv_file2, column_mapping)

        # Match DataFrames
        return self.match_dataframes(df1, df2, **kwargs)
