"""
Name matcher module.

This module provides the main interface for name matching functionality.
"""

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from sqlalchemy import Engine

from .csv_handler import read_csv_to_dataframe, standardize_dataframe
from .matcher import (
    compare_name_components,
    jaro_winkler_similarity,
    damerau_levenshtein_similarity,
    monge_elkan_similarity,
)
from .parser import parse_name, tokenize_name
from .matcher import jaro_winkler_similarity as default_similarity_func # for default value
from .scorer import MatchClassification, classify_match, score_name_match
from .standardizer import standardize_name, standardize_name_components
from .config import get_matching_thresholds # Import the new function
import logging # Ensure logging is imported
import time

logger = logging.getLogger(__name__)

# Import database modules if available
try:
    from .db.connection import get_engine, session_scope
    from .db.models import PersonRecord, MatchResult
    from .db.operations import (
        get_records_from_table,
        get_records_as_dataframe,
        save_match_results,
        get_blocking_candidates,
    )
    _has_db_support = True
except ImportError as e:
    logger.warning(f"Database support not available: {e}")
    _has_db_support = False

# Import GPU acceleration if available
try:
    from .gpu_acceleration import GPUNameMatcher, get_gpu_status, configure_gpu
    from .config import get_gpu_config
    _has_gpu_support = True

    # Configure GPU from config file
    try:
        gpu_config = get_gpu_config()
        configure_gpu(gpu_config)
        logger.info(f"GPU configuration loaded: {gpu_config}")
    except Exception as e:
        logger.warning(f"Failed to load GPU configuration: {e}")
except ImportError as e:
    logger.warning(f"GPU acceleration not available: {e}")
    _has_gpu_support = False


class NameMatcher:
    """
    Main class for name matching functionality.
    """

    def __init__(
        self,
        match_threshold: float = 0.75,  # Lowered from 0.85
        non_match_threshold: float = 0.55,  # Lowered from 0.65
        name_weights: Dict[str, float] = None,
        additional_field_weights: Dict[str, float] = None,
        base_component_similarity_func = None,
        enable_gpu: bool = None,
        gpu_framework: str = None
    ):
        """
        Initialize the NameMatcher.

        Args:
            match_threshold: Threshold for classifying as a match. If None, loads from config.
            non_match_threshold: Threshold for classifying as a non-match. If None, loads from config.
            name_weights: Dictionary with weights for name components
            additional_field_weights: Dictionary with weights for additional fields
            base_component_similarity_func: The similarity function to use for comparing individual name components.
                                            Defaults to jaro_winkler_similarity.
            enable_gpu: Whether to enable GPU acceleration. If None, uses config setting.
            gpu_framework: Preferred GPU framework ('cupy', 'torch', 'numba'). If None, auto-selects.
        """
        config_thresholds = get_matching_thresholds()

        self.match_threshold = match_threshold if match_threshold is not None else config_thresholds["match_threshold"]
        self.non_match_threshold = non_match_threshold if non_match_threshold is not None else config_thresholds["non_match_threshold"]

        logger.info(f"NameMatcher initialized with match_threshold: {self.match_threshold}, non_match_threshold: {self.non_match_threshold}")

        self.base_component_similarity_func = base_component_similarity_func or default_similarity_func
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

        # Initialize GPU acceleration
        self.gpu_matcher = None
        if _has_gpu_support:
            try:
                # Use provided settings or fall back to config
                if enable_gpu is None:
                    gpu_config = get_gpu_config()
                    enable_gpu = gpu_config.get('enabled', True)

                if enable_gpu:
                    self.gpu_matcher = GPUNameMatcher(enable_gpu=True, framework=gpu_framework)
                    logger.info(f"GPU acceleration enabled: {self.gpu_matcher.get_gpu_info()}")
                else:
                    logger.info("GPU acceleration disabled by configuration")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU acceleration: {e}")
                self.gpu_matcher = None
        else:
            logger.info("GPU acceleration not available")

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
        component_scores = compare_name_components(name1_std, name2_std, similarity_function=self.base_component_similarity_func)

        # Add Monge-Elkan score using Damerau-Levenshtein as secondary
        # This requires tokenizing the full standardized names
        # For simplicity, we'll re-standardize and tokenize the full names here.
        # A more optimized approach might pass tokenized names around.

        # Reconstruct full names for Monge-Elkan tokenization
        # This is a simplified approach; ideally, standardization and tokenization
        # would be more streamlined if Monge-Elkan is a primary strategy.
        full_name1_str = " ".join(filter(None, [name1_std.get("first_name", ""), name1_std.get("middle_name", ""), name1_std.get("last_name", "")]))
        full_name2_str = " ".join(filter(None, [name2_std.get("first_name", ""), name2_std.get("middle_name", ""), name2_std.get("last_name", "")]))

        name1_tokens = tokenize_name(full_name1_str)
        name2_tokens = tokenize_name(full_name2_str)

        if name1_tokens and name2_tokens: # Ensure tokens are not empty
            component_scores["monge_elkan_dl"] = monge_elkan_similarity(
                name1_tokens,
                name2_tokens,
                damerau_levenshtein_similarity
            )
            component_scores["monge_elkan_jw"] = monge_elkan_similarity(
                name1_tokens,
                name2_tokens,
                jaro_winkler_similarity
            )
        else:
            component_scores["monge_elkan_dl"] = 0.0
            component_scores["monge_elkan_jw"] = 0.0


        # Calculate name score using existing component_scores (which uses Jaro-Winkler by default)
        # The new Monge-Elkan scores are available in component_scores but not directly used in the default name_score calculation yet.
        # This would require changing score_name_match or the weights. For now, it's just added as an additional metric.
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

    def match_dataframes_gpu(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        name_columns: Dict[str, str] = None,
        algorithm: str = 'jaro_winkler',
        threshold: float = None,
        batch_size: int = None
    ) -> pd.DataFrame:
        """
        GPU-accelerated batch matching between two DataFrames.

        This method uses GPU acceleration for similarity calculations when available,
        providing significant speedup for large datasets.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            name_columns: Dictionary mapping DataFrame columns to name components
            algorithm: Similarity algorithm ('jaro_winkler', 'levenshtein')
            threshold: Minimum similarity threshold for matches
            batch_size: Batch size for GPU processing

        Returns:
            DataFrame with match results
        """
        if not self.gpu_matcher or not self.gpu_matcher.enable_gpu:
            logger.info("GPU not available, falling back to CPU implementation")
            return self.match_dataframes(df1, df2, name_columns)

        if threshold is None:
            threshold = self.non_match_threshold

        if name_columns is None:
            name_columns = {
                "first_name": "first_name",
                "middle_name_last_name": "middle_name_last_name"
            }

        logger.info(f"Starting GPU-accelerated matching: {len(df1)} × {len(df2)} = {len(df1) * len(df2):,} comparisons")

        try:
            # Extract names from DataFrames
            names1 = []
            names2 = []

            for _, row in df1.iterrows():
                # Construct full name from components
                name_parts = []
                for component, column in name_columns.items():
                    if column in row and pd.notna(row[column]):
                        name_parts.append(str(row[column]).strip())
                full_name = ' '.join(name_parts)
                names1.append(full_name)

            for _, row in df2.iterrows():
                # Construct full name from components
                name_parts = []
                for component, column in name_columns.items():
                    if column in row and pd.notna(row[column]):
                        name_parts.append(str(row[column]).strip())
                full_name = ' '.join(name_parts)
                names2.append(full_name)

            # GPU-accelerated similarity calculation
            start_time = time.time()
            similarity_matrix = self.gpu_matcher.batch_similarity_matrix(
                names1, names2, algorithm=algorithm
            )
            gpu_time = time.time() - start_time

            total_comparisons = len(names1) * len(names2)
            logger.info(f"GPU similarity calculation: {total_comparisons:,} comparisons in {gpu_time:.4f}s "
                       f"({total_comparisons/gpu_time:.0f} comp/sec)")

            # Extract matches above threshold
            results = []
            matches_found = 0

            for i in range(len(names1)):
                for j in range(len(names2)):
                    score = similarity_matrix[i, j]
                    if score >= threshold:
                        # Get IDs
                        id1 = df1.iloc[i].get('hh_id', i)
                        id2 = df2.iloc[j].get('hh_id', j)

                        # Classify match
                        if score >= self.match_threshold:
                            classification = 'MATCH'
                        elif score >= self.non_match_threshold:
                            classification = 'POSSIBLE_MATCH'
                        else:
                            classification = 'NON_MATCH'

                        results.append({
                            'id1': id1,
                            'id2': id2,
                            'score': float(score),
                            'classification': classification,
                            'algorithm': algorithm,
                            'name1': names1[i],
                            'name2': names2[j]
                        })
                        matches_found += 1

            logger.info(f"GPU matching completed: {matches_found} matches found above threshold {threshold}")

            return pd.DataFrame(results) if results else pd.DataFrame(columns=[
                'id1', 'id2', 'score', 'classification', 'algorithm', 'name1', 'name2'
            ])

        except Exception as e:
            logger.error(f"GPU matching failed: {e}, falling back to CPU")
            return self.match_dataframes(df1, df2, name_columns)

    def match_db_tables(
        self,
        table1_name: str,
        table2_name: str,
        engine: Optional[Engine] = None,
        connection_key: str = "default",
        use_blocking: bool = True,
        blocking_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        save_results: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Match records between two database tables.

        Args:
            table1_name: Name of the first table
            table2_name: Name of the second table
            engine: SQLAlchemy engine
            connection_key: Key to identify the connection in the engine cache
            use_blocking: Whether to use blocking to reduce the number of comparisons
            blocking_fields: List of fields to use for blocking
            limit: Maximum number of matches to return
            save_results: Whether to save match results to the database
            **kwargs: Additional arguments to pass to match_dataframes

        Returns:
            DataFrame with match results

        Raises:
            ImportError: If database support is not available
            SQLAlchemyError: If a database error occurs
        """
        if not _has_db_support:
            raise ImportError(
                "Database support is not available. "
                "Make sure SQLAlchemy and PyMySQL are installed."
            )

        # Get records from tables as DataFrames
        df1 = get_records_as_dataframe(table1_name, engine, connection_key)
        df2 = get_records_as_dataframe(table2_name, engine, connection_key)

        # Use blocking if requested
        if use_blocking and blocking_fields:
            # Get candidate pairs using blocking
            candidate_pairs = get_blocking_candidates(
                table1_name, table2_name, blocking_fields, engine, connection_key, limit
            )

            # Filter DataFrames to include only candidates
            if candidate_pairs:
                record1_ids = [pair[0] for pair in candidate_pairs]
                record2_ids = [pair[1] for pair in candidate_pairs]
                df1 = df1[df1["id"].isin(record1_ids)]
                df2 = df2[df2["id"].isin(record2_ids)]

        # Match DataFrames
        results = self.match_dataframes(df1, df2, **kwargs)

        # Save results if requested
        if save_results and not results.empty:
            # Convert results to list of dictionaries
            match_results = []
            for _, row in results.iterrows():
                match_result = {
                    "record1_id": row["id1"],
                    "record2_id": row["id2"],
                    "score": row["score"],
                    "classification": row["classification"],
                }

                # Add component scores if available
                for col in results.columns:
                    if col.startswith("score_") and col not in ["score_name_score"]:
                        match_result[col] = row[col]

                match_results.append(match_result)

            # Save to database
            save_match_results(match_results, engine, connection_key)

        return results

    def match_db_records(
        self,
        record1_id: int,
        record2_id: int,
        engine: Optional[Engine] = None,
        connection_key: str = "default",
        save_result: bool = True,
    ) -> Tuple[float, MatchClassification, Dict[str, float]]:
        """
        Match two specific records from the database.

        Args:
            record1_id: ID of the first record
            record2_id: ID of the second record
            engine: SQLAlchemy engine
            connection_key: Key to identify the connection in the engine cache
            save_result: Whether to save the match result to the database

        Returns:
            Tuple of (overall_score, classification, component_scores)

        Raises:
            ImportError: If database support is not available
            SQLAlchemyError: If a database error occurs
            ValueError: If either record is not found
        """
        if not _has_db_support:
            raise ImportError(
                "Database support is not available. "
                "Make sure SQLAlchemy and PyMySQL are installed."
            )

        # Get engine if not provided
        if engine is None:
            engine = get_engine(connection_key=connection_key)

        # Get records from database
        with session_scope(engine, connection_key) as session:
            record1 = session.query(PersonRecord).filter(PersonRecord.id == record1_id).first()
            record2 = session.query(PersonRecord).filter(PersonRecord.id == record2_id).first()

            if record1 is None:
                raise ValueError(f"Record with ID {record1_id} not found")
            if record2 is None:
                raise ValueError(f"Record with ID {record2_id} not found")

            # Convert to dictionaries
            record1_dict = record1.to_dict()
            record2_dict = record2.to_dict()

        # Extract name components
        name1 = {
            "first_name": record1_dict["first_name"],
            "middle_name": record1_dict["middle_name"],
            "last_name": record1_dict["last_name"],
        }

        name2 = {
            "first_name": record2_dict["first_name"],
            "middle_name": record2_dict["middle_name"],
            "last_name": record2_dict["last_name"],
        }

        # Extract additional fields
        additional_fields1 = {
            "birthdate": record1_dict["birthdate"],
            "province_name": record1_dict["province_name"],
            "city_name": record1_dict["city_name"],
            "barangay_name": record1_dict["barangay_name"],
        }

        additional_fields2 = {
            "birthdate": record2_dict["birthdate"],
            "province_name": record2_dict["province_name"],
            "city_name": record2_dict["city_name"],
            "barangay_name": record2_dict["barangay_name"],
        }

        # Match names
        score, classification, component_scores = self.match_names(
            name1, name2, additional_fields1, additional_fields2
        )

        # Save result if requested
        if save_result:
            match_result = {
                "record1_id": record1_id,
                "record2_id": record2_id,
                "score": score,
                "classification": classification.value,
            }

            # Add component scores
            for key, value in component_scores.items():
                if key != "name_score":
                    match_result[f"score_{key}"] = value

            # Save to database
            save_match_results([match_result], engine, connection_key)

        return score, classification, component_scores
