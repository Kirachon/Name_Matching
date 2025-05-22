"""
Database operations for the Name Matching application.

This module provides functions for interacting with the database.
"""

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from sqlalchemy import Engine, and_, func, or_, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import logging
from .connection import get_engine, session_scope
from .models import MatchResult, PersonRecord

logger = logging.getLogger(__name__)


def get_records_from_table(
    table_name: str,
    engine: Optional[Engine] = None,
    connection_key: str = "default",
    limit: Optional[int] = None,
    offset: Optional[int] = 0,
    filters: Optional[Dict] = None,
) -> List[Dict]:
    """
    Get records from a database table.

    Args:
        table_name: Name of the table to query
        engine: SQLAlchemy engine
        connection_key: Key to identify the connection in the engine cache
        limit: Maximum number of records to return
        offset: Number of records to skip
        filters: Dictionary of column-value pairs to filter by

    Returns:
        List of records as dictionaries

    Raises:
        SQLAlchemyError: If a database error occurs
    """
    logger.info(
        f"Getting records from table: {table_name}, limit: {limit}, offset: {offset}, filters: {filters}"
    )
    try:
        with session_scope(engine, connection_key) as session:
            # Build query
            query = select(PersonRecord).filter(PersonRecord.source_table == table_name)

            # Apply filters if provided
            if filters:
                filter_conditions = []
                for column, value in filters.items():
                    if hasattr(PersonRecord, column):
                        filter_conditions.append(getattr(PersonRecord, column) == value)
                if filter_conditions:
                    query = query.filter(and_(*filter_conditions))

            # Apply limit and offset
            if limit is not None:
                query = query.limit(limit)
            if offset is not None:
                query = query.offset(offset)

            # Execute query
            result = session.execute(query).scalars().all()

            # Convert to dictionaries
            records_to_return = [record.to_dict() for record in result]
            logger.info(f"Retrieved {len(records_to_return)} records from table {table_name}.")
            return records_to_return
    except SQLAlchemyError as e:
        logger.error(f"Error getting records from table {table_name}: {e}", exc_info=True)
        raise SQLAlchemyError(f"Error getting records from table {table_name}: {e}")


def get_records_as_dataframe(
    table_name: str,
    engine: Optional[Engine] = None,
    connection_key: str = "default",
    limit: Optional[int] = None,
    offset: Optional[int] = 0,
    filters: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Get records from a database table as a pandas DataFrame.

    Args:
        table_name: Name of the table to query
        engine: SQLAlchemy engine
        connection_key: Key to identify the connection in the engine cache
        limit: Maximum number of records to return
        offset: Number of records to skip
        filters: Dictionary of column-value pairs to filter by

    Returns:
        DataFrame with records

    Raises:
        SQLAlchemyError: If a database error occurs
    """
    logger.info(
        f"Getting records as DataFrame from table: {table_name}, limit: {limit}, offset: {offset}, filters: {filters}"
    )
    # get_records_from_table already logs success/failure
    records = get_records_from_table(
        table_name, engine, connection_key, limit, offset, filters
    )
    df = pd.DataFrame(records)
    logger.info(f"Converted {len(df)} records from table {table_name} to DataFrame.")
    return df


def save_records(
    records: List[Dict],
    table_name: str,
    engine: Optional[Engine] = None,
    connection_key: str = "default",
) -> List[int]:
    """
    Save records to the database.

    Args:
        records: List of record dictionaries
        table_name: Name of the source table
        engine: SQLAlchemy engine
        connection_key: Key to identify the connection in the engine cache

    Returns:
        List of record IDs

    Raises:
        SQLAlchemyError: If a database error occurs
    """
    logger.info(f"Saving {len(records)} records to table: {table_name}")
    try:
        with session_scope(engine, connection_key) as session:
            # Convert dictionaries to PersonRecord objects
            person_records = [
                PersonRecord.from_dict(record, table_name) for record in records
            ]

            # Add to session
            session.add_all(person_records)
            session.flush()  # Flush to get IDs

            # Return IDs
            saved_ids = [record.id for record in person_records]
            logger.info(f"Saved {len(saved_ids)} records to table {table_name} with IDs: {saved_ids[:10]}...") # Log first 10 IDs
            return saved_ids
    except SQLAlchemyError as e:
        logger.error(f"Error saving records to table {table_name}: {e}", exc_info=True)
        raise SQLAlchemyError(f"Error saving records to table {table_name}: {e}")


def save_match_results(
    match_results: List[Dict],
    engine: Optional[Engine] = None,
    connection_key: str = "default",
) -> List[int]:
    """
    Save match results to the database.

    Args:
        match_results: List of match result dictionaries
        engine: SQLAlchemy engine
        connection_key: Key to identify the connection in the engine cache

    Returns:
        List of match result IDs

    Raises:
        SQLAlchemyError: If a database error occurs
    """
    logger.info(f"Saving {len(match_results)} match results.")
    try:
        with session_scope(engine, connection_key) as session:
            # Convert dictionaries to MatchResult objects
            match_result_objects = [
                MatchResult.from_dict(result) for result in match_results
            ]

            # Add to session
            session.add_all(match_result_objects)
            session.flush()  # Flush to get IDs

            # Return IDs
            saved_ids = [result.id for result in match_result_objects]
            logger.info(f"Saved {len(saved_ids)} match results with IDs: {saved_ids[:10]}...") # Log first 10 IDs
            return saved_ids
    except SQLAlchemyError as e:
        logger.error(f"Error saving match results: {e}", exc_info=True)
        raise SQLAlchemyError(f"Error saving match results: {e}")


def get_match_results(
    record1_id: Optional[int] = None,
    record2_id: Optional[int] = None,
    classification: Optional[str] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = 0,
    engine: Optional[Engine] = None,
    connection_key: str = "default",
) -> List[Dict]:
    """
    Get match results from the database.

    Args:
        record1_id: Filter by record1_id
        record2_id: Filter by record2_id
        classification: Filter by classification
        min_score: Filter by minimum score
        max_score: Filter by maximum score
        limit: Maximum number of results to return
        offset: Number of results to skip
        engine: SQLAlchemy engine
        connection_key: Key to identify the connection in the engine cache

    Returns:
        List of match results as dictionaries

    Raises:
        SQLAlchemyError: If a database error occurs
    """
    logger.info(
        f"Getting match results with filters - record1_id: {record1_id}, record2_id: {record2_id}, "
        f"classification: {classification}, min_score: {min_score}, max_score: {max_score}, "
        f"limit: {limit}, offset: {offset}"
    )
    try:
        with session_scope(engine, connection_key) as session:
            # Build query
            query = select(MatchResult)

            # Apply filters
            if record1_id is not None:
                query = query.filter(MatchResult.record1_id == record1_id)
            if record2_id is not None:
                query = query.filter(MatchResult.record2_id == record2_id)
            if classification is not None:
                query = query.filter(MatchResult.classification == classification)
            if min_score is not None:
                query = query.filter(MatchResult.score >= min_score)
            if max_score is not None:
                query = query.filter(MatchResult.score <= max_score)

            # Apply limit and offset
            if limit is not None:
                query = query.limit(limit)
            if offset is not None:
                query = query.offset(offset)

            # Execute query
            result = session.execute(query).scalars().all()

            # Convert to dictionaries
            results_to_return = [match_result.to_dict() for match_result in result]
            logger.info(f"Retrieved {len(results_to_return)} match results.")
            return results_to_return
    except SQLAlchemyError as e:
        logger.error(f"Error getting match results: {e}", exc_info=True)
        raise SQLAlchemyError(f"Error getting match results: {e}")


def delete_match_results(
    match_result_ids: List[int],
    engine: Optional[Engine] = None,
    connection_key: str = "default",
) -> int:
    """
    Delete match results from the database.

    Args:
        match_result_ids: List of match result IDs to delete
        engine: SQLAlchemy engine
        connection_key: Key to identify the connection in the engine cache

    Returns:
        Number of deleted match results

    Raises:
        SQLAlchemyError: If a database error occurs
    """
    logger.info(f"Deleting {len(match_result_ids)} match results with IDs: {match_result_ids[:10]}...")
    try:
        with session_scope(engine, connection_key) as session:
            # Delete match results
            deleted_count = session.query(MatchResult).filter(
                MatchResult.id.in_(match_result_ids)
            ).delete(synchronize_session=False)
            logger.info(f"Deleted {deleted_count} match results.")
            return deleted_count
    except SQLAlchemyError as e:
        logger.error(f"Error deleting match results: {e}", exc_info=True)
        raise SQLAlchemyError(f"Error deleting match results: {e}")


def get_blocking_candidates(
    table1_name: str,
    table2_name: str,
    blocking_fields: List[str] = None,
    engine: Optional[Engine] = None,
    connection_key: str = "default",
    limit: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Get candidate pairs for matching using blocking.

    Args:
        table1_name: Name of the first table
        table2_name: Name of the second table
        blocking_fields: List of fields to use for blocking
        engine: SQLAlchemy engine
        connection_key: Key to identify the connection in the engine cache
        limit: Maximum number of candidate pairs to return

    Returns:
        List of (record1_id, record2_id) tuples

    Raises:
        SQLAlchemyError: If a database error occurs
    """
    effective_blocking_fields = blocking_fields if blocking_fields is not None else ["province_name", "city_name"]
    logger.info(
        f"Getting blocking candidates for tables: {table1_name}, {table2_name} using fields: {effective_blocking_fields}, limit: {limit}"
    )
    try:
        with session_scope(engine, connection_key) as session:
            # Aliases for the two tables
            r1 = PersonRecord.__table__.alias("r1")
            r2 = PersonRecord.__table__.alias("r2")

            # Build query
            query = select(r1.c.id, r2.c.id).where(
                and_(
                    r1.c.source_table == table1_name,
                    r2.c.source_table == table2_name,
                )
            )

            # Add blocking conditions
            for field in blocking_fields:
                if hasattr(PersonRecord, field):
                    query = query.where(
                        or_(
                            and_(
                                getattr(r1.c, field).is_not(None),
                                getattr(r2.c, field).is_not(None),
                                getattr(r1.c, field) == getattr(r2.c, field),
                            ),
                            and_(
                                getattr(r1.c, field).is_(None),
                                getattr(r2.c, field).is_(None),
                            ),
                        )
                    )

            # Apply limit
            if limit is not None:
                query = query.limit(limit)

            # Execute query
            result = session.execute(query).all()

            # Return as list of tuples
            candidates = [(r[0], r[1]) for r in result]
            logger.info(f"Retrieved {len(candidates)} blocking candidates.")
            return candidates
    except SQLAlchemyError as e:
        logger.error(f"Error getting blocking candidates: {e}", exc_info=True)
        raise SQLAlchemyError(f"Error getting blocking candidates: {e}")


def execute_raw_query(
    query: str,
    params: Optional[Dict] = None,
    engine: Optional[Engine] = None,
    connection_key: str = "default",
) -> List[Dict]:
    """
    Execute a raw SQL query.

    Args:
        query: SQL query string
        params: Query parameters
        engine: SQLAlchemy engine
        connection_key: Key to identify the connection in the engine cache

    Returns:
        Query results as a list of dictionaries

    Raises:
        SQLAlchemyError: If a database error occurs
    """
    logger.info(f"Executing raw query (first 100 chars): {query[:100]}... with params: {params}")
    if params is None:
        params = {}

    try:
        with session_scope(engine, connection_key) as session:
            # Execute query
            result_proxy = session.execute(text(query), params)
            
            # For SELECT queries, process rows. For others, rowcount might be relevant.
            if result_proxy.returns_rows:
                rows = [dict(row._mapping) for row in result_proxy] # Use _mapping for newer SQLAlchemy
                logger.info(f"Raw query returned {len(rows)} rows.")
                return rows
            else:
                logger.info(f"Raw query affected {result_proxy.rowcount} rows.")
                return [{"affected_rows": result_proxy.rowcount}] # Or appropriate response
            
    except SQLAlchemyError as e:
        logger.error(f"Error executing raw query: {e}", exc_info=True)
        raise SQLAlchemyError(f"Error executing raw query: {e}")


def get_table_names(
    engine: Optional[Engine] = None,
    connection_key: str = "default",
) -> List[str]:
    """
    Get a list of table names in the database.

    Args:
        engine: SQLAlchemy engine
        connection_key: Key to identify the connection in the engine cache

    Returns:
        List of table names

    Raises:
        SQLAlchemyError: If a database error occurs
    """
    logger.info("Getting all table names from database.")
    try:
        # Get engine if not provided
        if engine is None:
            engine = get_engine(connection_key=connection_key) # get_engine logs its actions

        # Get table names
        inspector = Engine.inspect(engine) # Changed from engine.inspect() to Engine.inspect(engine)
        table_names = inspector.get_table_names()
        logger.info(f"Found table names: {table_names}")
        return table_names
    except SQLAlchemyError as e:
        logger.error(f"Error getting table names: {e}", exc_info=True)
        raise SQLAlchemyError(f"Error getting table names: {e}")


def get_distinct_values(
    table_name: str,
    column_name: str,
    engine: Optional[Engine] = None,
    connection_key: str = "default",
    limit: Optional[int] = None,
) -> List:
    """
    Get distinct values for a column in a table.

    Args:
        table_name: Name of the table
        column_name: Name of the column
        engine: SQLAlchemy engine
        connection_key: Key to identify the connection in the engine cache
        limit: Maximum number of values to return

    Returns:
        List of distinct values

    Raises:
        SQLAlchemyError: If a database error occurs
        ValueError: If the column does not exist
    """
    logger.info(
        f"Getting distinct values for column: {column_name} from table: {table_name}, limit: {limit}"
    )
    try:
        with session_scope(engine, connection_key) as session:
            # Check if column exists
            if not hasattr(PersonRecord, column_name):
                logger.error(f"Column {column_name} does not exist in PersonRecord model.")
                raise ValueError(f"Column {column_name} does not exist in PersonRecord")

            # Build query
            query = (
                select(getattr(PersonRecord, column_name))
                .filter(PersonRecord.source_table == table_name)
                .distinct()
            )

            # Apply limit
            if limit is not None:
                query = query.limit(limit)

            # Execute query
            result = session.execute(query).scalars().all()

            # Return as list
            distinct_values = list(result)
            logger.info(f"Retrieved {len(distinct_values)} distinct values for {column_name} in {table_name}.")
            return distinct_values
    except ValueError as ve: # Catch specific ValueError for column existence
        logger.error(str(ve)) # Already logged by the check
        raise
    except SQLAlchemyError as e:
        logger.error(
            f"Error getting distinct values for {column_name} in {table_name}: {e}", exc_info=True
        )
        raise SQLAlchemyError(
            f"Error getting distinct values for {column_name} in {table_name}: {e}"
        )
