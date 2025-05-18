"""
Database operations for the Name Matching application.

This module provides functions for interacting with the database.
"""

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from sqlalchemy import Engine, and_, func, or_, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .connection import get_engine, session_scope
from .models import MatchResult, PersonRecord


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
            return [record.to_dict() for record in result]
    except SQLAlchemyError as e:
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
    records = get_records_from_table(
        table_name, engine, connection_key, limit, offset, filters
    )
    return pd.DataFrame(records)


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
            return [record.id for record in person_records]
    except SQLAlchemyError as e:
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
            return [result.id for result in match_result_objects]
    except SQLAlchemyError as e:
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
            return [match_result.to_dict() for match_result in result]
    except SQLAlchemyError as e:
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
    try:
        with session_scope(engine, connection_key) as session:
            # Delete match results
            deleted = session.query(MatchResult).filter(
                MatchResult.id.in_(match_result_ids)
            ).delete(synchronize_session=False)
            return deleted
    except SQLAlchemyError as e:
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
    if blocking_fields is None:
        blocking_fields = ["province_name", "city_name"]

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
            return [(r[0], r[1]) for r in result]
    except SQLAlchemyError as e:
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
    if params is None:
        params = {}

    try:
        with session_scope(engine, connection_key) as session:
            # Execute query
            result = session.execute(text(query), params)

            # Convert to dictionaries
            return [dict(row) for row in result]
    except SQLAlchemyError as e:
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
    try:
        # Get engine if not provided
        if engine is None:
            engine = get_engine(connection_key=connection_key)

        # Get table names
        inspector = Engine.inspect(engine)
        return inspector.get_table_names()
    except SQLAlchemyError as e:
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
    try:
        with session_scope(engine, connection_key) as session:
            # Check if column exists
            if not hasattr(PersonRecord, column_name):
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
            return list(result)
    except SQLAlchemyError as e:
        raise SQLAlchemyError(
            f"Error getting distinct values for {column_name} in {table_name}: {e}"
        )
