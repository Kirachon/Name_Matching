"""
Database connection module.

This module provides functionality for connecting to MySQL databases.
"""

import os
import time
from contextlib import contextmanager
from typing import Dict, Generator, Optional, Union

import dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Global engine instances for different database connections
_engines: Dict[str, Engine] = {}
_session_factories: Dict[str, sessionmaker] = {}

# Default connection parameters
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1  # seconds


def get_connection_string(
    db_name: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[Union[str, int]] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    use_ssl: bool = False,
) -> str:
    """
    Get a SQLAlchemy connection string for MySQL.

    Args:
        db_name: Database name (default: from MYSQL_DATABASE env var)
        host: Database host (default: from MYSQL_HOST env var)
        port: Database port (default: from MYSQL_PORT env var)
        user: Database user (default: from MYSQL_USER env var)
        password: Database password (default: from MYSQL_PASSWORD env var)
        use_ssl: Whether to use SSL for the connection

    Returns:
        SQLAlchemy connection string
    """
    # Use provided values or fall back to environment variables
    db_name = db_name or os.environ.get("MYSQL_DATABASE", "name_matching")
    host = host or os.environ.get("MYSQL_HOST", "localhost")
    port = port or os.environ.get("MYSQL_PORT", "3306")
    user = user or os.environ.get("MYSQL_USER", "root")
    password = password or os.environ.get("MYSQL_PASSWORD", "")

    # Build connection string
    conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"

    # Add SSL if requested
    if use_ssl:
        conn_str += "?ssl=true"

    return conn_str


def get_engine(
    db_name: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[Union[str, int]] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    use_ssl: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: int = 30,
    pool_recycle: int = 1800,
    connection_key: str = "default",
) -> Engine:
    """
    Get a SQLAlchemy engine for the specified database.

    Args:
        db_name: Database name
        host: Database host
        port: Database port
        user: Database user
        password: Database password
        use_ssl: Whether to use SSL for the connection
        pool_size: Connection pool size
        max_overflow: Maximum number of connections to overflow
        pool_timeout: Timeout for getting a connection from the pool
        pool_recycle: Time in seconds to recycle connections
        connection_key: Key to identify this connection in the engine cache

    Returns:
        SQLAlchemy engine
    """
    global _engines

    # Return cached engine if it exists
    if connection_key in _engines:
        return _engines[connection_key]

    # Create connection string
    conn_str = get_connection_string(db_name, host, port, user, password, use_ssl)

    # Create engine with connection pooling
    engine = create_engine(
        conn_str,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        pool_pre_ping=True,  # Check connection validity before using it
    )

    # Cache the engine
    _engines[connection_key] = engine

    return engine


def get_session_factory(engine: Engine) -> sessionmaker:
    """
    Get a session factory for the specified engine.

    Args:
        engine: SQLAlchemy engine

    Returns:
        SQLAlchemy sessionmaker
    """
    return sessionmaker(bind=engine)


def get_session(
    engine: Optional[Engine] = None,
    connection_key: str = "default",
) -> Session:
    """
    Get a SQLAlchemy session.

    Args:
        engine: SQLAlchemy engine (default: engine from get_engine())
        connection_key: Key to identify the connection in the engine cache

    Returns:
        SQLAlchemy session
    """
    global _session_factories

    # Get engine if not provided
    if engine is None:
        engine = get_engine(connection_key=connection_key)

    # Get or create session factory
    if connection_key not in _session_factories:
        _session_factories[connection_key] = get_session_factory(engine)

    # Create and return session
    return _session_factories[connection_key]()


@contextmanager
def session_scope(
    engine: Optional[Engine] = None,
    connection_key: str = "default",
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> Generator[Session, None, None]:
    """
    Context manager for database sessions with automatic commit/rollback.

    Args:
        engine: SQLAlchemy engine
        connection_key: Key to identify the connection in the engine cache
        max_retries: Maximum number of retries for connection errors
        retry_delay: Delay between retries in seconds

    Yields:
        SQLAlchemy session

    Raises:
        SQLAlchemyError: If a database error occurs after all retries
    """
    session = get_session(engine, connection_key)
    retries = 0

    while True:
        try:
            yield session
            session.commit()
            break
        except OperationalError as e:
            # Handle connection errors with retry logic
            if retries < max_retries:
                retries += 1
                time.sleep(retry_delay)
                session.rollback()
                continue
            else:
                session.rollback()
                raise SQLAlchemyError(f"Database error after {max_retries} retries: {e}")
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()


def init_db(
    engine: Optional[Engine] = None,
    connection_key: str = "default",
    create_tables: bool = False,
) -> None:
    """
    Initialize the database.

    Args:
        engine: SQLAlchemy engine
        connection_key: Key to identify the connection in the engine cache
        create_tables: Whether to create tables in the database

    Raises:
        SQLAlchemyError: If a database error occurs
    """
    from .models import Base

    # Get engine if not provided
    if engine is None:
        engine = get_engine(connection_key=connection_key)

    # Create tables if requested
    if create_tables:
        Base.metadata.create_all(engine)
