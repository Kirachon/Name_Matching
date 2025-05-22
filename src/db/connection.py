"""
Database connection module.

This module provides functionality for connecting to MySQL databases.
"""

import os
import time
import logging
from contextlib import contextmanager
from typing import Dict, Generator, Optional, Union

import dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
# Import the new configuration function
from src.config import get_db_config
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Setup logger for this module
logger = logging.getLogger(__name__)

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
    # Get base configuration from config file / environment variables
    config_params = get_db_config()

    # Use provided values, then config_params, then defaults if any
    final_db_name = db_name or config_params.get("database") or "name_matching"
    final_host = host or config_params.get("host") or "localhost"
    # For port, ensure it's an int if provided, otherwise from config, else default
    final_port_str = None
    if port is not None:
        final_port_str = str(port)
    elif config_params.get("port") is not None:
        final_port_str = str(config_params.get("port"))
    else:
        final_port_str = "3306" # Default MySQL port

    final_user = user or config_params.get("user") or "root"
    final_password = password or config_params.get("password") or ""

    # Handle use_ssl: function argument takes precedence, then config, then default (False)
    final_use_ssl = use_ssl
    if not use_ssl and config_params.get("use_ssl") is not None: # only check config if use_ssl arg is False (its default)
        final_use_ssl = config_params.get("use_ssl", False)


    # Build connection string
    conn_str = f"mysql+pymysql://{final_user}:{final_password}@{final_host}:{final_port_str}/{final_db_name}"

    # Add SSL if requested
    if final_use_ssl:
        conn_str += "?ssl=true" # Assuming pymysql uses ?ssl=true, adjust if different

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
        logger.debug(f"Returning cached engine for key: {connection_key}")
        return _engines[connection_key]

    # Create connection string
    logger.debug(f"Creating new engine for key: {connection_key}")
    conn_str = get_connection_string(db_name, host, port, user, password, use_ssl)
    logger.debug(f"Connection string for {connection_key}: {conn_str.replace(final_password, '****') if final_password else conn_str}")


    # Create engine with connection pooling
    try:
        engine = create_engine(
            conn_str,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,  # Check connection validity before using it
        )
        logger.info(f"SQLAlchemy engine created for key '{connection_key}' connected to {final_host}:{final_port_str}/{final_db_name}")
    except Exception as e:
        logger.error(f"Failed to create SQLAlchemy engine for key '{connection_key}': {e}", exc_info=True)
        raise # Re-raise the exception after logging

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
        logger.debug(f"No engine provided for get_session (key: {connection_key}), calling get_engine.")
        engine = get_engine(connection_key=connection_key)

    # Get or create session factory
    if connection_key not in _session_factories:
        logger.debug(f"Creating new session factory for key: {connection_key}")
        _session_factories[connection_key] = get_session_factory(engine)
    else:
        logger.debug(f"Using existing session factory for key: {connection_key}")

    # Create and return session
    logger.debug(f"Creating new session for key: {connection_key}")
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
            logger.debug(f"Yielding session (key: {connection_key}, attempt: {retries + 1})")
            yield session
            session.commit()
            logger.debug(f"Session committed (key: {connection_key})")
            break
        except OperationalError as e:
            logger.warning(f"OperationalError during session (key: {connection_key}, attempt: {retries + 1}/{max_retries}): {e}")
            session.rollback()
            if retries < max_retries:
                retries += 1
                logger.info(f"Retrying in {retry_delay}s... (key: {connection_key})")
                time.sleep(retry_delay)
                continue
            else:
                logger.error(f"Database operational error after {max_retries} retries (key: {connection_key}): {e}", exc_info=True)
                raise SQLAlchemyError(f"Database error after {max_retries} retries: {e}")
        except Exception as e:
            logger.error(f"Exception during session (key: {connection_key}): {e}", exc_info=True)
            session.rollback()
            raise
        finally:
            logger.debug(f"Closing session (key: {connection_key})")
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
        logger.info(f"Creating database tables (key: {connection_key}) if they do not exist.")
        try:
            Base.metadata.create_all(engine)
            logger.info(f"Database tables checked/created successfully (key: {connection_key}).")
        except Exception as e:
            logger.error(f"Error creating database tables (key: {connection_key}): {e}", exc_info=True)
            raise
