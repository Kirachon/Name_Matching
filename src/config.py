"""
Configuration module for the Name Matching application.

This module provides functionality for loading and managing configuration settings.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union

import dotenv

# Load environment variables from .env file if it exists
dotenv.load_dotenv()


class Config:
    """Configuration class for the Name Matching application."""

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration.

        Args:
            config_file: Path to a .env configuration file
        """
        # Load configuration from file if provided
        if config_file:
            dotenv.load_dotenv(config_file)

        # Database configuration
        self.db_config = {
            "host": os.environ.get("MYSQL_HOST", "localhost"),
            "port": os.environ.get("MYSQL_PORT", "3306"),
            "user": os.environ.get("MYSQL_USER", "root"),
            "password": os.environ.get("MYSQL_PASSWORD", ""),
            "database": os.environ.get("MYSQL_DATABASE", "name_matching"),
            "use_ssl": os.environ.get("MYSQL_USE_SSL", "false").lower() == "true",
        }

        # Connection pool configuration
        self.pool_config = {
            "pool_size": int(os.environ.get("DB_POOL_SIZE", "5")),
            "max_overflow": int(os.environ.get("DB_MAX_OVERFLOW", "10")),
            "pool_timeout": int(os.environ.get("DB_POOL_TIMEOUT", "30")),
            "pool_recycle": int(os.environ.get("DB_POOL_RECYCLE", "1800")),
        }

        # Name matching configuration
        self.matching_config = {
            "match_threshold": float(os.environ.get("MATCH_THRESHOLD", "0.75")),
            "non_match_threshold": float(os.environ.get("NON_MATCH_THRESHOLD", "0.55")),
            "name_weights": {
                "first_name": float(os.environ.get("WEIGHT_FIRST_NAME", "0.4")),
                "middle_name": float(os.environ.get("WEIGHT_MIDDLE_NAME", "0.2")),
                "last_name": float(os.environ.get("WEIGHT_LAST_NAME", "0.3")),
                "full_name_sorted": float(os.environ.get("WEIGHT_FULL_NAME_SORTED", "0.1")),
            },
            "additional_field_weights": {
                "birthdate": float(os.environ.get("WEIGHT_BIRTHDATE", "0.3")),
                "geography": float(os.environ.get("WEIGHT_GEOGRAPHY", "0.3")),
            },
        }

    def get_db_config(self) -> Dict:
        """
        Get the database configuration.

        Returns:
            Dictionary with database configuration
        """
        return self.db_config

    def get_pool_config(self) -> Dict:
        """
        Get the connection pool configuration.

        Returns:
            Dictionary with connection pool configuration
        """
        return self.pool_config

    def get_matching_config(self) -> Dict:
        """
        Get the name matching configuration.

        Returns:
            Dictionary with name matching configuration
        """
        return self.matching_config

    def get_connection_string(self) -> str:
        """
        Get a SQLAlchemy connection string for MySQL.

        Returns:
            SQLAlchemy connection string
        """
        db_config = self.get_db_config()
        conn_str = (
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

        # Add SSL if requested
        if db_config["use_ssl"]:
            conn_str += "?ssl=true"

        return conn_str


# Create a default configuration instance
config = Config()
