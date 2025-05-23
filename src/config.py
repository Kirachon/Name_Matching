"""Enhanced configuration module for the Name Matching application.

This module handles loading configuration from multiple sources:
- config.ini files
- environment variables
- command line arguments
- remote configuration services
"""

import configparser
import os
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

CONFIG = None
DEFAULT_CONFIG_FILE_PATH = "config.ini"

def load_config(config_file_path: str = DEFAULT_CONFIG_FILE_PATH) -> None:
    """
    Loads configuration settings from an INI file.

    The loaded configuration is stored in a global variable `CONFIG`.
    If the file is not found, a warning is printed, and CONFIG remains None.

    Args:
        config_file_path (str): Path to the configuration file.
                                Defaults to "config.ini".
    """
    global CONFIG
    CONFIG = configparser.ConfigParser()
    if os.path.exists(config_file_path):
        CONFIG.read(config_file_path)
        print(f"Configuration loaded from '{config_file_path}'.")
    else:
        CONFIG = None # Ensure CONFIG is None if file not found
        print(f"Warning: Configuration file '{config_file_path}' not found. Using fallback to environment variables.")

def get_db_config() -> dict:
    """
    Returns a dictionary with database connection parameters.

    Reads from the loaded INI configuration first (from the [database] section).
    Falls back to environment variables (DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME)
    for any parameters not found in the INI file.

    Returns:
        dict: Database configuration with keys: 'host', 'port', 'user', 'password', 'database'.
              Values might be None if not found in config or environment.
    """
    db_config = {
        "host": None,
        "port": None,
        "user": None,
        "password": None,
        "database": None,
        "use_ssl": False, # Default to False
    }

    ini_config_found = False
    if CONFIG and CONFIG.has_section("database"):
        ini_config_found = True
        db_config["host"] = CONFIG.get("database", "host", fallback=None)
        db_config["port"] = CONFIG.getint("database", "port", fallback=None)
        db_config["user"] = CONFIG.get("database", "user", fallback=None)
        db_config["password"] = CONFIG.get("database", "password", fallback=None)
        db_config["database"] = CONFIG.get("database", "database", fallback=None)
        db_config["use_ssl"] = CONFIG.getboolean("database", "use_ssl", fallback=False)

    # Fallback to environment variables
    db_config["host"] = db_config["host"] or os.getenv("DB_HOST")
    env_db_port = os.getenv("DB_PORT")
    if db_config["port"] is None and env_db_port is not None:
        try:
            db_config["port"] = int(env_db_port)
        except ValueError:
            print(f"Warning: Invalid environment variable DB_PORT value '{env_db_port}'. Using None for port.")
            db_config["port"] = None
    elif db_config["port"] is None: # If still None (not in INI, not in ENV or invalid ENV)
        db_config["port"] = None # Explicitly set to None if not found or invalid

    db_config["user"] = db_config["user"] or os.getenv("DB_USER")
    db_config["password"] = db_config["password"] or os.getenv("DB_PASSWORD")
    db_config["database"] = db_config["database"] or os.getenv("DB_NAME") # Using DB_NAME as per previous observation

    # Handle use_ssl, ensuring it's a boolean
    if db_config["use_ssl"] is None: # If not set by INI
        env_use_ssl = os.getenv("DB_USE_SSL", "false")
        db_config["use_ssl"] = env_use_ssl.lower() == "true"

    if not ini_config_found:
        print("No [database] section in INI or INI file not loaded. Relying on environment variables for database config.")

    return db_config

def get_gpu_config() -> dict:
    """
    Returns a dictionary with GPU acceleration parameters.

    Tries to read from the loaded INI configuration first.
    If not available, falls back to environment variables.
    If neither is available, uses default values.

    Returns:
        dict: GPU configuration parameters
    """
    gpu_config = {
        "enabled": True,
        "framework": None,  # Auto-select
        "device_id": 0,
        "batch_size": 1000,
        "memory_limit_gb": 4.0,
        "fallback_threshold": 10000
    }

    # Try to read from INI configuration first
    if CONFIG and CONFIG.has_section('gpu'):
        print("Reading GPU config from INI file...")

        if CONFIG.has_option('gpu', 'enabled'):
            gpu_config["enabled"] = CONFIG.getboolean('gpu', 'enabled')
        if CONFIG.has_option('gpu', 'framework'):
            framework = CONFIG.get('gpu', 'framework')
            gpu_config["framework"] = framework if framework.lower() != 'none' else None
        if CONFIG.has_option('gpu', 'device_id'):
            gpu_config["device_id"] = CONFIG.getint('gpu', 'device_id')
        if CONFIG.has_option('gpu', 'batch_size'):
            gpu_config["batch_size"] = CONFIG.getint('gpu', 'batch_size')
        if CONFIG.has_option('gpu', 'memory_limit_gb'):
            gpu_config["memory_limit_gb"] = CONFIG.getfloat('gpu', 'memory_limit_gb')
        if CONFIG.has_option('gpu', 'fallback_threshold'):
            gpu_config["fallback_threshold"] = CONFIG.getint('gpu', 'fallback_threshold')
    else:
        print("No [gpu] section in INI or INI file not loaded. Checking environment variables for GPU config...")

        # Fallback to environment variables
        if os.getenv("GPU_ENABLED"):
            gpu_config["enabled"] = os.getenv("GPU_ENABLED").lower() in ['true', '1', 'yes']
        if os.getenv("GPU_FRAMEWORK"):
            framework = os.getenv("GPU_FRAMEWORK")
            gpu_config["framework"] = framework if framework.lower() != 'none' else None
        if os.getenv("GPU_DEVICE_ID"):
            gpu_config["device_id"] = int(os.getenv("GPU_DEVICE_ID"))
        if os.getenv("GPU_BATCH_SIZE"):
            gpu_config["batch_size"] = int(os.getenv("GPU_BATCH_SIZE"))
        if os.getenv("GPU_MEMORY_LIMIT_GB"):
            gpu_config["memory_limit_gb"] = float(os.getenv("GPU_MEMORY_LIMIT_GB"))
        if os.getenv("GPU_FALLBACK_THRESHOLD"):
            gpu_config["fallback_threshold"] = int(os.getenv("GPU_FALLBACK_THRESHOLD"))

        print("No [gpu] section in INI or INI file not loaded. Using default GPU config with environment variable overrides.")

    return gpu_config

# Attempt to load configuration when the module is imported.
# This makes the configuration available immediately if config.ini exists in the default location.
# The application can call load_config() explicitly with a different path if needed.
if __name__ == '__main__':
    # This part is for testing the module directly
    # Create a dummy config.ini for testing
    print("Running config.py directly for testing...")
    with open("test_config.ini", "w") as f:
        f.write("[database]\n")
        f.write("host = localhost_ini\n")
        f.write("port = 5432\n")
        f.write("user = user_ini\n")
        f.write("password = pass_ini\n")
        f.write("database = db_ini\n")
        f.write("use_ssl = true\n")

    print("\n--- Test Case 1: Loading from 'test_config.ini' ---")
    load_config("test_config.ini")
    print(f"CONFIG object: {CONFIG}")
    if CONFIG:
        print(f"Sections in CONFIG: {CONFIG.sections()}")
    db_settings = get_db_config()
    print(f"DB Settings from INI: {db_settings}")

    print("\n--- Test Case 2: Simulating missing config file ---")
    load_config("non_existent_config.ini") # This will print a warning
    # Set some environment variables for fallback testing
    os.environ["DB_HOST"] = "localhost_env"
    os.environ["DB_PORT"] = "5433"
    os.environ["DB_USER"] = "user_env"
    os.environ["DB_PASSWORD"] = "pass_env"
    os.environ["DB_NAME"] = "db_env"
    os.environ["DB_USE_SSL"] = "true" # Test env var for SSL
    db_settings_env = get_db_config()
    print(f"DB Settings from ENV: {db_settings_env}")
    del os.environ["DB_USE_SSL"] # Clean up ssl env var

    print("\n--- Test Case 3: Partial INI, partial ENV ---")
    with open("partial_test_config.ini", "w") as f:
        f.write("[database]\n")
        f.write("host = partial_host_ini\n")
        f.write("port = 1234\n")
    load_config("partial_test_config.ini")
    # DB_USER, DB_PASSWORD, DB_NAME should come from ENV
    db_settings_partial = get_db_config()
    print(f"DB Settings (Partial INI): {db_settings_partial}")

    # Clean up dummy files and env vars
    os.remove("test_config.ini")
    os.remove("partial_test_config.ini")
    del os.environ["DB_HOST"]
    del os.environ["DB_PORT"]
    del os.environ["DB_USER"]
    del os.environ["DB_PASSWORD"]
    del os.environ["DB_NAME"]
    # Ensure DB_USE_SSL is cleaned up if it was set for testing
    if os.getenv("DB_USE_SSL"):
        del os.environ["DB_USE_SSL"]
    print("\nTesting finished.")
else:
    import logging

# --- Logging Configuration ---
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_logging_config() -> dict:
    """
    Returns a dictionary with logging configuration parameters.

    Reads from the loaded INI configuration first (from the [logging] section).
    Falls back to environment variables (LOG_LEVEL) for any parameters not found.
    """
    log_config = {
        "level": DEFAULT_LOG_LEVEL,
        "format": DEFAULT_LOG_FORMAT, # Initially use default, can be made configurable later if needed
        # "file": None, # Placeholder for future file logging
    }

    if CONFIG and CONFIG.has_section("logging"):
        log_config["level"] = CONFIG.get("logging", "level", fallback=DEFAULT_LOG_LEVEL).upper()
        # log_config["file"] = CONFIG.get("logging", "file", fallback=None) # For future file logging

    # Fallback to environment variable for log level
    log_config["level"] = os.getenv("LOG_LEVEL", log_config["level"]).upper()

    return log_config

def setup_logging() -> None:
    """
    Sets up basic logging for the application.

    The log level and format are determined by get_logging_config().
    Logs are output to the console.
    """
    logging_config = get_logging_config()
    log_level_str = logging_config.get("level", DEFAULT_LOG_LEVEL)
    log_format = logging_config.get("format", DEFAULT_LOG_FORMAT)

    numeric_level = getattr(logging, log_level_str, None)
    if not isinstance(numeric_level, int):
        print(f"Warning: Invalid log level '{log_level_str}'. Defaulting to {DEFAULT_LOG_LEVEL}.")
        numeric_level = getattr(logging, DEFAULT_LOG_LEVEL)

    logging.basicConfig(level=numeric_level, format=log_format)
    # Mute noisy libraries if necessary, e.g.:
    # logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    # Test log message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Level: {log_level_str}")

# --- Matching Configuration ---
DEFAULT_MATCH_THRESHOLD = 0.75
DEFAULT_NON_MATCH_THRESHOLD = 0.55

def get_matching_thresholds() -> dict:
    """
    Returns a dictionary with matching threshold parameters.

    Reads from the loaded INI configuration first (from the [matching] section).
    Uses predefined defaults if not found in the INI file.
    """
    thresholds = {
        "match_threshold": DEFAULT_MATCH_THRESHOLD,
        "non_match_threshold": DEFAULT_NON_MATCH_THRESHOLD,
    }

    if CONFIG and CONFIG.has_section("matching"):
        thresholds["match_threshold"] = CONFIG.getfloat(
            "matching", "match_threshold", fallback=DEFAULT_MATCH_THRESHOLD
        )
        thresholds["non_match_threshold"] = CONFIG.getfloat(
            "matching", "non_match_threshold", fallback=DEFAULT_NON_MATCH_THRESHOLD
        )
        # Note: logger may not be available during module import
        pass
    else:
        # Note: logger may not be available during module import
        pass

    return thresholds

# Enhanced Configuration Classes

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "namematching"
    use_ssl: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    default_ttl: int = 3600
    key_prefix: str = "name_match:"
    max_connections: int = 10


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    cors_origins: list = field(default_factory=lambda: ["*"])
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30


@dataclass
class MatchingConfig:
    """Matching algorithm configuration."""
    match_threshold: float = 0.75
    non_match_threshold: float = 0.55
    enable_gpu: bool = False
    enable_caching: bool = True
    cache_ttl: int = 3600
    max_batch_size: int = 10000
    default_similarity_function: str = "jaro_winkler"
    component_weights: Dict[str, float] = field(default_factory=lambda: {
        "first_name": 0.3,
        "middle_name": 0.2,
        "last_name": 0.3,
        "birthdate": 0.1,
        "geography": 0.1
    })


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enable_prometheus: bool = True
    prometheus_port: int = 8001
    enable_health_checks: bool = True
    health_check_interval: int = 30
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5


@dataclass
class GPUConfig:
    """GPU configuration."""
    enabled: bool = False
    device_id: int = 0
    memory_limit: Optional[int] = None
    batch_size: int = 1000
    enable_mixed_precision: bool = False


@dataclass
class AppConfig:
    """Main application configuration."""
    environment: str = "development"
    debug: bool = False
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api: APIConfig = field(default_factory=APIConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create AppConfig from dictionary."""
        config = cls()

        # Update database config
        if 'database' in config_dict:
            db_config = config_dict['database']
            for key, value in db_config.items():
                if hasattr(config.database, key):
                    setattr(config.database, key, value)

        # Update redis config
        if 'redis' in config_dict:
            redis_config = config_dict['redis']
            for key, value in redis_config.items():
                if hasattr(config.redis, key):
                    setattr(config.redis, key, value)

        # Update API config
        if 'api' in config_dict:
            api_config = config_dict['api']
            for key, value in api_config.items():
                if hasattr(config.api, key):
                    setattr(config.api, key, value)

        # Update matching config
        if 'matching' in config_dict:
            matching_config = config_dict['matching']
            for key, value in matching_config.items():
                if hasattr(config.matching, key):
                    setattr(config.matching, key, value)

        # Update monitoring config
        if 'monitoring' in config_dict:
            monitoring_config = config_dict['monitoring']
            for key, value in monitoring_config.items():
                if hasattr(config.monitoring, key):
                    setattr(config.monitoring, key, value)

        # Update GPU config
        if 'gpu' in config_dict:
            gpu_config = config_dict['gpu']
            for key, value in gpu_config.items():
                if hasattr(config.gpu, key):
                    setattr(config.gpu, key, value)

        # Update top-level config
        for key in ['environment', 'debug']:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        return config

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create AppConfig from environment variables."""
        config = cls()

        # Environment and debug
        config.environment = os.getenv('ENVIRONMENT', 'development')
        config.debug = os.getenv('DEBUG', 'false').lower() == 'true'

        # Database config from environment
        config.database.host = os.getenv('DB_HOST', config.database.host)
        config.database.port = int(os.getenv('DB_PORT', str(config.database.port)))
        config.database.user = os.getenv('DB_USER', config.database.user)
        config.database.password = os.getenv('DB_PASSWORD', config.database.password)
        config.database.database = os.getenv('DB_NAME', config.database.database)
        config.database.use_ssl = os.getenv('DB_USE_SSL', 'false').lower() == 'true'

        # Redis config from environment
        config.redis.host = os.getenv('REDIS_HOST', config.redis.host)
        config.redis.port = int(os.getenv('REDIS_PORT', str(config.redis.port)))
        config.redis.db = int(os.getenv('REDIS_DB', str(config.redis.db)))
        config.redis.password = os.getenv('REDIS_PASSWORD')

        # API config from environment
        config.api.host = os.getenv('API_HOST', config.api.host)
        config.api.port = int(os.getenv('API_PORT', str(config.api.port)))
        config.api.workers = int(os.getenv('API_WORKERS', str(config.api.workers)))
        config.api.jwt_secret_key = os.getenv('JWT_SECRET_KEY', config.api.jwt_secret_key)

        # Matching config from environment
        config.matching.match_threshold = float(os.getenv('MATCH_THRESHOLD', str(config.matching.match_threshold)))
        config.matching.non_match_threshold = float(os.getenv('NON_MATCH_THRESHOLD', str(config.matching.non_match_threshold)))
        config.matching.enable_gpu = os.getenv('ENABLE_GPU', 'false').lower() == 'true'
        config.matching.enable_caching = os.getenv('ENABLE_CACHING', 'true').lower() == 'true'

        # GPU config from environment
        config.gpu.enabled = os.getenv('GPU_ENABLED', 'false').lower() == 'true'
        config.gpu.device_id = int(os.getenv('GPU_DEVICE_ID', str(config.gpu.device_id)))

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert AppConfig to dictionary."""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'user': self.database.user,
                'password': self.database.password,
                'database': self.database.database,
                'use_ssl': self.database.use_ssl,
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow,
                'pool_timeout': self.database.pool_timeout,
                'pool_recycle': self.database.pool_recycle
            },
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'db': self.redis.db,
                'password': self.redis.password,
                'default_ttl': self.redis.default_ttl,
                'key_prefix': self.redis.key_prefix,
                'max_connections': self.redis.max_connections
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'workers': self.api.workers,
                'reload': self.api.reload,
                'log_level': self.api.log_level,
                'cors_origins': self.api.cors_origins,
                'jwt_secret_key': self.api.jwt_secret_key,
                'jwt_algorithm': self.api.jwt_algorithm,
                'jwt_expire_minutes': self.api.jwt_expire_minutes
            },
            'matching': {
                'match_threshold': self.matching.match_threshold,
                'non_match_threshold': self.matching.non_match_threshold,
                'enable_gpu': self.matching.enable_gpu,
                'enable_caching': self.matching.enable_caching,
                'cache_ttl': self.matching.cache_ttl,
                'max_batch_size': self.matching.max_batch_size,
                'default_similarity_function': self.matching.default_similarity_function,
                'component_weights': self.matching.component_weights
            },
            'monitoring': {
                'enable_prometheus': self.monitoring.enable_prometheus,
                'prometheus_port': self.monitoring.prometheus_port,
                'enable_health_checks': self.monitoring.enable_health_checks,
                'health_check_interval': self.monitoring.health_check_interval,
                'log_level': self.monitoring.log_level,
                'log_format': self.monitoring.log_format,
                'log_file': self.monitoring.log_file,
                'max_log_size': self.monitoring.max_log_size,
                'log_backup_count': self.monitoring.log_backup_count
            },
            'gpu': {
                'enabled': self.gpu.enabled,
                'device_id': self.gpu.device_id,
                'memory_limit': self.gpu.memory_limit,
                'batch_size': self.gpu.batch_size,
                'enable_mixed_precision': self.gpu.enable_mixed_precision
            }
        }


class ConfigManager:
    """Enhanced configuration manager."""

    def __init__(self):
        self._config: Optional[AppConfig] = None
        self._config_sources: list = []
        self.logger = logging.getLogger(__name__)

    def load_from_file(self, config_path: Union[str, Path]) -> 'ConfigManager':
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            self.logger.warning(f"Configuration file not found: {config_path}")
            return self

        try:
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
            elif config_path.suffix.lower() in ['.ini', '.cfg']:
                config_dict = self._load_from_ini(config_path)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

            if self._config is None:
                self._config = AppConfig.from_dict(config_dict)
            else:
                # Merge with existing config
                self._merge_config(config_dict)

            self._config_sources.append(f"file:{config_path}")
            self.logger.info(f"Loaded configuration from {config_path}")

        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")

        return self

    def load_from_env(self) -> 'ConfigManager':
        """Load configuration from environment variables."""
        if self._config is None:
            self._config = AppConfig.from_env()
        else:
            # Merge environment variables
            env_config = AppConfig.from_env()
            self._merge_config(env_config.to_dict())

        self._config_sources.append("environment")
        self.logger.info("Loaded configuration from environment variables")
        return self

    def _load_from_ini(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from INI file."""
        parser = configparser.ConfigParser()
        parser.read(config_path)

        config_dict = {}

        for section_name in parser.sections():
            section_dict = {}
            for key, value in parser.items(section_name):
                # Try to convert to appropriate type
                try:
                    # Try boolean
                    if value.lower() in ['true', 'false']:
                        section_dict[key] = value.lower() == 'true'
                    # Try integer
                    elif value.isdigit():
                        section_dict[key] = int(value)
                    # Try float
                    elif '.' in value and value.replace('.', '').isdigit():
                        section_dict[key] = float(value)
                    # Keep as string
                    else:
                        section_dict[key] = value
                except ValueError:
                    section_dict[key] = value

            config_dict[section_name] = section_dict

        return config_dict

    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing."""
        if self._config is None:
            self._config = AppConfig.from_dict(new_config)
        else:
            # Deep merge logic here
            merged_dict = self._config.to_dict()
            self._deep_merge(merged_dict, new_config)
            self._config = AppConfig.from_dict(merged_dict)

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        if self._config is None:
            # Load default configuration
            self.load_from_env()
        return self._config

    def save_to_file(self, config_path: Union[str, Path], format: str = 'json'):
        """Save current configuration to file."""
        if self._config is None:
            raise ValueError("No configuration loaded")

        config_path = Path(config_path)
        config_dict = self._config.to_dict()

        if format.lower() == 'json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Configuration saved to {config_path}")

    def get_sources(self) -> list:
        """Get list of configuration sources."""
        return self._config_sources.copy()


# Global configuration manager
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get or create global configuration manager."""
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager()

        # Try to load from default locations
        for config_file in ['config.json', 'config.ini', 'config.cfg']:
            if Path(config_file).exists():
                _config_manager.load_from_file(config_file)
                break

        # Always load from environment
        _config_manager.load_from_env()

    return _config_manager


def get_app_config() -> AppConfig:
    """Get application configuration."""
    return get_config_manager().get_config()


# Enhanced getter functions for backward compatibility
def get_enhanced_db_config() -> DatabaseConfig:
    """Get enhanced database configuration."""
    return get_app_config().database


def get_redis_config() -> RedisConfig:
    """Get Redis configuration."""
    return get_app_config().redis


def get_api_config() -> APIConfig:
    """Get API configuration."""
    return get_app_config().api


def get_enhanced_matching_config() -> MatchingConfig:
    """Get enhanced matching configuration."""
    return get_app_config().matching


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration."""
    return get_app_config().monitoring


def get_enhanced_gpu_config() -> GPUConfig:
    """Get enhanced GPU configuration."""
    return get_app_config().gpu


# --- Main script execution / module import actions ---

# Default load attempt when imported as a module
load_config()
# Setup logging immediately after loading config
setup_logging()
