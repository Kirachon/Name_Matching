import configparser
import os

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

# --- Main script execution / module import actions ---

    # Default load attempt when imported as a module
    load_config()
    # Setup logging immediately after loading config
    setup_logging()
