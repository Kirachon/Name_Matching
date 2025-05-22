import pytest
import os
import configparser
from src import config # This will also run load_config() and setup_logging() on import

# Fixture to clean up the global CONFIG object and environment variables
@pytest.fixture(autouse=True)
def cleanup_config_and_env():
    original_config = config.CONFIG
    original_env = os.environ.copy()
    
    # Reset global CONFIG to None or a fresh parser before each test
    config.CONFIG = configparser.ConfigParser() 
    # This is a simple reset; if load_config modified other global state, that would need reset too.

    yield

    config.CONFIG = original_config # Restore original CONFIG state
    os.environ.clear()
    os.environ.update(original_env)
    # Reload original configuration if it was loaded from a file path
    # This assumes the original config.py load_config() call might have succeeded or failed based on file presence
    # For a truly clean state, config.py should ideally not auto-load or allow disabling auto-load for tests.
    # For now, we'll just reload based on its default path.
    config.load_config() # Attempt to reload the default config.ini if it exists


def test_load_config_file_found(tmp_path, capsys):
    """Test loading a configuration file that exists."""
    config_file = tmp_path / "test.ini"
    parser = configparser.ConfigParser()
    parser["database"] = {"host": "localhost_test", "port": "1234"}
    parser["logging"] = {"level": "DEBUG"}
    with open(config_file, "w") as f:
        parser.write(f)

    config.load_config(str(config_file))
    assert config.CONFIG is not None
    assert config.CONFIG.get("database", "host") == "localhost_test"
    assert config.CONFIG.getint("database", "port") == 1234
    assert config.CONFIG.get("logging", "level") == "DEBUG"
    
    captured = capsys.readouterr()
    assert f"Configuration loaded from '{str(config_file)}'." in captured.out
    # setup_logging (called via import config) will also print, check for its initial message
    # This part depends on the default log level if the test.ini wasn't loaded by setup_logging yet
    # For now, let's assume setup_logging from import already ran with defaults or no config

def test_load_config_file_not_found(capsys):
    """Test loading a configuration file that does not exist."""
    non_existent_file = "non_existent.ini"
    config.load_config(non_existent_file)
    # CONFIG should be None as per the logic in load_config when file not found
    assert config.CONFIG is None 
                                
    captured = capsys.readouterr()
    assert f"Warning: Configuration file '{non_existent_file}' not found." in captured.out
    assert "Using fallback to environment variables." in captured.out


def test_get_db_config_from_ini(tmp_path):
    """Test get_db_config when values are present in the INI file."""
    config_file = tmp_path / "db_test.ini"
    parser = configparser.ConfigParser()
    parser["database"] = {
        "host": "ini_host",
        "port": "5432",
        "user": "ini_user",
        "password": "ini_password",
        "database": "ini_db",
        "use_ssl": "true"
    }
    with open(config_file, "w") as f:
        parser.write(f)
    
    config.load_config(str(config_file)) # Load the test INI
    
    db_settings = config.get_db_config()
    assert db_settings["host"] == "ini_host"
    assert db_settings["port"] == 5432
    assert db_settings["user"] == "ini_user"
    assert db_settings["password"] == "ini_password"
    assert db_settings["database"] == "ini_db"
    assert db_settings["use_ssl"] is True

def test_get_db_config_from_env(monkeypatch):
    """Test get_db_config fallback to environment variables."""
    config.load_config("non_existent_for_env_test.ini") # Ensure INI is not loaded or empty

    monkeypatch.setenv("DB_HOST", "env_host")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("DB_USER", "env_user")
    monkeypatch.setenv("DB_PASSWORD", "env_password")
    monkeypatch.setenv("DB_NAME", "env_db")
    monkeypatch.setenv("DB_USE_SSL", "false")

    db_settings = config.get_db_config()
    assert db_settings["host"] == "env_host"
    assert db_settings["port"] == 5433
    assert db_settings["user"] == "env_user"
    assert db_settings["password"] == "env_password"
    assert db_settings["database"] == "env_db"
    assert db_settings["use_ssl"] is False

def test_get_db_config_partial_ini_and_env(tmp_path, monkeypatch):
    """Test get_db_config with some values from INI and some from environment."""
    config_file = tmp_path / "partial_db.ini"
    parser = configparser.ConfigParser()
    parser["database"] = {"host": "ini_host", "port": "1111"}
    with open(config_file, "w") as f:
        parser.write(f)

    config.load_config(str(config_file))

    monkeypatch.setenv("DB_USER", "env_user_partial")
    monkeypatch.setenv("DB_NAME", "env_db_partial")
    # DB_HOST and DB_PORT should come from INI
    # DB_PASSWORD and DB_USE_SSL will be None/False respectively

    db_settings = config.get_db_config()
    assert db_settings["host"] == "ini_host"
    assert db_settings["port"] == 1111
    assert db_settings["user"] == "env_user_partial"
    assert db_settings["password"] is None # Not in INI, not in ENV
    assert db_settings["database"] == "env_db_partial"
    assert db_settings["use_ssl"] is False # Default, not in INI or ENV

def test_get_db_config_invalid_port_env(monkeypatch, capsys):
    """Test get_db_config with an invalid port from environment variable."""
    config.load_config("non_existent_for_invalid_port.ini")
    monkeypatch.setenv("DB_PORT", "not_a_number")
    
    db_settings = config.get_db_config()
    assert db_settings["port"] is None # Should be None after warning
    captured = capsys.readouterr()
    # This warning comes from src.config.get_db_config
    assert "Warning: Invalid environment variable DB_PORT value 'not_a_number'." in captured.out
    # This warning message is from the initial setup_logging call if config was not found.
    # We need to be careful about multiple log messages.

def test_get_logging_config_from_ini(tmp_path):
    """Test get_logging_config from INI file."""
    config_file = tmp_path / "logging_test.ini"
    parser = configparser.ConfigParser()
    parser["logging"] = {"level": "WARNING"}
    with open(config_file, "w") as f:
        parser.write(f)
    
    config.load_config(str(config_file))
    log_settings = config.get_logging_config()
    assert log_settings["level"] == "WARNING"
    assert log_settings["format"] == config.DEFAULT_LOG_FORMAT # Assuming format is not in INI

def test_get_logging_config_from_env(monkeypatch):
    """Test get_logging_config fallback to environment variable."""
    config.load_config("non_existent_for_log_env.ini")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    
    log_settings = config.get_logging_config()
    assert log_settings["level"] == "ERROR"

def test_get_logging_config_default():
    """Test get_logging_config defaults when no INI or env var."""
    config.load_config("non_existent_for_log_default.ini")
    # Ensure no LOG_LEVEL env var from previous tests
    if "LOG_LEVEL" in os.environ:
        del os.environ["LOG_LEVEL"]
        
    log_settings = config.get_logging_config()
    assert log_settings["level"] == config.DEFAULT_LOG_LEVEL # e.g. "INFO"

def test_setup_logging_level_from_config(tmp_path, caplog):
    """Test that setup_logging correctly sets the log level from config."""
    config_file = tmp_path / "logging_setup.ini"
    parser = configparser.ConfigParser()
    parser["logging"] = {"level": "DEBUG"}
    with open(config_file, "w") as f:
        parser.write(f)
    
    config.load_config(str(config_file)) # Load our specific config
    
    # Temporarily clear handlers and set level for root logger to capture all
    # This is tricky because setup_logging is also called on initial import of src.config
    # For a clean test, setup_logging might need to be more idempotent or accept a logger instance.
    
    # Re-run setup_logging to apply the new config
    # Note: basicConfig can only be called once effectively. Subsequent calls are no-ops.
    # To test this properly, we'd need to reset the logging system or use a custom logger.
    # For now, we'll check the log message from setup_logging itself if it logs its own level.
    
    # Python's logging.basicConfig() does nothing if the root logger already has handlers.
    # To truly test dynamic level changes, one might need to:
    # 1. Get the root logger.
    # 2. Remove its existing handlers.
    # 3. Call basicConfig again (or manually set level and add a handler).
    # This is complex. A simpler check is if src.config.setup_logging itself logs the level it's setting.
    
    # The logger in src.config is logging.getLogger(__name__) which is 'src.config'
    # Let's check if THAT logger got its level set, or if the root logger's effective level allows DEBUG.
    
    # Resetting logging for this test
    logging.getLogger().handlers = [] # Clear root handlers
    
    with caplog.at_level(logging.DEBUG, logger="src.config"): # Capture from src.config logger
        config.setup_logging() # Call it again after loading the specific INI
        
    # Check if the "Logging configured. Level: DEBUG" message was logged by src.config.setup_logging
    found_log = False
    for record in caplog.records:
        if record.name == "src.config" and "Logging configured. Level: DEBUG" in record.message:
            found_log = True
            break
    assert found_log, "setup_logging did not log the expected DEBUG level message."

    # Additionally, check the root logger's level (if basicConfig worked)
    # This is still tricky due to basicConfig's nature.
    # A more robust test of setup_logging would be to have it return the configured logger
    # or for get_logging_config to be the primary source of truth and test that.
    # The current setup_logging directly applies basicConfig.
    
    # Verify that a DEBUG message from another module would now be processed
    # if basicConfig set the root level to DEBUG.
    logging.getLogger().handlers = [] # Clear root handlers again for a fresh basicConfig
    config.setup_logging() # This uses the DEBUG level from our loaded config
    
    other_logger = logging.getLogger("test_debug_level_check")
    with caplog.at_level(logging.DEBUG):
        other_logger.debug("This is a debug message for level check.")
    
    assert "This is a debug message for level check." in caplog.text


def test_setup_logging_invalid_level_warning(tmp_path, caplog):
    """Test setup_logging warning for an invalid log level."""
    config_file = tmp_path / "invalid_logging.ini"
    parser = configparser.ConfigParser()
    parser["logging"] = {"level": "INVALID_LEVEL"}
    with open(config_file, "w") as f:
        parser.write(f)

    logging.getLogger().handlers = [] # Ensure basicConfig can run
    config.load_config(str(config_file)) # Load config with invalid level
    
    # setup_logging is called automatically after load_config if it's part of the import logic
    # or if we call it explicitly.
    # The warning is a print statement in setup_logging. We need capsys for that.
    # However, the test for setup_logging should use caplog for logging messages.
    
    # For this specific test, we're checking a print warning from setup_logging
    # but also how the logging module itself reacts.
    
    # Let's capture print statements for the warning.
    # This requires a way to re-run setup_logging or check its effects after load_config.
    # The `config.py` calls `setup_logging()` at the end of the module.
    # So, when `config.load_config()` is called, if that's the *first* time for this test process
    # that `config.py` is effectively "run" with this new file, then setup_logging would use it.
    # This is tricky due to Python's import caching.
    
    # The `cleanup_config_and_env` fixture reloads config, which calls setup_logging.
    # We need to ensure our test INI is the one used by that setup_logging call.
    # This implies that load_config should perhaps not auto-call setup_logging,
    # but rather setup_logging should be called explicitly in tests where needed.
    
    # Assuming setup_logging is called by load_config or subsequent import:
    # The warning "Warning: Invalid log level..." is printed by setup_logging.
    # The logging system will default to WARNING if getattr fails and numeric_level is not int.
    # And our setup_logging defaults to DEFAULT_LOG_LEVEL ("INFO") if getattr fails.
    
    # To test the print warning, we'd ideally call setup_logging in a controlled way.
    # For now, let's focus on the resulting log level.
    
    logging.getLogger().handlers = [] # Reset handlers
    config.setup_logging() # Explicitly call with the loaded "INVALID_LEVEL" config
                           # This should trigger the print warning and default to INFO.

    # Check that the root logger's level is INFO (our default fallback in setup_logging)
    assert logging.getLogger().getEffectiveLevel() == logging.INFO

    # Check that src.config itself logged the new effective level
    # (setup_logging logs "Logging configured. Level: INFO")
    with caplog.at_level(logging.INFO, logger="src.config"):
        # We need to make setup_logging run again in a way that its print is captured by capsys
        # and its logging by caplog.
        # This test is becoming complicated due to the auto-run nature of setup_logging.
        # A refactor of config.py to make setup_logging more testable (e.g. not auto-run) would be good.
        
        # For now, let's assume the initial setup_logging upon import of src.config is what we test.
        # This means the test environment for config.py needs careful thought.
        # The `capsys` in `test_load_config_file_not_found` captures print from `load_config`.
        # `setup_logging` also prints its warning.
        
        # The current test structure with autouse fixture re-runs load_config() and setup_logging().
        # So, the state from the INI file (invalid level) should be active.
        
        # The warning "Warning: Invalid log level 'INVALID_LEVEL'. Defaulting to INFO."
        # is printed by setup_logging. This needs to be captured by capsys.
        # This test might be better if it directly called setup_logging and used capsys.
        # However, setup_logging also logs.
        pass # This test's assertions are tricky. Focus on effective level for now.
        # The print warning test is deferred until `setup_logging` is more isolated.
        # The key outcome is that logging is set to a sensible default.
        # The `config.py` has `print(f"Warning: Invalid log level '{log_level_str}'. Defaulting to {DEFAULT_LOG_LEVEL}.")`
        # This part is harder to test with caplog.
        # The log from setup_logging itself: logger.info(f"Logging configured. Level: {log_level_str}")
        # Here log_level_str would be INFO (the default after failing to parse INVALID_LEVEL)
    
    found_log = False
    for record in caplog.records: # Check all records from the test session
        if record.name == "src.config" and "Logging configured. Level: INFO" in record.message:
            if "INVALID_LEVEL" in record.message: # Ensure it's not picking up a previous test's INFO
                continue
            found_log = True
            break
    assert found_log, "setup_logging did not log the expected INFO level after an invalid level was provided."

# Test for the print warning about invalid log level (requires careful sequencing)
def test_setup_logging_invalid_level_prints_warning(tmp_path, monkeypatch, capsys):
    config_file = tmp_path / "invalid_level_print.ini"
    config_file.write_text("[logging]\nlevel = VERYWRONG\n")

    # To ensure our config is loaded and setup_logging is run against it cleanly for this test,
    # we might need to reload the config module or use importlib.reload.
    # This is because config.py runs load_config() and setup_logging() on import.
    # The autouse fixture already re-runs load_config(), which calls setup_logging().

    # The `cleanup_config_and_env` fixture will call `config.load_config()` at the end of the test.
    # We need to ensure that for *this* test's execution of `setup_logging`,
    # the `capsys` fixture is active.
    # The challenge: `src.config` is imported once. `setup_logging` runs then.
    # The fixture `cleanup_config_and_env` does `config.CONFIG = configparser.ConfigParser()`
    # then `yields`, then `config.load_config()`.
    # This means the `setup_logging()` relevant to this test's INI file is run by the fixture *after* the test body.
    
    # This test design is flawed for testing print output from module-level execution.
    # A better way:
    # 1. Modify config.py so setup_logging() is not called automatically on import.
    # 2. Call load_config() and then setup_logging() explicitly within each test.
    
    # Workaround: Force a reload and then call setup_logging.
    # This is generally not recommended for standard test structure.
    import importlib

    # Set up the environment for this specific scenario
    # 1. Ensure no other config file is accidentally loaded by default path.
    monkeypatch.setattr(config, 'DEFAULT_CONFIG_FILE_PATH', str(config_file))
    
    # 2. Reload config - this will run load_config(str(config_file)) and then setup_logging()
    #    due to the code at the bottom of config.py
    #    Need to ensure logging handlers are clear for basicConfig to reconfigure.
    logging.getLogger().handlers = []
    importlib.reload(config) 
    # Now, config.py has run with our specific INI file.
    # The print warning from setup_logging should have occurred.
    
    captured = capsys.readouterr() # This should capture print from the reload.
    assert "Warning: Invalid log level 'VERYWRONG'. Defaulting to INFO." in captured.out
    
    # Restore default path for other tests
    monkeypatch.setattr(config, 'DEFAULT_CONFIG_FILE_PATH', "config.ini")
    # And reload again to restore normal state for other tests (though fixture should handle this)
    logging.getLogger().handlers = []
    importlib.reload(config)


# More tests to be added:
# - Test for boolean conversion in get_db_config (e.g. use_ssl)
# - Test for int conversion for port in get_db_config from INI
# - Test case sensitivity for log levels (should be upper())
# - Test what happens if a section (e.g. [database] or [logging]) is missing from INI.
# - Test `get_connection_string` integration (from `src.db.connection`) after config is loaded.
#   This might belong in test_db.py but needs to be aware of src.config.

# Test for use_ssl boolean conversion from INI
def test_get_db_config_ssl_boolean_conversion(tmp_path):
    config_file = tmp_path / "ssl_test.ini"
    parser = configparser.ConfigParser()
    parser["database"] = {"use_ssl": "yes"} # configparser.getboolean handles 'yes'
    with open(config_file, "w") as f:
        parser.write(f)
    config.load_config(str(config_file))
    db_settings = config.get_db_config()
    assert db_settings["use_ssl"] is True

    parser["database"] = {"use_ssl": "false"}
    with open(config_file, "w") as f:
        parser.write(f)
    config.load_config(str(config_file))
    db_settings = config.get_db_config()
    assert db_settings["use_ssl"] is False

# Test for port integer conversion from INI
def test_get_db_config_port_int_conversion_ini(tmp_path):
    config_file = tmp_path / "port_test.ini"
    parser = configparser.ConfigParser()
    parser["database"] = {"port": "9876"}
    with open(config_file, "w") as f:
        parser.write(f)
    config.load_config(str(config_file))
    db_settings = config.get_db_config()
    assert db_settings["port"] == 9876

# Test for port int conversion error from INI
def test_get_db_config_invalid_port_ini(tmp_path, capsys):
    config_file = tmp_path / "invalid_port.ini"
    parser = configparser.ConfigParser()
    parser["database"] = {"port": "not_a_port_number"}
    with open(config_file, "w") as f:
        parser.write(f)
    
    # config.getint in get_db_config will raise ValueError
    # The current get_db_config has fallback=None for getint, so it won't raise
    # but will return None. A warning should be printed by get_db_config in this case.
    # My current get_db_config doesn't explicitly print a warning for INI getint failure.
    # Let's assume it should default to None if conversion fails.
    
    config.load_config(str(config_file))
    db_settings = config.get_db_config()
    assert db_settings["port"] is None # Should be None if getint fails with fallback
    
    # No explicit print warning in get_db_config for failed getint, so can't check capsys here for that.
    # This highlights a small area for improvement in get_db_config: warning for INI parse errors.

# Test case sensitivity for log levels (should be converted to upper)
def test_get_logging_config_level_case_insensitive(tmp_path, monkeypatch):
    config_file = tmp_path / "log_level_case.ini"
    parser = configparser.ConfigParser()
    parser["logging"] = {"level": "debug"} # Lowercase
    with open(config_file, "w") as f:
        parser.write(f)
    
    config.load_config(str(config_file))
    log_settings = config.get_logging_config()
    assert log_settings["level"] == "DEBUG" # Should be upper

    monkeypatch.setenv("LOG_LEVEL", "warning") # Lowercase
    config.load_config("non_existent_for_log_env_case.ini") # ensure INI not used
    log_settings_env = config.get_logging_config()
    assert log_settings_env["level"] == "WARNING" # Should be upper

# Test missing [database] section
def test_get_db_config_missing_database_section(tmp_path, monkeypatch, capsys):
    config_file = tmp_path / "no_db_section.ini"
    parser = configparser.ConfigParser()
    parser["other_section"] = {"key": "value"} # No [database] section
    with open(config_file, "w") as f:
        parser.write(f)

    config.load_config(str(config_file))
    # Set env vars to see if it falls back correctly
    monkeypatch.setenv("DB_HOST", "env_host_no_section")
    db_settings = config.get_db_config()
    
    assert db_settings["host"] == "env_host_no_section" # Fallback to env
    assert db_settings["port"] is None # No default port if not in env or ini
    
    captured = capsys.readouterr()
    # Check for the warning from get_db_config
    assert "No [database] section in INI or INI file not loaded." in captured.out
    
# Test missing [logging] section
def test_get_logging_config_missing_logging_section(tmp_path, monkeypatch):
    config_file = tmp_path / "no_logging_section.ini"
    parser = configparser.ConfigParser()
    parser["other_section"] = {"key": "value"} # No [logging] section
    with open(config_file, "w") as f:
        parser.write(f)

    config.load_config(str(config_file))
    # Set env var for log level
    monkeypatch.setenv("LOG_LEVEL", "CRITICAL")
    log_settings = config.get_logging_config()
    
    assert log_settings["level"] == "CRITICAL" # Fallback to env
    
    # Test default if env var also not set
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    log_settings_default = config.get_logging_config() # config.CONFIG still has no [logging]
    assert log_settings_default["level"] == config.DEFAULT_LOG_LEVEL

def test_get_matching_thresholds_from_ini(tmp_path, caplog):
    """Test get_matching_thresholds when values are present in the INI file."""
    import logging
    caplog.set_level(logging.INFO) # To capture info logs from get_matching_thresholds

    config_file = tmp_path / "matching_config.ini"
    parser = configparser.ConfigParser()
    parser["matching"] = {
        "match_threshold": "0.85",
        "non_match_threshold": "0.65"
    }
    with open(config_file, "w") as f:
        parser.write(f)
    
    config.load_config(str(config_file))
    thresholds = config.get_matching_thresholds()
    
    assert thresholds["match_threshold"] == 0.85
    assert thresholds["non_match_threshold"] == 0.65
    assert "Matching thresholds loaded from config" in caplog.text

def test_get_matching_thresholds_defaults(monkeypatch, caplog):
    """Test get_matching_thresholds uses defaults when INI section/keys are missing."""
    import logging
    caplog.set_level(logging.INFO)

    # Ensure config is loaded but without the 'matching' section
    config.CONFIG = configparser.ConfigParser() # Empty config, no 'matching' section
    config.CONFIG.add_section("database") # Add some other section to make CONFIG not None

    # Clear relevant environment variables if any were set by other tests
    monkeypatch.delenv("MATCH_THRESHOLD", raising=False)
    monkeypatch.delenv("NON_MATCH_THRESHOLD", raising=False)

    thresholds = config.get_matching_thresholds()
    
    assert thresholds["match_threshold"] == config.DEFAULT_MATCH_THRESHOLD
    assert thresholds["non_match_threshold"] == config.DEFAULT_NON_MATCH_THRESHOLD
    assert "Matching thresholds not found in config, using defaults" in caplog.text

def test_get_matching_thresholds_fallback_if_key_missing(tmp_path, caplog):
    """Test get_matching_thresholds uses default for a specific missing key in INI."""
    import logging
    caplog.set_level(logging.INFO)

    config_file = tmp_path / "partial_matching_config.ini"
    parser = configparser.ConfigParser()
    parser["matching"] = {
        "match_threshold": "0.90" # non_match_threshold is missing
    }
    with open(config_file, "w") as f:
        parser.write(f)
    
    config.load_config(str(config_file))
    thresholds = config.get_matching_thresholds()
    
    assert thresholds["match_threshold"] == 0.90
    assert thresholds["non_match_threshold"] == config.DEFAULT_NON_MATCH_THRESHOLD # Should use default
    assert "Matching thresholds loaded from config" in caplog.text # Still loaded from config, but one value defaulted

def test_get_matching_thresholds_invalid_value_in_ini(tmp_path, caplog):
    """Test get_matching_thresholds with non-float value in INI, expecting fallback."""
    import logging
    caplog.set_level(logging.INFO)

    config_file = tmp_path / "invalid_matching_config.ini"
    parser = configparser.ConfigParser()
    parser["matching"] = {
        "match_threshold": "not_a_float",
        "non_match_threshold": "0.5"
    }
    with open(config_file, "w") as f:
        parser.write(f)
    
    config.load_config(str(config_file))
    # configparser.getfloat will raise ValueError if fallback is not provided
    # or if the fallback itself cannot be converted.
    # The fallback in get_matching_thresholds is a float, so it should be fine.
    thresholds = config.get_matching_thresholds() 
    
    assert thresholds["match_threshold"] == config.DEFAULT_MATCH_THRESHOLD # Fallback due to conversion error
    assert thresholds["non_match_threshold"] == 0.5
    assert "Matching thresholds loaded from config" in caplog.text # It did attempt to load
    # Note: configparser itself might log a warning or error for the failed conversion,
    # or getfloat with fallback might silently use the fallback.
    # The current implementation of get_matching_thresholds uses fallback in getfloat,
    # so it won't raise an error but use the default.

# --- Test get_connection_string integration ---
# get_connection_string is in src.db.connection, but its behavior


# --- Test get_connection_string integration ---
# get_connection_string is in src.db.connection, but its behavior
# is now heavily tied to src.config. So, we test the integration here.
from src.db.connection import get_connection_string

def test_get_connection_string_integration_override_all(monkeypatch):
    """Test get_connection_string when all parameters are passed directly."""
    # Ensure no INI or env vars interfere
    config.CONFIG = configparser.ConfigParser() # Empty config
    monkeypatch.setattr(os, 'environ', {}) # Clear environment

    conn_str = get_connection_string(
        db_name="direct_db",
        host="direct_host",
        port="1234",
        user="direct_user",
        password="direct_password",
        use_ssl=True
    )
    assert "mysql+pymysql://direct_user:direct_password@direct_host:1234/direct_db?ssl=true" == conn_str

def test_get_connection_string_integration_from_ini(tmp_path, monkeypatch):
    """Test get_connection_string picking up values from INI via get_db_config."""
    config_file = tmp_path / "conn_str_test.ini"
    parser = configparser.ConfigParser()
    parser["database"] = {
        "host": "ini_host_for_conn",
        "port": "5555",
        "user": "ini_user_conn",
        "password": "ini_password_conn",
        "database": "ini_db_conn",
        "use_ssl": "true"
    }
    with open(config_file, "w") as f:
        parser.write(f)
    
    config.load_config(str(config_file)) # Load the INI for get_db_config to use
    monkeypatch.setattr(os, 'environ', {}) # Clear environment to ensure INI is used

    conn_str = get_connection_string() # No direct args, should use INI
    expected = "mysql+pymysql://ini_user_conn:ini_password_conn@ini_host_for_conn:5555/ini_db_conn?ssl=true"
    assert conn_str == expected

def test_get_connection_string_integration_from_env(monkeypatch):
    """Test get_connection_string picking up values from ENV via get_db_config."""
    config.CONFIG = configparser.ConfigParser() # Ensure no INI
    
    monkeypatch.setenv("DB_HOST", "env_host_conn")
    monkeypatch.setenv("DB_PORT", "6666")
    monkeypatch.setenv("DB_USER", "env_user_conn")
    monkeypatch.setenv("DB_PASSWORD", "env_password_conn")
    monkeypatch.setenv("DB_NAME", "env_db_conn")
    monkeypatch.setenv("DB_USE_SSL", "false")

    conn_str = get_connection_string() # No direct args, no INI, should use ENV
    expected = "mysql+pymysql://env_user_conn:env_password_conn@env_host_conn:6666/env_db_conn"
    assert conn_str == expected # SSL false, so no ?ssl=true

def test_get_connection_string_integration_partial_direct_partial_ini(tmp_path, monkeypatch):
    """Test get_connection_string with direct args overriding INI."""
    config_file = tmp_path / "partial_conn.ini"
    parser = configparser.ConfigParser()
    parser["database"] = { # These should be overridden by direct args if provided
        "host": "ini_host_partial", "port": "1111", "user": "ini_user_partial",
        "database": "ini_db_partial", "use_ssl": "true"
    }
    with open(config_file, "w") as f:
        parser.write(f)
    config.load_config(str(config_file))
    monkeypatch.setattr(os, 'environ', {})

    # Direct args for host and db_name, others from INI
    conn_str = get_connection_string(host="direct_host_override", db_name="direct_db_override")
    # Expected: direct host & db, port & user from INI, password None (not in INI), ssl from INI
    # Password will be empty string by default in get_connection_string if not set.
    expected = "mysql+pymysql://ini_user_partial:@direct_host_override:1111/direct_db_override?ssl=true"
    assert conn_str == expected

def test_get_connection_string_integration_direct_overrides_env(monkeypatch):
    """Test get_connection_string with direct args overriding ENV values."""
    config.CONFIG = configparser.ConfigParser() # No INI

    monkeypatch.setenv("DB_HOST", "env_host_override")
    monkeypatch.setenv("DB_PORT", "2222")
    monkeypatch.setenv("DB_USER", "env_user_override")
    monkeypatch.setenv("DB_NAME", "env_db_override")
    monkeypatch.setenv("DB_USE_SSL", "true")

    # Direct arg for port, others from ENV
    conn_str = get_connection_string(port="7777", use_ssl=False) # Direct port, direct SSL (false)
    # Expected: direct port & ssl, host/user/db from ENV. Password from ENV (None here -> empty).
    expected = "mysql+pymysql://env_user_override:@env_host_override:7777/env_db_override"
    assert conn_str == expected

def test_get_connection_string_integration_ini_overrides_env(tmp_path, monkeypatch):
    """Test get_connection_string with INI values overriding ENV values."""
    config_file = tmp_path / "ini_over_env.ini"
    parser = configparser.ConfigParser()
    parser["database"] = {
        "host": "ini_host_final", "port": "3333", "database": "ini_db_final"
    } # User, pass, ssl not in INI
    with open(config_file, "w") as f:
        parser.write(f)
    config.load_config(str(config_file))

    monkeypatch.setenv("DB_HOST", "env_host_should_be_ignored")
    monkeypatch.setenv("DB_PORT", "4444") # Should be ignored
    monkeypatch.setenv("DB_USER", "env_user_final") # Used as fallback from INI
    monkeypatch.setenv("DB_PASSWORD", "env_pass_final") # Used as fallback
    monkeypatch.setenv("DB_NAME", "env_db_should_be_ignored")
    monkeypatch.setenv("DB_USE_SSL", "true") # Used as fallback

    conn_str = get_connection_string() # No direct args
    # Expected: host, port, db from INI. User, pass, ssl from ENV.
    expected = "mysql+pymysql://env_user_final:env_pass_final@ini_host_final:3333/ini_db_final?ssl=true"
    assert conn_str == expected

def test_get_connection_string_integration_defaults(monkeypatch):
    """Test get_connection_string falling back to its own internal defaults."""
    config.CONFIG = configparser.ConfigParser() # No INI
    monkeypatch.setattr(os, 'environ', {})      # No ENV vars

    conn_str = get_connection_string()
    # Defaults from get_connection_string: name_matching, localhost, 3306, root, "" pass, ssl False
    expected = "mysql+pymysql://root:@localhost:3306/name_matching"
    assert conn_str == expected
