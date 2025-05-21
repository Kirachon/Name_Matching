import pytest
import subprocess
import sys
import os
from unittest.mock import patch, mock_open

# Assuming src.cli.main is the entry point function
# And that src.config and other initializations happen when src.cli is imported or main() is called.
from src import cli as name_matching_cli # To access main if needed
from src.config import DEFAULT_CONFIG_FILE_PATH # To potentially create/mock config.ini

# Helper function to run CLI commands using subprocess
# This is more robust for testing exit codes and actual executable behavior.
def run_cli_command(command_args):
    """Runs the CLI script with given arguments using subprocess."""
    # Construct the command
    # Assuming the CLI can be run as a module: python -m src.cli
    # Or directly if it has a shebang and is executable: ./src/cli.py (needs path adjustment)
    # For simplicity, let's try to find the src/cli.py path relative to tests
    
    # This path might need adjustment based on how tests are run (from repo root or tests/ dir)
    # If tests are run from repo root:
    cli_script_path = os.path.join(os.path.dirname(__file__), "..", "src", "cli.py")
    if not os.path.exists(cli_script_path):
        # If tests are run from tests/ directory (less common for `python -m pytest`)
        cli_script_path = os.path.join("..", "src", "cli.py") # This path is less likely to be right

    # Fallback, assuming it's in PYTHONPATH and can be found
    if not os.path.exists(cli_script_path) and os.path.exists("src/cli.py"):
        cli_script_path = "src/cli.py"


    if not os.path.exists(cli_script_path):
        pytest.skip(f"CLI script not found at expected path: {cli_script_path} or src/cli.py. Skipping CLI tests.")
        return None, None, -1 # Should not happen if skipped

    cmd = [sys.executable, cli_script_path] + command_args
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout, stderr, process.returncode

# --- Test Cases ---

def test_cli_help_output():
    """Test the CLI help message."""
    stdout, stderr, exit_code = run_cli_command(["--help"])
    assert exit_code == 0
    assert "Name Matching CLI" in stdout
    assert "Command to run" in stdout # from subparsers
    assert "match-names" in stdout
    assert "match-csv" in stdout
    # db support might not be available, so match-db might not be in help
    # assert "match-db" in stdout 
    assert stderr == ""

def test_cli_no_command():
    """Test CLI output when no command is given."""
    stdout, stderr, exit_code = run_cli_command([])
    assert exit_code == 0 # Argparse by default returns 0 for help / no command
    assert "Name Matching CLI" in stdout # Should print help
    assert stderr == ""

def test_cli_match_names_command_basic():
    """Test the match-names command with basic input."""
    # This command should succeed and output scores.
    # The NameMatcher and underlying functions are tested elsewhere,
    # here we primarily test CLI plumbing.
    name1 = "John Doe"
    name2 = "Jon Doe"
    stdout, stderr, exit_code = run_cli_command(["match-names", name1, name2])
    
    assert exit_code == 0
    assert stderr == "" # Expect no errors on stderr for successful run
    assert "Match score:" in stdout
    assert "Classification:" in stdout
    assert "Component scores:" in stdout
    # Further assertions could check for specific score ranges if stable/mocked.

def test_cli_match_names_command_with_thresholds():
    """Test the match-names command with custom thresholds."""
    name1 = "Jane Smith"
    name2 = "Jayne Smyth"
    stdout, stderr, exit_code = run_cli_command([
        "match-names", name1, name2,
        "--match-threshold", "0.9",
        "--non-match-threshold", "0.7"
    ])
    assert exit_code == 0
    assert stderr == ""
    assert "Match score:" in stdout 
    # Classification will depend on these thresholds and the actual score for these names.

# --- Tests for Error Handling ---

def test_cli_match_csv_file_not_found(tmp_path):
    """Test match-csv command when an input file is not found."""
    non_existent_file = "non_existent_file.csv"
    # Create a dummy file for the other input to isolate the error
    dummy_file = tmp_path / "dummy.csv"
    dummy_file.write_text("header\nvalue\n")

    stdout, stderr, exit_code = run_cli_command([
        "match-csv", non_existent_file, str(dummy_file)
    ])
    assert exit_code == 2 # Expected exit code for file/input error
    assert f"Error: Input file not found: {non_existent_file}" in stderr
    assert stdout == ""

    stdout, stderr, exit_code = run_cli_command([
        "match-csv", str(dummy_file), non_existent_file
    ])
    assert exit_code == 2
    assert f"Error: Input file not found: {non_existent_file}" in stderr
    assert stdout == ""

@pytest.mark.skipif(not name_matching_cli.HAS_DB_SUPPORT, reason="Database support not available")
def test_cli_match_db_unavailable_driver(monkeypatch):
    """
    Test match-db command when database support is enabled but a driver might fail.
    This is hard to test perfectly without a complex DB setup.
    We'll simulate SQLAlchemyError during engine creation or operation.
    For now, let's test the "HAS_DB_SUPPORT" is false path more directly.
    """
    # This test relies on HAS_DB_SUPPORT being true, then we'll mock something to fail.
    # A more direct test for exit code 4 (missing dependency)
    # We can achieve this by temporarily patching HAS_DB_SUPPORT in src.cli
    with patch('src.cli.HAS_DB_SUPPORT', False):
        stdout, stderr, exit_code = run_cli_command(["match-db", "table1", "table2"])
        assert exit_code == 4
        assert "Error: Database support is not available." in stderr

# To test exit code 3 (Database error), we would need to:
# 1. Ensure HAS_DB_SUPPORT is True.
# 2. Mock 'get_engine' or a subsequent DB operation call within 'match_db_tables'
#    to raise SQLAlchemyError.
# This requires more intricate mocking within the subprocess execution context,
# or by calling cli.main() directly.

# Example of how one might test DB error by calling main directly (more complex setup)
@pytest.mark.skipif(not name_matching_cli.HAS_DB_SUPPORT, reason="Database support not available")
@patch('src.cli.get_engine') # Mock at the point it's used in src.cli
def test_cli_match_db_sqlalchemy_error(mock_get_engine, capsys):
    """Test match-db command when a SQLAlchemyError occurs."""
    from sqlalchemy.exc import SQLAlchemyError

    # Configure the mock to raise SQLAlchemyError
    mock_get_engine.side_effect = SQLAlchemyError("Simulated DB connection error")

    # Prepare arguments for calling main directly
    test_args = ["match-db", "tableA", "tableB"]
    
    with pytest.raises(SystemExit) as e:
        # Patch sys.argv before calling main
        with patch.object(sys, 'argv', [sys.executable] + test_args):
            name_matching_cli.main()
            
    assert e.value.code == 3 # Expected exit code for database error
    
    captured = capsys.readouterr() # capsys works with direct calls
    assert "A database error occurred: Simulated DB connection error" in captured.err
    assert "Please check database connection and table names." in captured.err


# Placeholder for testing CSV parsing errors (EmptyDataError, ParserError)
# These would be similar to test_cli_match_csv_file_not_found but with valid file paths
# and malformed content.
def test_cli_match_csv_empty_file(tmp_path):
    file1 = tmp_path / "empty.csv"
    file1.write_text("") # Empty file
    file2 = tmp_path / "valid.csv"
    file2.write_text("id,name\n1,test\n")

    stdout, stderr, exit_code = run_cli_command(["match-csv", str(file1), str(file2)])
    assert exit_code == 2
    assert "Error: One of the CSV files is empty or invalid" in stderr


def test_cli_match_csv_parser_error(tmp_path):
    file1 = tmp_path / "malformed.csv"
    # Create a file that might cause a parser error, e.g. inconsistent quotes or delimiters
    file1.write_text('id,name\n1,"test\n2,another"test\n') 
    file2 = tmp_path / "valid2.csv"
    file2.write_text("id,name\n1,test\n")

    # The exact error message for ParserError can vary.
    # We're checking that it's classified as a file/input error (exit code 2)
    # and a relevant message is shown.
    stdout, stderr, exit_code = run_cli_command(["match-csv", str(file1), str(file2)])
    assert exit_code == 2
    assert "Error: Could not parse one of the CSV files" in stderr


# Test for general/unexpected error (exit code 1 from main's catch-all)
# This is harder to reliably trigger without knowing unhandled spots or by deep mocking.
# One way is to make a normally reliable part of a command function fail unexpectedly.
@patch('src.cli.NameMatcher.match_names') # Mock a core function called by 'match-names'
def test_cli_match_names_unexpected_error(mock_match_names_method, capsys):
    mock_match_names_method.side_effect = RuntimeError("A very unexpected runtime error!")

    test_args = ["match-names", "nameA", "nameB"]
    with pytest.raises(SystemExit) as e:
        with patch.object(sys, 'argv', [sys.executable] + test_args):
            name_matching_cli.main()
            
    assert e.value.code == 1
    captured = capsys.readouterr()
    assert "An unexpected critical error occurred" in captured.err
    assert "A very unexpected runtime error!" in captured.err


# Note: Testing config file loading implicitly through CLI behavior is complex.
# For example, if a command relies on a specific config for DB connection,
# and that config is missing/malformed, it should lead to a DB error (exit code 3).
# These are more integration-style tests for the CLI.
# The tests in test_config.py already cover config loading logic directly.

# It's also important to test that when config.ini is NOT present,
# the CLI commands that need it (like match-db) either fail gracefully (if config is essential)
# or use defaults/env vars correctly (if that's the design).
# The current CLI design seems to make `match-db` fail if `get_engine` fails, which
# would happen if DB config isn't found and no valid env vars are set.

@pytest.fixture(autouse=True)
def ensure_no_default_config_ini(monkeypatch):
    """
    Ensure that a default 'config.ini' does not interfere with CLI tests
    that might be sensitive to its presence, unless the test itself creates one.
    It mocks load_config to behave as if config.ini is not found, unless a specific
    test sets up its own config.ini and calls load_config explicitly or via CLI.
    """
    # This is tricky. If src.cli imports src.config, then src.config.load_config()
    # runs once at import time. We need to influence *that* call for some tests.
    # A simpler approach for CLI tests: if a default config.ini in the project root
    # can affect tests, it's better to temporarily rename/move it or run tests
    # in an environment where it's not present.

    # For now, we'll assume tests that care about specific config states will manage it.
    # This fixture is more of a reminder.
    # A robust way: patch 'os.path.exists' as used by 'load_config' for 'config.ini'.
    
    # Let's try to make `load_config` in `src.config` initially see no file,
    # so CLI commands rely on defaults or env vars unless a test sets up a specific config.
    
    # This patch affects the initial import of src.config if src.cli imports it.
    # It makes it so the default "config.ini" is not found during CLI test setup.
    def mock_path_exists(path):
        if path == DEFAULT_CONFIG_FILE_PATH: # or "config.ini" if that's the hardcoded default
            return False # Simulate config.ini not existing
        return os.path.exists(path) # Real behavior for other paths

    # This approach has issues because `os.path.exists` is global.
    # A better way is to control `DEFAULT_CONFIG_FILE_PATH` or mock `config.CONFIG` itself.
    # The `cleanup_config_and_env` in `test_config.py` is a good model but for `src.config`'s state.

    # For CLI tests, we are more concerned with the *behavior* of the CLI
    # assuming a certain config state, rather than testing config loading itself here.
    # So, if a CLI command fails due to missing DB config (which would come from config.ini or env),
    # that's the behavior we test, e.g., expecting exit code 3.

    pass # Placeholder, actual strategy might involve tmp_path for specific config files.

# Example test: CLI command that depends on DB, assuming no config.ini and no env vars
# This should lead to a DB error when get_engine tries to get parameters.
# This test is currently very similar to test_cli_match_db_sqlalchemy_error if get_db_config()
# returns all Nones and create_engine fails.
# The distinction is subtle: one is "DB system error", other is "config leads to DB error".
# Both should ideally result in exit code 3.

@pytest.mark.skipif(not name_matching_cli.HAS_DB_SUPPORT, reason="Database support not available")
@patch.dict(os.environ, {}, clear=True) # Clear all env vars for this test
@patch('src.config.CONFIG', None) # Ensure no INI config is loaded globally in src.config
@patch('src.config.load_config') # Prevent load_config from running and finding anything
def test_cli_match_db_missing_config_and_env(mock_load_config, capsys):

    # Ensure that when get_db_config is called, it finds nothing from INI (mocked above)
    # and nothing from ENV (cleared and not re-set).
    # This means host, port, user, etc., will be None or defaults that are unlikely to work.
    # create_engine in src.db.connection should then fail.

    test_args = ["match-db", "tableX", "tableY"]
    with pytest.raises(SystemExit) as e:
        with patch.object(sys, 'argv', [sys.executable] + test_args):
            # We need to ensure that src.config.get_db_config() within the CLI execution
            # returns all Nones or unusable defaults.
            # The @patch('src.config.CONFIG', None) helps with the INI part.
            # The @patch.dict(os.environ, {}, clear=True) helps with env part.
            name_matching_cli.main()
            
    assert e.value.code == 3 # Expecting a database-related error code
    
    captured = capsys.readouterr()
    assert "A database error occurred" in captured.err # General DB error message
    # The specific error from create_engine might be like "Can't connect to MySQL server on 'None'"
    # or similar, depending on how create_engine handles None parameters.
    # Or it might be an ArgumentError if parameters are completely missing.
    # The key is that it's caught and reported as a DB error.
    assert "Please check database connection and table names." in captured.err

# --- CLI Logging Tests ---

@patch('src.cli.match_names') # We patch the function itself to avoid its execution details
def test_cli_main_command_debug_logging(mock_match_names_func, caplog):
    """Test that the main function logs debug messages for command execution."""
    import logging
    # We need to set the level for the logger used in src.cli
    # The logger in src.cli is logging.getLogger(__name__) which is 'src.cli'
    # caplog by default captures WARNING and above. To capture DEBUG, we need to set its level.
    # Also, the logger 'src.cli' itself must be set to DEBUG level for messages to be processed.
    # This implies that the logging setup (e.g. from src.config) should allow DEBUG level.
    
    # For this test, let's assume the global logging level is set to DEBUG
    # (e.g., via config.ini or env var for a real run, or by direct setup here for test).
    # We will set the 'src.cli' logger level directly for this test and use caplog.
    
    # Configure src.cli logger to DEBUG and capture with caplog at DEBUG
    cli_logger = logging.getLogger("src.cli")
    original_level = cli_logger.level
    cli_logger.setLevel(logging.DEBUG) # Ensure src.cli logger processes DEBUG messages
    
    # caplog.set_level for the specific logger if needed, or just for the test duration globally
    # For simplicity, if root logger is DEBUG, 'src.cli' will also output if its level is DEBUG.
    # Let's ensure caplog captures DEBUG.
    
    test_args = ["match-names", "log_name1", "log_name2"]
    
    with caplog.at_level(logging.DEBUG, logger="src.cli"):
        with patch.object(sys, 'argv', [sys.executable] + test_args):
            try:
                name_matching_cli.main()
            except SystemExit: # Catch SystemExit if main calls it (e.g. on error)
                pass # We are interested in logs before any potential exit

    assert mock_match_names_func.called # Ensure the command function was called
    
    # Check for the debug log message
    found_log = False
    for record in caplog.records:
        if record.name == "src.cli" and record.levelname == "DEBUG":
            if "Executing command: match-names" in record.message and "log_name1" in record.message:
                found_log = True
                break
    assert found_log, "CLI did not log the expected DEBUG message for command execution."

    cli_logger.setLevel(original_level) # Restore original level
```
