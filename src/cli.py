#!/usr/bin/env python
"""
Command-line interface for the Name Matching application.

This module provides a command-line interface for using the Name Matching library.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import pandas as pd

from src import NameMatcher, HAS_DB_SUPPORT

# Setup logger for this module
logger = logging.getLogger(__name__)

# Import database modules if available
if HAS_DB_SUPPORT:
    from src import get_engine, init_db


def match_names(args):
    """Match two names."""
    # Create matcher
    matcher = NameMatcher(
        match_threshold=args.match_threshold,
        non_match_threshold=args.non_match_threshold,
    )
    
    # Parse additional fields if provided
    additional_fields1 = {}
    additional_fields2 = {}
    
    if args.birthdate1:
        additional_fields1["birthdate"] = args.birthdate1
    if args.birthdate2:
        additional_fields2["birthdate"] = args.birthdate2
    
    if args.province1:
        additional_fields1["province_name"] = args.province1
    if args.province2:
        additional_fields2["province_name"] = args.province2
    
    if args.city1:
        additional_fields1["city_name"] = args.city1
    if args.city2:
        additional_fields2["city_name"] = args.city2
    
    # Match names
    score, classification, component_scores = matcher.match_names(
        args.name1,
        args.name2,
        additional_fields1 if additional_fields1 else None,
        additional_fields2 if additional_fields2 else None,
    )
    
    # Log results
    logger.info(f"Match score: {score:.4f}")
    logger.info(f"Classification: {classification}")
    logger.info("Component scores:")
    for component, comp_score_val in component_scores.items(): # Renamed score to avoid conflict
        logger.info(f"  {component}: {comp_score_val:.4f}")


def match_csv_files(args):
    """Match records between two CSV files."""
    logger.info(f"Starting CSV matching for files: {args.file1}, {args.file2}")
    # Check if files exist
    if not os.path.isfile(args.file1):
        user_message = f"Error: Input file not found: {args.file1}. Please check the file path."
        logger.error(user_message)
        print(user_message, file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(args.file2):
        user_message = f"Error: Input file not found: {args.file2}. Please check the file path."
        logger.error(user_message)
        print(user_message, file=sys.stderr)
        sys.exit(2)
    
    # Create matcher
    matcher = NameMatcher(
        match_threshold=args.match_threshold,
        non_match_threshold=args.non_match_threshold,
    )
    
    # Parse column mapping
    column_mapping = {}
    if args.column_mapping:
        for mapping in args.column_mapping:
            if "=" in mapping:
                csv_col, std_col = mapping.split("=", 1)
                column_mapping[csv_col] = std_col
    
    # Match files
    try:
        results = matcher.match_csv_files(
            args.file1,
            args.file2,
            column_mapping=column_mapping if column_mapping else None,
            limit=args.limit,
        )
        
        # Save results
        if args.output:
            results.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")
        else:
            # Log results
            logger.info(f"Found {len(results)} matches:")
            logger.info(results.to_string())
        logger.info("CSV matching finished.")
    except FileNotFoundError as e:
        user_message = f"Error: A file operation failed: {e}. Please check file paths and permissions."
        logger.error(user_message, exc_info=True)
        print(user_message, file=sys.stderr)
        sys.exit(2)
    except pd.errors.EmptyDataError as e:
        user_message = f"Error: One of the CSV files is empty or invalid: {e}. Please check the file contents."
        logger.error(user_message, exc_info=True)
        print(user_message, file=sys.stderr)
        sys.exit(2)
    except pd.errors.ParserError as e:
        user_message = f"Error: Could not parse one of the CSV files: {e}. Ensure it is a valid CSV."
        logger.error(user_message, exc_info=True)
        print(user_message, file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        user_message = f"An unexpected error occurred during CSV file matching: {e}"
        logger.error(user_message, exc_info=True)
        print(user_message, file=sys.stderr)
        sys.exit(1)


def match_db_tables(args):
    """Match records between two database tables."""
    logger.info(f"Starting database table matching for tables: {args.table1}, {args.table2}")
    if not HAS_DB_SUPPORT:
        user_message = "Error: Database support is not available. Please ensure SQLAlchemy and a database driver (e.g., PyMySQL) are installed correctly."
        logger.error(user_message)
        print(user_message, file=sys.stderr)
        sys.exit(4)
    
    # Create matcher
    matcher = NameMatcher(
        match_threshold=args.match_threshold,
        non_match_threshold=args.non_match_threshold,
    )
    
    # Initialize database
    try:
        engine = get_engine()
        
        # Create tables if requested
        if args.create_tables:
            init_db(engine, create_tables=True)
            logger.info("Database tables created.")
        
        # Parse blocking fields
        blocking_fields = args.blocking_fields.split(",") if args.blocking_fields else None
        
        # Match tables
        results = matcher.match_db_tables(
            args.table1,
            args.table2,
            engine=engine,
            use_blocking=not args.no_blocking,
            blocking_fields=blocking_fields,
            limit=args.limit,
            save_results=not args.no_save,
        )
        
        # Save results to CSV if requested
        if args.output:
            try:
                results.to_csv(args.output, index=False)
                logger.info(f"Results saved to {args.output}")
            except IOError as e:
                user_message = f"Error: Could not write results to output file {args.output}: {e}"
                logger.error(user_message, exc_info=True)
                print(user_message, file=sys.stderr)
                sys.exit(2) # File I/O error
        else:
            # Log results
            logger.info(f"Found {len(results)} matches:")
            logger.info(results.to_string())
        logger.info("Database table matching finished.")
    except SQLAlchemyError as e: # Specific to database errors
        user_message = f"A database error occurred: {e}. Please check database connection and table names."
        logger.error(user_message, exc_info=True)
        print(user_message, file=sys.stderr)
        sys.exit(3)
    except Exception as e: # Catch other unexpected errors
        user_message = f"An unexpected error occurred during database table matching: {e}"
        logger.error(user_message, exc_info=True)
        print(user_message, file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Name Matching CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.75,
        help="Threshold for classifying as a match (default: 0.75)",
    )
    common_parser.add_argument(
        "--non-match-threshold",
        type=float,
        default=0.55,
        help="Threshold for classifying as a non-match (default: 0.55)",
    )
    
    # Match names command
    names_parser = subparsers.add_parser(
        "match-names",
        parents=[common_parser],
        help="Match two names",
    )
    names_parser.add_argument("name1", help="First name")
    names_parser.add_argument("name2", help="Second name")
    names_parser.add_argument("--birthdate1", help="Birthdate for first name (YYYY-MM-DD)")
    names_parser.add_argument("--birthdate2", help="Birthdate for second name (YYYY-MM-DD)")
    names_parser.add_argument("--province1", help="Province for first name")
    names_parser.add_argument("--province2", help="Province for second name")
    names_parser.add_argument("--city1", help="City for first name")
    names_parser.add_argument("--city2", help="City for second name")
    names_parser.set_defaults(func=match_names)
    
    # Match CSV files command
    csv_parser = subparsers.add_parser(
        "match-csv",
        parents=[common_parser],
        help="Match records between two CSV files",
    )
    csv_parser.add_argument("file1", help="Path to first CSV file")
    csv_parser.add_argument("file2", help="Path to second CSV file")
    csv_parser.add_argument(
        "--column-mapping",
        nargs="+",
        help="Column mapping (e.g., 'Name=first_name' 'Surname=last_name')",
    )
    csv_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of matches to return per record",
    )
    csv_parser.add_argument(
        "--output",
        help="Path to output CSV file",
    )
    csv_parser.set_defaults(func=match_csv_files)
    
    # Match database tables command
    if HAS_DB_SUPPORT:
        db_parser = subparsers.add_parser(
            "match-db",
            parents=[common_parser],
            help="Match records between two database tables",
        )
        db_parser.add_argument("table1", help="Name of first table")
        db_parser.add_argument("table2", help="Name of second table")
        db_parser.add_argument(
            "--create-tables",
            action="store_true",
            help="Create tables if they don't exist",
        )
        db_parser.add_argument(
            "--no-blocking",
            action="store_true",
            help="Disable blocking (compare all records)",
        )
        db_parser.add_argument(
            "--blocking-fields",
            help="Comma-separated list of fields to use for blocking",
        )
        db_parser.add_argument(
            "--limit",
            type=int,
            help="Maximum number of matches to return per record",
        )
        db_parser.add_argument(
            "--no-save",
            action="store_true",
            help="Don't save match results to the database",
        )
        db_parser.add_argument(
            "--output",
            help="Path to output CSV file",
        )
        db_parser.set_defaults(func=match_db_tables)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run command
    if hasattr(args, "func"):
        logger.debug(f"Executing command: {args.command} with arguments: {vars(args)}")
        try:
            args.func(args)
            # If the function completes without sys.exit, it's a success for the command execution part
            logger.info(f"Command '{args.command}' completed successfully.")
        # Errors handled and exited within args.func (like FileNotFoundError, SQLAlchemyError) won't be caught here.
        # This top-level exception is for truly unexpected issues in the command functions or arg parsing.
        except Exception as e:
            user_message = f"An unexpected critical error occurred while executing command '{args.command}': {e}. Check logs for details."
            logger.error(user_message, exc_info=True)
            print(user_message, file=sys.stderr)
            sys.exit(1) # General critical error
    else:
        parser.print_help()
        # Consider sys.exit(0) or a specific code if no command is provided, if that's an error.
        # For now, it prints help and exits with 0 by default.


if __name__ == "__main__":
    main()
