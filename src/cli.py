#!/usr/bin/env python
"""
Command-line interface for the Name Matching application.

This module provides a command-line interface for using the Name Matching library.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from src import NameMatcher, HAS_DB_SUPPORT

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
    
    # Print results
    print(f"Match score: {score:.4f}")
    print(f"Classification: {classification}")
    print("Component scores:")
    for component, score in component_scores.items():
        print(f"  {component}: {score:.4f}")


def match_csv_files(args):
    """Match records between two CSV files."""
    # Check if files exist
    if not os.path.isfile(args.file1):
        print(f"Error: File not found: {args.file1}")
        return
    if not os.path.isfile(args.file2):
        print(f"Error: File not found: {args.file2}")
        return
    
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
            print(f"Results saved to {args.output}")
        else:
            # Print results
            print(f"Found {len(results)} matches:")
            print(results.to_string())
    except Exception as e:
        print(f"Error matching CSV files: {e}")


def match_db_tables(args):
    """Match records between two database tables."""
    if not HAS_DB_SUPPORT:
        print("Error: Database support is not available.")
        print("Make sure SQLAlchemy and PyMySQL are installed.")
        return
    
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
            print("Database tables created.")
        
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
            results.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        else:
            # Print results
            print(f"Found {len(results)} matches:")
            print(results.to_string())
    except Exception as e:
        print(f"Error matching database tables: {e}")


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
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
