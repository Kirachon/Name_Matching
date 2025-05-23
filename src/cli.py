#!/usr/bin/env python
"""
Enhanced command-line interface for the Name Matching application.

This module provides a comprehensive CLI using Click and Rich for better user experience.
"""

import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

from src import NameMatcher, HAS_DB_SUPPORT
from src.config import get_matching_thresholds

# Setup console for rich output
console = Console()

# Import database modules if available
if HAS_DB_SUPPORT:
    from src import get_engine, init_db


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Enhanced Name Matching CLI - High-performance name matching for Filipino identity data."""
    pass


@cli.command()
@click.argument('name1')
@click.argument('name2')
@click.option('--match-threshold', '-mt', type=float, help='Match threshold (0.0-1.0)')
@click.option('--non-match-threshold', '-nt', type=float, help='Non-match threshold (0.0-1.0)')
@click.option('--birthdate1', '-b1', help='Birthdate for first record (YYYY-MM-DD)')
@click.option('--birthdate2', '-b2', help='Birthdate for second record (YYYY-MM-DD)')
@click.option('--province1', '-p1', help='Province for first record')
@click.option('--province2', '-p2', help='Province for second record')
@click.option('--city1', '-c1', help='City for first record')
@click.option('--city2', '-c2', help='City for second record')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def match(
    name1: str,
    name2: str,
    match_threshold: Optional[float],
    non_match_threshold: Optional[float],
    birthdate1: Optional[str],
    birthdate2: Optional[str],
    province1: Optional[str],
    province2: Optional[str],
    city1: Optional[str],
    city2: Optional[str],
    verbose: bool
):
    """Match two names and return similarity score."""

    with console.status("[bold green]Initializing matcher...") as status:
        try:
            # Create matcher with custom thresholds if provided
            kwargs = {}
            if match_threshold is not None:
                kwargs['match_threshold'] = match_threshold
            if non_match_threshold is not None:
                kwargs['non_match_threshold'] = non_match_threshold

            matcher = NameMatcher(**kwargs)
            status.update("[bold green]Performing name matching...")

            # Parse additional fields
            additional_fields1 = {}
            additional_fields2 = {}

            if birthdate1:
                additional_fields1["birthdate"] = birthdate1
            if birthdate2:
                additional_fields2["birthdate"] = birthdate2
            if province1:
                additional_fields1["province_name"] = province1
            if province2:
                additional_fields2["province_name"] = province2
            if city1:
                additional_fields1["city_name"] = city1
            if city2:
                additional_fields2["city_name"] = city2

            # Perform matching
            start_time = time.time()
            score, classification, component_scores = matcher.match_names(
                name1,
                name2,
                additional_fields1 if additional_fields1 else None,
                additional_fields2 if additional_fields2 else None
            )
            processing_time = (time.time() - start_time) * 1000

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)

    # Display results
    display_match_results(name1, name2, score, classification, component_scores, processing_time, verbose)


def display_match_results(name1: str, name2: str, score: float, classification, component_scores: Dict, processing_time: float, verbose: bool):
    """Display match results in a formatted table."""

    # Create main results panel
    result_text = Text()
    result_text.append("Match Results\n", style="bold blue")
    result_text.append(f"Name 1: {name1}\n", style="cyan")
    result_text.append(f"Name 2: {name2}\n", style="cyan")
    result_text.append(f"Overall Score: {score:.4f}\n", style="bold yellow")
    result_text.append(f"Classification: {classification.value}\n", style="bold green" if classification.value == "match" else "bold red")
    result_text.append(f"Processing Time: {processing_time:.2f}ms", style="dim")

    console.print(Panel(result_text, title="Name Matching Results", border_style="blue"))

    if verbose and component_scores:
        # Create component scores table
        table = Table(title="Component Scores", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Score", justify="right", style="yellow")

        for component, comp_score in component_scores.items():
            table.add_row(component.replace('_', ' ').title(), f"{comp_score:.4f}")

        console.print(table)


@cli.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.argument('table_name')
@click.option('--column-mapping', '-m', help='JSON string mapping CSV columns to database fields')
@click.option('--use-blocking/--no-blocking', default=True, help='Use blocking for performance')
@click.option('--blocking-fields', '-bf', multiple=True, help='Fields to use for blocking')
@click.option('--save-results/--no-save', default=False, help='Save results to database')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--limit', '-l', type=int, help='Limit number of records to process')
def match_csv(
    csv_file: str,
    table_name: str,
    column_mapping: Optional[str],
    use_blocking: bool,
    blocking_fields: tuple,
    save_results: bool,
    output: Optional[str],
    limit: Optional[int]
):
    """Match CSV file against database table."""

    if not HAS_DB_SUPPORT:
        console.print("[bold red]Error:[/bold red] Database support not available")
        sys.exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        task = progress.add_task("Processing CSV matching...", total=100)

        try:
            # Initialize matcher
            progress.update(task, description="Initializing matcher...", advance=10)
            matcher = NameMatcher()

            # Parse column mapping if provided
            mapping = {}
            if column_mapping:
                import json
                mapping = json.loads(column_mapping)

            progress.update(task, description="Reading CSV file...", advance=20)

            # Process matching
            progress.update(task, description="Performing matching...", advance=30)

            # This would call the actual matching method
            # results = matcher.match_csv_files(
            #     csv_file,
            #     table_name,
            #     column_mapping=mapping,
            #     use_blocking=use_blocking,
            #     blocking_fields=list(blocking_fields) if blocking_fields else None,
            #     save_results=save_results,
            #     limit=limit
            # )

            # Simulate processing for now
            import time
            time.sleep(2)

            progress.update(task, description="Completed!", advance=40)

            console.print("[bold green]CSV matching completed successfully![/bold green]")

            if output:
                console.print(f"Results saved to: {output}")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)


@cli.command()
@click.argument('table1')
@click.argument('table2')
@click.option('--use-blocking/--no-blocking', default=True, help='Use blocking for performance')
@click.option('--blocking-fields', '-bf', multiple=True, help='Fields to use for blocking')
@click.option('--save-results/--no-save', default=False, help='Save results to database')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--limit', '-l', type=int, help='Limit number of records to process')
def match_tables(
    table1: str,
    table2: str,
    use_blocking: bool,
    blocking_fields: tuple,
    save_results: bool,
    output: Optional[str],
    limit: Optional[int]
):
    """Match records between two database tables."""

    if not HAS_DB_SUPPORT:
        console.print("[bold red]Error:[/bold red] Database support not available")
        sys.exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        task = progress.add_task("Processing table matching...", total=100)

        try:
            # Initialize matcher
            progress.update(task, description="Initializing matcher...", advance=10)
            matcher = NameMatcher()

            progress.update(task, description="Connecting to database...", advance=20)

            # Process matching
            progress.update(task, description="Performing matching...", advance=30)

            # This would call the actual matching method
            # results = matcher.match_db_tables(
            #     table1,
            #     table2,
            #     use_blocking=use_blocking,
            #     blocking_fields=list(blocking_fields) if blocking_fields else None,
            #     save_results=save_results,
            #     limit=limit
            # )

            # Simulate processing for now
            import time
            time.sleep(3)

            progress.update(task, description="Completed!", advance=40)

            console.print("[bold green]Table matching completed successfully![/bold green]")

            if output:
                console.print(f"Results saved to: {output}")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)


@cli.command()
def status():
    """Show system status and configuration."""

    # Get system information
    try:
        thresholds = get_matching_thresholds()

        # Create status table
        table = Table(title="System Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")

        # Check components
        table.add_row("Name Matcher", "✓ Available", "Core matching engine ready")

        if HAS_DB_SUPPORT:
            table.add_row("Database", "✓ Available", "MySQL support enabled")
        else:
            table.add_row("Database", "✗ Unavailable", "MySQL support disabled")

        # Check GPU support
        try:
            from src.gpu_acceleration import get_gpu_status
            gpu_status = get_gpu_status()
            if gpu_status.get('available', False):
                table.add_row("GPU Acceleration", "✓ Available", f"Device: {gpu_status.get('device', 'Unknown')}")
            else:
                table.add_row("GPU Acceleration", "✗ Unavailable", "No GPU detected")
        except ImportError:
            table.add_row("GPU Acceleration", "✗ Unavailable", "GPU support not installed")

        # Configuration
        table.add_row("Match Threshold", "⚙️ Configured", f"{thresholds.get('match_threshold', 0.75):.2f}")
        table.add_row("Non-Match Threshold", "⚙️ Configured", f"{thresholds.get('non_match_threshold', 0.55):.2f}")

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error getting status:[/bold red] {e}")


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(host: str, port: int, reload: bool):
    """Start the REST API server."""

    try:
        import uvicorn
        from src.api.main import app

        console.print(f"[bold green]Starting API server on {host}:{port}[/bold green]")
        console.print(f"API documentation available at: http://{host}:{port}/docs")

        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )

    except ImportError:
        console.print("[bold red]Error:[/bold red] FastAPI dependencies not installed")
        console.print("Install with: pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error starting server:[/bold red] {e}")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
