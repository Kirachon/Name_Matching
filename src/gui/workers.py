"""
Background workers for the Name Matching GUI.

This module provides background workers for long-running operations in the Name Matching GUI.
"""

import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import pandas as pd
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from src import NameMatcher


class WorkerSignals(QObject):
    """Signals for worker threads."""
    
    # Signal for progress updates (value, maximum)
    progress = pyqtSignal(int, int)
    
    # Signal for status updates
    status = pyqtSignal(str)
    
    # Signal for errors
    error = pyqtSignal(str)
    
    # Signal for completion with result
    result = pyqtSignal(object)
    
    # Signal for completion without result
    finished = pyqtSignal()


class DatabaseConnectionWorker(QThread):
    """Worker for database connection operations."""
    
    def __init__(
        self,
        db_name: str,
        host: str,
        port: str,
        user: str,
        password: str,
        use_ssl: bool = False,
        parent: Optional[QObject] = None
    ):
        """
        Initialize the worker.
        
        Args:
            db_name: Database name
            host: Database host
            port: Database port
            user: Database user
            password: Database password
            use_ssl: Whether to use SSL
            parent: Parent object
        """
        super().__init__(parent)
        self.db_name = db_name
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.use_ssl = use_ssl
        self.signals = WorkerSignals()
    
    def run(self):
        """Run the worker."""
        try:
            self.signals.status.emit("Connecting to database...")
            
            # Import here to avoid circular imports
            from src.db.connection import get_engine
            
            # Set environment variables for connection
            import os
            os.environ["MYSQL_HOST"] = self.host
            os.environ["MYSQL_PORT"] = self.port
            os.environ["MYSQL_DATABASE"] = self.db_name
            os.environ["MYSQL_USER"] = self.user
            os.environ["MYSQL_PASSWORD"] = self.password
            
            # Connect to database
            engine = get_engine(
                db_name=self.db_name,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                use_ssl=self.use_ssl
            )
            
            # Test connection
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            
            self.signals.result.emit(engine)
            self.signals.status.emit("Connected to database.")
            
        except Exception as e:
            self.signals.error.emit(f"Error connecting to database: {str(e)}")
            traceback.print_exc()
        
        finally:
            self.signals.finished.emit()


class TableListWorker(QThread):
    """Worker for retrieving table list from database."""
    
    def __init__(self, engine: Engine, parent: Optional[QObject] = None):
        """
        Initialize the worker.
        
        Args:
            engine: SQLAlchemy engine
            parent: Parent object
        """
        super().__init__(parent)
        self.engine = engine
        self.signals = WorkerSignals()
    
    def run(self):
        """Run the worker."""
        try:
            self.signals.status.emit("Retrieving table list...")
            
            # Get table list
            with self.engine.connect() as conn:
                result = conn.execute("SHOW TABLES")
                tables = [row[0] for row in result]
            
            self.signals.result.emit(tables)
            self.signals.status.emit(f"Retrieved {len(tables)} tables.")
            
        except Exception as e:
            self.signals.error.emit(f"Error retrieving table list: {str(e)}")
            traceback.print_exc()
        
        finally:
            self.signals.finished.emit()


class TablePreviewWorker(QThread):
    """Worker for retrieving table preview from database."""
    
    def __init__(
        self, 
        engine: Engine, 
        table_name: str, 
        limit: int = 100,
        parent: Optional[QObject] = None
    ):
        """
        Initialize the worker.
        
        Args:
            engine: SQLAlchemy engine
            table_name: Table name
            limit: Maximum number of rows to retrieve
            parent: Parent object
        """
        super().__init__(parent)
        self.engine = engine
        self.table_name = table_name
        self.limit = limit
        self.signals = WorkerSignals()
    
    def run(self):
        """Run the worker."""
        try:
            self.signals.status.emit(f"Retrieving preview for table {self.table_name}...")
            
            # Get table preview
            query = f"SELECT * FROM {self.table_name} LIMIT {self.limit}"
            df = pd.read_sql(query, self.engine)
            
            self.signals.result.emit(df)
            self.signals.status.emit(f"Retrieved {len(df)} rows from {self.table_name}.")
            
        except Exception as e:
            self.signals.error.emit(f"Error retrieving table preview: {str(e)}")
            traceback.print_exc()
        
        finally:
            self.signals.finished.emit()


class MatchingWorker(QThread):
    """Worker for matching operations."""
    
    def __init__(
        self,
        matcher: NameMatcher,
        match_type: str,
        match_params: Dict[str, Any],
        parent: Optional[QObject] = None
    ):
        """
        Initialize the worker.
        
        Args:
            matcher: NameMatcher instance
            match_type: Type of matching operation ('names', 'csv', 'db')
            match_params: Parameters for the matching operation
            parent: Parent object
        """
        super().__init__(parent)
        self.matcher = matcher
        self.match_type = match_type
        self.match_params = match_params
        self.signals = WorkerSignals()
        self._is_cancelled = False
    
    def cancel(self):
        """Cancel the worker."""
        self._is_cancelled = True
    
    def run(self):
        """Run the worker."""
        try:
            if self._is_cancelled:
                return
            
            self.signals.status.emit(f"Starting {self.match_type} matching...")
            
            # Perform matching based on type
            if self.match_type == 'names':
                result = self._match_names()
            elif self.match_type == 'csv':
                result = self._match_csv()
            elif self.match_type == 'db':
                result = self._match_db()
            else:
                raise ValueError(f"Unknown match type: {self.match_type}")
            
            if self._is_cancelled:
                return
            
            self.signals.result.emit(result)
            self.signals.status.emit("Matching completed.")
            
        except Exception as e:
            if not self._is_cancelled:
                self.signals.error.emit(f"Error during matching: {str(e)}")
                traceback.print_exc()
        
        finally:
            self.signals.finished.emit()
    
    def _match_names(self):
        """Match individual names."""
        name1 = self.match_params.get('name1', '')
        name2 = self.match_params.get('name2', '')
        additional_fields1 = self.match_params.get('additional_fields1', {})
        additional_fields2 = self.match_params.get('additional_fields2', {})
        
        # Match names
        score, classification, component_scores = self.matcher.match_names(
            name1,
            name2,
            additional_fields1 if additional_fields1 else None,
            additional_fields2 if additional_fields2 else None,
        )
        
        return {
            'score': score,
            'classification': classification,
            'component_scores': component_scores
        }
    
    def _match_csv(self):
        """Match CSV files."""
        file1 = self.match_params.get('file1', '')
        file2 = self.match_params.get('file2', '')
        column_mapping = self.match_params.get('column_mapping', {})
        limit = self.match_params.get('limit', None)
        
        # Match CSV files
        results = self.matcher.match_csv_files(
            file1,
            file2,
            column_mapping=column_mapping if column_mapping else None,
            limit=limit,
        )
        
        return results
    
    def _match_db(self):
        """Match database tables."""
        table1 = self.match_params.get('table1', '')
        table2 = self.match_params.get('table2', '')
        engine = self.match_params.get('engine', None)
        use_blocking = self.match_params.get('use_blocking', True)
        blocking_fields = self.match_params.get('blocking_fields', None)
        limit = self.match_params.get('limit', None)
        save_results = self.match_params.get('save_results', True)
        
        # Get total record count for progress updates
        with engine.connect() as conn:
            if use_blocking and blocking_fields:
                # If using blocking, we can't easily determine the total count
                # Just use a placeholder
                total_count = 100
            else:
                # Get count of records in both tables
                count1 = conn.execute(f"SELECT COUNT(*) FROM {table1}").scalar()
                count2 = conn.execute(f"SELECT COUNT(*) FROM {table2}").scalar()
                total_count = count1 * count2
        
        self.signals.progress.emit(0, total_count)
        
        # Match database tables
        results = self.matcher.match_db_tables(
            table1,
            table2,
            engine=engine,
            use_blocking=use_blocking,
            blocking_fields=blocking_fields,
            limit=limit,
            save_results=save_results,
        )
        
        self.signals.progress.emit(total_count, total_count)
        
        return results
