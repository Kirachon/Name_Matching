"""
GUI outline for the Name Matching application.

This is a conceptual outline of how a GUI could be implemented for the Name Matching application.
Note: This is not a complete implementation, just a sketch of the structure.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd

from src import NameMatcher, HAS_DB_SUPPORT

# Import database modules if available
if HAS_DB_SUPPORT:
    from src import get_engine, init_db


class NameMatchingApp(tk.Tk):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Name Matching Application")
        self.geometry("800x600")
        
        # Create matcher
        self.matcher = NameMatcher()
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.names_tab = NameMatchingTab(self.notebook, self.matcher)
        self.csv_tab = CSVMatchingTab(self.notebook, self.matcher)
        
        # Add database tab if supported
        if HAS_DB_SUPPORT:
            self.db_tab = DatabaseMatchingTab(self.notebook, self.matcher)
            self.notebook.add(self.db_tab, text="Database Matching")
        
        # Add tabs to notebook
        self.notebook.add(self.names_tab, text="Name Matching")
        self.notebook.add(self.csv_tab, text="CSV Matching")
        
        # Create menu
        self.create_menu()
    
    def create_menu(self):
        """Create the application menu."""
        menu_bar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=menu_bar)
    
    def show_about(self):
        """Show the about dialog."""
        messagebox.showinfo(
            "About",
            "Name Matching Application\n\n"
            "A tool for matching and comparing names, with special focus on Filipino names."
        )


class NameMatchingTab(ttk.Frame):
    """Tab for matching individual names."""
    
    def __init__(self, parent, matcher):
        super().__init__(parent)
        self.matcher = matcher
        
        # Create widgets
        self.create_widgets()
    
    def create_widgets(self):
        """Create the tab widgets."""
        # Name 1 frame
        name1_frame = ttk.LabelFrame(self, text="Name 1")
        name1_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(name1_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.name1_entry = ttk.Entry(name1_frame, width=40)
        self.name1_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(name1_frame, text="Birthdate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.birthdate1_entry = ttk.Entry(name1_frame, width=20)
        self.birthdate1_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(name1_frame, text="Province:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.province1_entry = ttk.Entry(name1_frame, width=30)
        self.province1_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(name1_frame, text="City:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.city1_entry = ttk.Entry(name1_frame, width=30)
        self.city1_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Name 2 frame
        name2_frame = ttk.LabelFrame(self, text="Name 2")
        name2_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(name2_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.name2_entry = ttk.Entry(name2_frame, width=40)
        self.name2_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(name2_frame, text="Birthdate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.birthdate2_entry = ttk.Entry(name2_frame, width=20)
        self.birthdate2_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(name2_frame, text="Province:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.province2_entry = ttk.Entry(name2_frame, width=30)
        self.province2_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(name2_frame, text="City:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.city2_entry = ttk.Entry(name2_frame, width=30)
        self.city2_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Match button
        self.match_button = ttk.Button(self, text="Match Names", command=self.match_names)
        self.match_button.pack(pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, width=80, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def match_names(self):
        """Match the names and display results."""
        # Get input values
        name1 = self.name1_entry.get().strip()
        name2 = self.name2_entry.get().strip()
        
        if not name1 or not name2:
            messagebox.showerror("Error", "Please enter both names.")
            return
        
        # Get additional fields
        additional_fields1 = {}
        additional_fields2 = {}
        
        birthdate1 = self.birthdate1_entry.get().strip()
        if birthdate1:
            additional_fields1["birthdate"] = birthdate1
        
        birthdate2 = self.birthdate2_entry.get().strip()
        if birthdate2:
            additional_fields2["birthdate"] = birthdate2
        
        province1 = self.province1_entry.get().strip()
        if province1:
            additional_fields1["province_name"] = province1
        
        province2 = self.province2_entry.get().strip()
        if province2:
            additional_fields2["province_name"] = province2
        
        city1 = self.city1_entry.get().strip()
        if city1:
            additional_fields1["city_name"] = city1
        
        city2 = self.city2_entry.get().strip()
        if city2:
            additional_fields2["city_name"] = city2
        
        try:
            # Match names
            score, classification, component_scores = self.matcher.match_names(
                name1,
                name2,
                additional_fields1 if additional_fields1 else None,
                additional_fields2 if additional_fields2 else None,
            )
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Match score: {score:.4f}\n")
            self.results_text.insert(tk.END, f"Classification: {classification}\n\n")
            self.results_text.insert(tk.END, "Component scores:\n")
            
            for component, comp_score in component_scores.items():
                self.results_text.insert(tk.END, f"  {component}: {comp_score:.4f}\n")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error matching names: {e}")


class CSVMatchingTab(ttk.Frame):
    """Tab for matching CSV files."""
    
    def __init__(self, parent, matcher):
        super().__init__(parent)
        self.matcher = matcher
        
        # Create widgets
        self.create_widgets()
    
    def create_widgets(self):
        """Create the tab widgets."""
        # File 1 frame
        file1_frame = ttk.LabelFrame(self, text="CSV File 1")
        file1_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(file1_frame, text="File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.file1_entry = ttk.Entry(file1_frame, width=50)
        self.file1_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.file1_button = ttk.Button(file1_frame, text="Browse...", command=self.browse_file1)
        self.file1_button.grid(row=0, column=2, padx=5, pady=5)
        
        # File 2 frame
        file2_frame = ttk.LabelFrame(self, text="CSV File 2")
        file2_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(file2_frame, text="File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.file2_entry = ttk.Entry(file2_frame, width=50)
        self.file2_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.file2_button = ttk.Button(file2_frame, text="Browse...", command=self.browse_file2)
        self.file2_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Options frame
        options_frame = ttk.LabelFrame(self, text="Options")
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(options_frame, text="Column Mapping:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.mapping_entry = ttk.Entry(options_frame, width=50)
        self.mapping_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(options_frame, text="(e.g., Name=first_name,Surname=last_name)").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(options_frame, text="Limit:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.limit_entry = ttk.Entry(options_frame, width=10)
        self.limit_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(options_frame, text="Output File:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_entry = ttk.Entry(options_frame, width=50)
        self.output_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        self.output_button = ttk.Button(options_frame, text="Browse...", command=self.browse_output)
        self.output_button.grid(row=2, column=2, padx=5, pady=5)
        
        # Match button
        self.match_button = ttk.Button(self, text="Match Files", command=self.match_files)
        self.match_button.pack(pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, width=80, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def browse_file1(self):
        """Browse for CSV file 1."""
        filename = filedialog.askopenfilename(
            title="Select CSV File 1",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if filename:
            self.file1_entry.delete(0, tk.END)
            self.file1_entry.insert(0, filename)
    
    def browse_file2(self):
        """Browse for CSV file 2."""
        filename = filedialog.askopenfilename(
            title="Select CSV File 2",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if filename:
            self.file2_entry.delete(0, tk.END)
            self.file2_entry.insert(0, filename)
    
    def browse_output(self):
        """Browse for output file."""
        filename = filedialog.asksaveasfilename(
            title="Save Results As",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            defaultextension=".csv",
        )
        if filename:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, filename)
    
    def match_files(self):
        """Match the CSV files and display results."""
        # Get input values
        file1 = self.file1_entry.get().strip()
        file2 = self.file2_entry.get().strip()
        
        if not file1 or not file2:
            messagebox.showerror("Error", "Please select both CSV files.")
            return
        
        # Parse column mapping
        column_mapping = {}
        mapping_str = self.mapping_entry.get().strip()
        if mapping_str:
            for mapping in mapping_str.split(","):
                if "=" in mapping:
                    csv_col, std_col = mapping.split("=", 1)
                    column_mapping[csv_col.strip()] = std_col.strip()
        
        # Parse limit
        limit = None
        limit_str = self.limit_entry.get().strip()
        if limit_str:
            try:
                limit = int(limit_str)
            except ValueError:
                messagebox.showerror("Error", "Limit must be a number.")
                return
        
        # Get output file
        output_file = self.output_entry.get().strip()
        
        try:
            # Match files
            results = self.matcher.match_csv_files(
                file1,
                file2,
                column_mapping=column_mapping if column_mapping else None,
                limit=limit,
            )
            
            # Save results if output file specified
            if output_file:
                results.to_csv(output_file, index=False)
                messagebox.showinfo("Success", f"Results saved to {output_file}")
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Found {len(results)} matches.\n\n")
            
            if not results.empty:
                self.results_text.insert(tk.END, results.to_string())
        
        except Exception as e:
            messagebox.showerror("Error", f"Error matching files: {e}")


class DatabaseMatchingTab(ttk.Frame):
    """Tab for matching database tables."""
    
    def __init__(self, parent, matcher):
        super().__init__(parent)
        self.matcher = matcher
        
        # Create widgets
        self.create_widgets()
    
    def create_widgets(self):
        """Create the tab widgets."""
        # Database connection frame
        db_frame = ttk.LabelFrame(self, text="Database Connection")
        db_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(db_frame, text="Host:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.host_entry = ttk.Entry(db_frame, width=30)
        self.host_entry.insert(0, "localhost")
        self.host_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(db_frame, text="Port:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.port_entry = ttk.Entry(db_frame, width=10)
        self.port_entry.insert(0, "3306")
        self.port_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(db_frame, text="Database:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.db_entry = ttk.Entry(db_frame, width=30)
        self.db_entry.insert(0, "name_matching")
        self.db_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(db_frame, text="User:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.user_entry = ttk.Entry(db_frame, width=30)
        self.user_entry.insert(0, "root")
        self.user_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(db_frame, text="Password:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        self.password_entry = ttk.Entry(db_frame, width=30, show="*")
        self.password_entry.grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)
        
        self.connect_button = ttk.Button(db_frame, text="Connect", command=self.connect_db)
        self.connect_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        self.create_tables_button = ttk.Button(db_frame, text="Create Tables", command=self.create_tables)
        self.create_tables_button.grid(row=3, column=2, columnspan=2, padx=5, pady=5)
        
        # Tables frame
        tables_frame = ttk.LabelFrame(self, text="Tables")
        tables_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(tables_frame, text="Table 1:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.table1_entry = ttk.Entry(tables_frame, width=30)
        self.table1_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(tables_frame, text="Table 2:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.table2_entry = ttk.Entry(tables_frame, width=30)
        self.table2_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Options frame
        options_frame = ttk.LabelFrame(self, text="Options")
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.use_blocking_var = tk.BooleanVar(value=True)
        self.use_blocking_check = ttk.Checkbutton(
            options_frame,
            text="Use Blocking",
            variable=self.use_blocking_var,
        )
        self.use_blocking_check.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(options_frame, text="Blocking Fields:").grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.blocking_fields_entry = ttk.Entry(options_frame, width=30)
        self.blocking_fields_entry.insert(0, "province_name,city_name")
        self.blocking_fields_entry.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(options_frame, text="Limit:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.limit_entry = ttk.Entry(options_frame, width=10)
        self.limit_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.save_results_var = tk.BooleanVar(value=True)
        self.save_results_check = ttk.Checkbutton(
            options_frame,
            text="Save Results to Database",
            variable=self.save_results_var,
        )
        self.save_results_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(options_frame, text="Output File:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_entry = ttk.Entry(options_frame, width=50)
        self.output_entry.grid(row=3, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        self.output_button = ttk.Button(options_frame, text="Browse...", command=self.browse_output)
        self.output_button.grid(row=3, column=3, padx=5, pady=5)
        
        # Match button
        self.match_button = ttk.Button(self, text="Match Tables", command=self.match_tables)
        self.match_button.pack(pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, width=80, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize engine
        self.engine = None
    
    def browse_output(self):
        """Browse for output file."""
        filename = filedialog.asksaveasfilename(
            title="Save Results As",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            defaultextension=".csv",
        )
        if filename:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, filename)
    
    def connect_db(self):
        """Connect to the database."""
        # Get connection parameters
        host = self.host_entry.get().strip()
        port = self.port_entry.get().strip()
        db_name = self.db_entry.get().strip()
        user = self.user_entry.get().strip()
        password = self.password_entry.get()
        
        try:
            # Set environment variables for connection
            import os
            os.environ["MYSQL_HOST"] = host
            os.environ["MYSQL_PORT"] = port
            os.environ["MYSQL_DATABASE"] = db_name
            os.environ["MYSQL_USER"] = user
            os.environ["MYSQL_PASSWORD"] = password
            
            # Connect to database
            self.engine = get_engine()
            
            messagebox.showinfo("Success", "Connected to database.")
        except Exception as e:
            messagebox.showerror("Error", f"Error connecting to database: {e}")
    
    def create_tables(self):
        """Create database tables."""
        if not self.engine:
            messagebox.showerror("Error", "Please connect to the database first.")
            return
        
        try:
            # Create tables
            init_db(self.engine, create_tables=True)
            
            messagebox.showinfo("Success", "Database tables created.")
        except Exception as e:
            messagebox.showerror("Error", f"Error creating tables: {e}")
    
    def match_tables(self):
        """Match the database tables and display results."""
        if not self.engine:
            messagebox.showerror("Error", "Please connect to the database first.")
            return
        
        # Get input values
        table1 = self.table1_entry.get().strip()
        table2 = self.table2_entry.get().strip()
        
        if not table1 or not table2:
            messagebox.showerror("Error", "Please enter both table names.")
            return
        
        # Get options
        use_blocking = self.use_blocking_var.get()
        
        blocking_fields = None
        if use_blocking:
            blocking_fields_str = self.blocking_fields_entry.get().strip()
            if blocking_fields_str:
                blocking_fields = blocking_fields_str.split(",")
        
        limit = None
        limit_str = self.limit_entry.get().strip()
        if limit_str:
            try:
                limit = int(limit_str)
            except ValueError:
                messagebox.showerror("Error", "Limit must be a number.")
                return
        
        save_results = self.save_results_var.get()
        
        output_file = self.output_entry.get().strip()
        
        try:
            # Match tables
            results = self.matcher.match_db_tables(
                table1,
                table2,
                engine=self.engine,
                use_blocking=use_blocking,
                blocking_fields=blocking_fields,
                limit=limit,
                save_results=save_results,
            )
            
            # Save results if output file specified
            if output_file:
                results.to_csv(output_file, index=False)
                messagebox.showinfo("Success", f"Results saved to {output_file}")
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Found {len(results)} matches.\n\n")
            
            if not results.empty:
                self.results_text.insert(tk.END, results.to_string())
        
        except Exception as e:
            messagebox.showerror("Error", f"Error matching tables: {e}")


# This is just an outline, not a complete implementation
if __name__ == "__main__":
    print("This is just an outline of how a GUI could be implemented.")
    print("To create a working GUI, you would need to implement the missing functionality.")
