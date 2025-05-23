"""
REST API module for Name Matching application.

This module provides a RESTful API interface for name matching functionality.
"""

from .main import app
from .models import *
from .auth import *

__all__ = ["app"]
