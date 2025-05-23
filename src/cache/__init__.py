"""
Caching module for Name Matching application.

This module provides distributed caching capabilities using Redis.
"""

from .cache_manager import CacheManager

__all__ = ["CacheManager"]
