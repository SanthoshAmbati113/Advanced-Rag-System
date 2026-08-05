"""
Cache Service - Abstract backend interface for semantic caching.

This module defines the interface for cache backends. The current implementation
uses Qdrant, but the interface is designed to be backend-agnostic, allowing
easy migration to Redis or other backends in the future.
"""

from abc import ABC, abstractmethod
from typing import Optional
from app.cache.models import CacheEntry, CacheResult


class CacheBackend(ABC):
    """
    Abstract base class for cache backends.
    
    This interface allows swapping cache implementations (Qdrant, Redis, etc.)
    without changing the rest of the application.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the cache backend (create collections, connections, etc.)."""
        pass
    
    @abstractmethod
    def lookup(self, query_embedding: list[float], threshold: float) -> CacheResult:
        """
        Look up a cached answer based on query embedding similarity.
        
        Args:
            query_embedding: The vector embedding of the user's query
            threshold: Minimum similarity score to consider a cache hit
            
        Returns:
            CacheResult with hit status and cached data if found
        """
        pass
    
    @abstractmethod
    def store(self, entry: CacheEntry) -> None:
        """
        Store a new cache entry.
        
        Args:
            entry: The CacheEntry to store
        """
        pass
    
    @abstractmethod
    def increment_hit_count(self, entry_id: str) -> None:
        """
        Increment the hit count for a cache entry.
        
        Args:
            entry_id: The unique identifier of the cache entry
        """
        pass
