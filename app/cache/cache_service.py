"""
Cache Service - Abstract backend interface for static FAQ lookup.
"""

from abc import ABC, abstractmethod

from app.cache.models import CacheResult


class CacheBackend(ABC):
    """Read-only FAQ lookup backend. Writes happen only via the FAQ seeder."""

    @abstractmethod
    def initialize(self) -> None:
        """Connect and verify the FAQ collection exists."""
        pass

    @abstractmethod
    def lookup(self, query_embedding: list[float], threshold: float) -> CacheResult:
        """Return a hit if the nearest FAQ variant exceeds the similarity threshold."""
        pass
