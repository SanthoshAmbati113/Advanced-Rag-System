"""
Semantic Cache Module for RAG System

This module provides semantic response caching to reduce latency and token costs.
It uses Qdrant as the backend for storing cached question-answer pairs with
vector similarity search for cache lookup.

The cache layer is modular and can be extended with Redis or other backends
without changing the rest of the application.
"""

from app.cache.semantic_cache import SemanticCache, get_cache, reset_cache
from app.cache.models import CacheEntry, CacheResult
from app.cache.cache_service import CacheBackend
from app.cache.qdrant_cache import QdrantCacheBackend

__all__ = [
    "SemanticCache",
    "get_cache",
    "reset_cache",
    "CacheEntry",
    "CacheResult",
    "CacheBackend",
    "QdrantCacheBackend"
]
