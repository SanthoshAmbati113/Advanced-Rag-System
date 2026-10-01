"""Static FAQ cache: Qdrant similarity lookup over a frozen Q&A collection."""

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
    "QdrantCacheBackend",
]
