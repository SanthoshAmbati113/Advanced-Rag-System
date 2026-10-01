"""
Static FAQ cache orchestrator.

Embeds the incoming query and searches the pre-seeded FAQ collection.
Nothing is written back after a RAG miss.
"""

import logfire
from typing import Optional

from app.config import settings
from app.cache.cache_service import CacheBackend
from app.cache.qdrant_cache import QdrantCacheBackend
from app.cache.models import CacheResult
from app.services.retrieval.embedding import embed_query


class SemanticCache:
    """FAQ similarity lookup placed before the LangGraph pipeline."""

    def __init__(self, backend: Optional[CacheBackend] = None):
        self.enabled = settings.CACHE_ENABLED
        self.threshold = settings.CACHE_SIMILARITY_THRESHOLD
        self.collection_name = settings.CACHE_COLLECTION
        self.backend = backend if backend is not None else QdrantCacheBackend()

        if self.enabled:
            self.backend.initialize()
            logfire.info(
                f"FAQ cache enabled (threshold={self.threshold}, "
                f"collection={self.collection_name})"
            )
        else:
            logfire.info("FAQ cache disabled via configuration.")

    def lookup(self, question: str) -> CacheResult:
        if not self.enabled:
            return CacheResult(hit=False)

        with logfire.span("FAQ Cache Lookup", question=question[:80]):
            try:
                with logfire.span("Generate Query Embedding"):
                    query_embedding = embed_query(question)
                return self.backend.lookup(query_embedding, self.threshold)
            except Exception as e:
                logfire.error(f"FAQ cache lookup error: {e}")
                return CacheResult(hit=False)


_cache_instance: Optional[SemanticCache] = None


def get_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance


def reset_cache() -> None:
    global _cache_instance
    _cache_instance = None
