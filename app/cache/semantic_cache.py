"""
Semantic Cache - Main orchestration layer for response caching.

This module provides the main interface for semantic response caching in the RAG system.
It handles the cache lookup and storage logic, integrating with the RAG pipeline to
reduce latency and token costs by caching similar queries.

The cache layer is designed to be:
- Optional: Can be disabled via configuration
- Modular: Backend-agnostic via CacheBackend interface
- Non-blocking: Cache failures don't break the pipeline
- Observable: Full Logfire tracing for all operations
"""

import logfire
from datetime import datetime
from typing import Optional

from app.config import settings
from app.cache.cache_service import CacheBackend
from app.cache.qdrant_cache import QdrantCacheBackend
from app.cache.models import CacheEntry, CacheResult
from app.services.retrieval.embedding import embed_query


class SemanticCache:
    """
    Main semantic cache orchestrator.
    
    This class provides a high-level interface for caching RAG responses.
    It handles:
    - Cache initialization and configuration
    - Query embedding generation
    - Cache lookup with similarity threshold
    - Cache storage after successful RAG pipeline execution
    - Full observability via Logfire spans
    
    The cache is designed to be inserted BEFORE the retrieval stage in app/main.py,
    allowing similar queries to bypass the entire RAG pipeline.
    """
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        """
        Initialize the semantic cache.
        
        Args:
            backend: Optional custom cache backend. If None, uses QdrantCacheBackend.
                    This allows easy swapping to Redis or other backends.
        """
        self.enabled = settings.CACHE_ENABLED
        self.threshold = settings.CACHE_SIMILARITY_THRESHOLD
        self.collection_name = settings.CACHE_COLLECTION
        
        # Initialize backend (default to Qdrant, but allow injection)
        self.backend = backend if backend is not None else QdrantCacheBackend()
        
        if self.enabled:
            self.backend.initialize()
            logfire.info(
                f"Semantic cache enabled (threshold={self.threshold}, "
                f"collection={self.collection_name})"
            )
        else:
            logfire.info("Semantic cache disabled via configuration.")
    
    def lookup(self, question: str) -> CacheResult:
        """
        Look up a cached answer for the given question.
        
        This method:
        1. Generates an embedding for the question
        2. Searches the cache for similar questions
        3. Returns the cached answer if similarity exceeds threshold
        
        Args:
            question: The user's question
            
        Returns:
            CacheResult with hit status and cached data if found
        """
        if not self.enabled:
            return CacheResult(hit=False)
        
        with logfire.span("Semantic Cache Lookup", question=question[:80]):
            try:
                # Generate embedding for the query
                with logfire.span("Generate Query Embedding"):
                    query_embedding = embed_query(question)
                
                # Perform cache lookup
                result = self.backend.lookup(query_embedding, self.threshold)
                
                return result
                
            except Exception as e:
                logfire.error(f"Cache lookup error: {e}")
                # On error, treat as cache miss and continue with pipeline
                return CacheResult(hit=False)
    
    def store(
        self,
        question: str,
        answer: str,
        model_name: str,
        document_version: str = "v1"
    ) -> None:
        """
        Store a question-answer pair in the cache.
        
        This method should be called AFTER the RAG pipeline successfully
        generates an answer. It stores the Q&A pair with metadata for
        future cache hits.
        
        Args:
            question: The original user question
            answer: The generated answer from the RAG pipeline
            model_name: The LLM model used to generate the answer
            document_version: Version identifier for the document collection
        """
        if not self.enabled:
            return
        
        with logfire.span("Semantic Cache Store", question=question[:50]):
            try:
                # Generate embedding for storage
                with logfire.span("Generate Storage Embedding"):
                    embedding = embed_query(question)
                
                # Create cache entry
                entry = CacheEntry(
                    question=question,
                    answer=answer,
                    embedding=embedding,
                    timestamp=datetime.utcnow(),
                    model_name=model_name,
                    document_version=document_version,
                    hit_count=0
                )
                
                # Store in backend
                self.backend.store(entry)
                
            except Exception as e:
                logfire.error(f"Cache store error: {e}")
                # Don't raise - cache failures shouldn't break the pipeline


# Global cache instance (initialized on module import)
# This follows the singleton pattern for easy integration
_cache_instance: Optional[SemanticCache] = None


def get_cache() -> SemanticCache:
    """
    Get the global semantic cache instance.
    
    This function provides lazy initialization of the cache, following
    the existing dependency injection pattern in the project.
    
    Returns:
        The global SemanticCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance


def reset_cache() -> None:
    """
    Reset the global cache instance.
    
    This is primarily useful for testing or when configuration changes.
    """
    global _cache_instance
    _cache_instance = None
