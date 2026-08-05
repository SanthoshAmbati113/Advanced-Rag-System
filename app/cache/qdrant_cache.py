"""
Qdrant-based implementation of the cache backend.

This module provides a concrete implementation of CacheBackend using Qdrant
as the storage backend. It stores question-answer pairs as vectors and
performs similarity search using cosine distance.
"""

import uuid
import logfire
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.config import settings
from app.cache.cache_service import CacheBackend
from app.cache.models import CacheEntry, CacheResult


class QdrantCacheBackend(CacheBackend):
    """
    Qdrant-based cache backend implementation.
    
    Stores cached Q&A pairs in a separate Qdrant collection with vector
    similarity search for cache lookup. This implementation is designed to
    be easily replaceable with Redis or other backends via the CacheBackend interface.
    """
    
    def __init__(self):
        """Initialize the Qdrant cache backend."""
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = settings.CACHE_COLLECTION
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize the cache collection in Qdrant.
        
        Creates the collection if it doesn't exist with appropriate
        vector configuration based on the embedding model dimension.
        """
        if self._initialized:
            return
        
        with logfire.span("Cache Initialization", collection=self.collection_name):
            from app.services.retrieval.embedding import get_embedding_dim
            
            vector_dim = get_embedding_dim()
            
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=vector_dim,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                )
                logfire.info(
                    f"Created cache collection '{self.collection_name}' "
                    f"({vector_dim}-dim, Cosine)."
                )
            else:
                logfire.info(f"Cache collection '{self.collection_name}' already exists.")
            
            self._initialized = True
    
    def lookup(self, query_embedding: list[float], threshold: float) -> CacheResult:
        """
        Look up a cached answer based on query embedding similarity.
        
        Performs a vector search in the cache collection and returns the
        highest-scoring match if it exceeds the similarity threshold.
        
        Args:
            query_embedding: The vector embedding of the user's query
            threshold: Minimum similarity score to consider a cache hit
            
        Returns:
            CacheResult with hit status and cached data if found
        """
        with logfire.span("Cache Lookup", threshold=threshold):
            try:
                # Search for similar cached queries
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=1,  # Only need the top match
                    with_payload=True
                )
                
                if not response.points:
                    logfire.info("Cache miss: No entries found.")
                    return CacheResult(hit=False)
                
                top_point = response.points[0]
                similarity_score = top_point.score
                
                # Check if similarity meets threshold
                if similarity_score >= threshold:
                    # Cache hit - increment hit count
                    self.increment_hit_count(top_point.id)
                    
                    payload = top_point.payload
                    entry = CacheEntry(
                        question=payload["question"],
                        answer=payload["answer"],
                        embedding=query_embedding,
                        timestamp=datetime.fromisoformat(payload["timestamp"]),
                        model_name=payload["model_name"],
                        document_version=payload["document_version"],
                        hit_count=payload.get("hit_count", 0) + 1
                    )
                    
                    # Estimate latency saved (average RAG pipeline ~2-3 seconds)
                    latency_saved_ms = 2000.0
                    
                    logfire.info(
                        f"Cache hit: similarity={similarity_score:.4f}, "
                        f"latency_saved={latency_saved_ms:.0f}ms"
                    )
                    
                    return CacheResult(
                        hit=True,
                        answer=entry.answer,
                        similarity_score=similarity_score,
                        latency_saved_ms=latency_saved_ms,
                        entry=entry
                    )
                else:
                    logfire.info(
                        f"Cache miss: similarity {similarity_score:.4f} below threshold {threshold}"
                    )
                    return CacheResult(hit=False)
                    
            except Exception as e:
                logfire.error(f"Cache lookup failed: {e}")
                # On error, treat as cache miss and continue with pipeline
                return CacheResult(hit=False)
    
    def store(self, entry: CacheEntry) -> None:
        """
        Store a new cache entry.
        
        Inserts the question-answer pair with its embedding into the cache collection.
        
        Args:
            entry: The CacheEntry to store
        """
        with logfire.span("Cache Store", question=entry.question[:50]):
            try:
                point_id = str(uuid.uuid4())
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        qdrant_models.PointStruct(
                            id=point_id,
                            vector=entry.embedding,
                            payload={
                                "question": entry.question,
                                "answer": entry.answer,
                                "timestamp": entry.timestamp.isoformat(),
                                "model_name": entry.model_name,
                                "document_version": entry.document_version,
                                "hit_count": entry.hit_count
                            }
                        )
                    ]
                )
                
                logfire.info(f"Stored cache entry: {point_id}")
                
            except Exception as e:
                logfire.error(f"Failed to store cache entry: {e}")
                # Don't raise - cache failures shouldn't break the pipeline
    
    def increment_hit_count(self, entry_id: str) -> None:
        """
        Increment the hit count for a cache entry.
        
        Args:
            entry_id: The unique identifier of the cache entry
        """
        try:
            # Retrieve the current payload
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[entry_id],
                with_payload=True
            )
            
            if points:
                payload = points[0].payload
                current_count = payload.get("hit_count", 0)
                payload["hit_count"] = current_count + 1
                
                # Update the payload
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=payload,
                    points=[entry_id]
                )
        except Exception as e:
            logfire.warning(f"Failed to increment hit count for {entry_id}: {e}")
