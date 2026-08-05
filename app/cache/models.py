"""
Data models for semantic caching.

This module defines the data structures used for caching question-answer pairs
and cache lookup results.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class CacheEntry:
    """
    Represents a cached question-answer pair stored in the cache.
    
    Attributes:
        question: The original user question
        answer: The generated answer from the RAG pipeline
        embedding: The vector embedding of the question
        timestamp: When this entry was created
        model_name: The LLM model used to generate the answer
        document_version: Version identifier for the document collection
        hit_count: Number of times this cache entry has been retrieved
    """
    question: str
    answer: str
    embedding: list[float]
    timestamp: datetime
    model_name: str
    document_version: str
    hit_count: int = 0


@dataclass
class CacheResult:
    """
    Result of a cache lookup operation.
    
    Attributes:
        hit: Whether the cache lookup was successful
        answer: The cached answer (if hit=True)
        similarity_score: The cosine similarity score (if hit=True)
        latency_saved_ms: Estimated latency saved by cache hit (if hit=True)
        entry: The full cache entry (if hit=True)
    """
    hit: bool
    answer: Optional[str] = None
    similarity_score: Optional[float] = None
    latency_saved_ms: Optional[float] = None
    entry: Optional[CacheEntry] = None
