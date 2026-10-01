"""
Data models for the static FAQ cache.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheEntry:
    """A frozen FAQ matched at lookup time."""

    question: str
    answer: str
    embedding: list[float]
    faq_id: str = ""
    source: str = ""
    topic: str = ""
    variant: str = ""


@dataclass
class CacheResult:
    """Result of an FAQ similarity lookup."""

    hit: bool
    answer: Optional[str] = None
    similarity_score: Optional[float] = None
    latency_saved_ms: Optional[float] = None
    entry: Optional[CacheEntry] = None
