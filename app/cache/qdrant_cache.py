"""
Read-only Qdrant backend for the static FAQ collection.

The collection is populated only by `python -m app.ingestion.faq_seeder`.
Query-time code never upserts points or mutates payloads.
"""

import logfire
from qdrant_client import QdrantClient

from app.config import settings
from app.cache.cache_service import CacheBackend
from app.cache.models import CacheEntry, CacheResult


class QdrantCacheBackend(CacheBackend):
    """Cosine search over pre-seeded FAQ question/alias vectors."""

    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self.collection_name = settings.CACHE_COLLECTION
        self._initialized = False

    def initialize(self) -> None:
        """Verify the FAQ collection exists. Do not create or write to it."""
        if self._initialized:
            return

        with logfire.span("FAQ Cache Initialization", collection=self.collection_name):
            try:
                if self.client.collection_exists(self.collection_name):
                    logfire.info(f"FAQ collection '{self.collection_name}' is ready.")
                else:
                    logfire.warning(
                        f"FAQ collection '{self.collection_name}' does not exist. "
                        "Lookups will miss until you run "
                        "`python -m app.ingestion.faq_seeder --wipe`."
                    )
            except Exception as e:
                logfire.error(f"FAQ collection check failed: {e}")
            self._initialized = True

    def lookup(self, query_embedding: list[float], threshold: float) -> CacheResult:
        with logfire.span("FAQ Cache Lookup", threshold=threshold):
            try:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=1,
                    with_payload=True,
                )

                if not response.points:
                    logfire.info("FAQ cache miss: no entries found.")
                    return CacheResult(hit=False)

                top_point = response.points[0]
                similarity_score = top_point.score
                if similarity_score < threshold:
                    logfire.info(
                        f"FAQ cache miss: similarity {similarity_score:.4f} "
                        f"below threshold {threshold}"
                    )
                    return CacheResult(hit=False)

                payload = top_point.payload or {}
                entry = CacheEntry(
                    question=payload.get("question", ""),
                    answer=payload.get("answer", ""),
                    embedding=query_embedding,
                    faq_id=payload.get("faq_id", ""),
                    source=payload.get("source", ""),
                    topic=payload.get("topic", ""),
                    variant=payload.get("variant") or payload.get("question", ""),
                )
                latency_saved_ms = 2000.0
                logfire.info(
                    f"FAQ cache hit: id={entry.faq_id} "
                    f"similarity={similarity_score:.4f}"
                )
                return CacheResult(
                    hit=True,
                    answer=entry.answer,
                    similarity_score=similarity_score,
                    latency_saved_ms=latency_saved_ms,
                    entry=entry,
                )
            except Exception as e:
                logfire.error(f"FAQ cache lookup failed: {e}")
                return CacheResult(hit=False)
