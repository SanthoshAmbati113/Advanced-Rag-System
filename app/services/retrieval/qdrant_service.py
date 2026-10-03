import logfire

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_query


client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

_sparse_model = None


def get_sparse_model():
    global _sparse_model

    if _sparse_model is None:
        logfire.info("Initializing sparse BM25 query model...")
        _sparse_model = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )

    return _sparse_model


def embed_sparse_query(query: str) -> models.SparseVector:
    model = get_sparse_model()
    embedding = next(model.query_embed(query))

    return models.SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )


def reciprocal_rank_fusion(
    dense_points,
    sparse_points,
    k: int = 60,
):
    """Combine dense and sparse rankings using RRF."""
    fused = {}

    for points in (dense_points, sparse_points):
        for rank, point in enumerate(points, start=1):
            point_id = str(point.id)

            if point_id not in fused:
                payload = point.payload or {}

                fused[point_id] = {
                    "id": point_id,
                    "content": payload.get("text", ""),
                    "source": payload.get("source", "Unknown"),
                    "source_type": payload.get("source_type", "Unknown"),
                    "score": 0.0,
                }

            fused[point_id]["score"] += 1.0 / (k + rank)

    return sorted(
        fused.values(),
        key=lambda item: item["score"],
        reverse=True,
    )


def search_enterprise_knowledge(
    query: str,
    limit: int = 15,
    candidates_per_retriever: int = 20,
):
    """Retrieve dense and sparse candidates, then fuse with RRF."""
    try:
        # Dense semantic search
        dense_vector = embed_query(query)

        dense_response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=dense_vector,
            using="dense",
            limit=candidates_per_retriever,
            with_payload=True,
        )

        # Sparse BM25 search
        sparse_vector = embed_sparse_query(query)

        sparse_response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=sparse_vector,
            using="bm25",
            limit=candidates_per_retriever,
            with_payload=True,
        )

        results = reciprocal_rank_fusion(
            dense_response.points,
            sparse_response.points,
        )

        results = results[:limit]

        logfire.info(
            "Hybrid retrieval completed",
            dense_candidates=len(dense_response.points),
            sparse_candidates=len(sparse_response.points),
            fused_candidates=len(results),
        )

        return results

    except Exception:
        logfire.exception("Hybrid Qdrant search failed")
        return []
