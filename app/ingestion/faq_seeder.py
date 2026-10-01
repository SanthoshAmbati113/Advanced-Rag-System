"""
One-time seeder for the static FAQ cache collection.

Reads DATA/faq/faqs.json and embeds each canonical question (aliases in the
JSON are ignored). Query-time lookup embeds the incoming question and does
cosine search against these FAQ vectors only.

Re-runs replace the collection by default so leftover alias points cannot linger.

Usage:
  python -m app.ingestion.faq_seeder
  python -m app.ingestion.faq_seeder DATA/faq/faqs.json
  python -m app.ingestion.faq_seeder DATA/faq/faqs.json --no-wipe
"""

from __future__ import annotations

import json
import os
import sys
import uuid
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_texts, get_embedding_dim

logfire.configure(service_name="faq-cache-seeder")

_FAQ_ID_NAMESPACE = uuid.UUID("7b1c9e2a-4f33-4c0d-9a6e-2d8f1b0c5e77")
DEFAULT_FAQ_PATH = os.path.join("DATA", "faq", "faqs.json")


def _point_id(faq_id: str) -> str:
    return str(uuid.uuid5(_FAQ_ID_NAMESPACE, faq_id))


def _canonical_faq(faq: dict) -> tuple[str, str, str]:
    """Return (faq_id, question, answer). Aliases are not used."""
    faq_id = (faq.get("id") or "").strip()
    question = (faq.get("question") or "").strip()
    answer = (faq.get("answer") or "").strip()
    if not faq_id or not question:
        raise ValueError(f"FAQ is missing required id/question: {faq!r}")
    if not answer:
        raise ValueError(f"FAQ '{faq_id}' is missing an answer.")
    return faq_id, question, answer


def _ensure_collection(client: QdrantClient, wipe: bool) -> None:
    name = settings.CACHE_COLLECTION
    exists = client.collection_exists(name)
    if wipe and exists:
        client.delete_collection(name)
        logfire.info(f"Deleted FAQ collection '{name}'.")
        exists = False
    if not exists:
        dim = get_embedding_dim()
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
        )
        logfire.info(f"Created FAQ collection '{name}' ({dim}-dim, Cosine).")
    else:
        logfire.info(f"FAQ collection '{name}' already exists.")


def seed_faqs(faq_path: str, wipe: bool = True) -> int:
    """Embed one vector per FAQ question and upsert. Returns points written."""
    with open(faq_path, encoding="utf-8") as f:
        faqs = json.load(f)
    if not isinstance(faqs, list) or not faqs:
        raise ValueError(f"Expected a non-empty JSON array in {faq_path}")

    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    _ensure_collection(client, wipe=wipe)

    records = [_canonical_faq(faq) for faq in faqs]
    texts = [question for _, question, _ in records]

    logfire.info(f"Embedding {len(texts)} canonical FAQ questions (aliases skipped).")
    embeddings = embed_texts(texts)
    if len(embeddings) != len(texts):
        raise RuntimeError("Embedding count does not match FAQ count.")

    points = []
    for faq, (_, question, answer), vector in zip(faqs, records, embeddings):
        faq_id = faq["id"].strip()
        points.append(
            models.PointStruct(
                id=_point_id(faq_id),
                vector=vector,
                payload={
                    "faq_id": faq_id,
                    "question": question,
                    "answer": answer,
                    "source": faq.get("source") or "",
                    "topic": faq.get("topic") or "",
                },
            )
        )

    client.upsert(collection_name=settings.CACHE_COLLECTION, points=points)
    logfire.info(
        f"Seeded {len(points)} static FAQ points into '{settings.CACHE_COLLECTION}'."
    )
    return len(points)


if __name__ == "__main__":
    wipe_requested = "--no-wipe" not in sys.argv
    clean_args = [a for a in sys.argv if a not in ("--wipe", "--no-wipe")]
    faq_path = clean_args[1] if len(clean_args) > 1 else DEFAULT_FAQ_PATH

    if not os.path.exists(faq_path):
        print(f"Error: FAQ file '{faq_path}' does not exist.")
        sys.exit(1)

    count = seed_faqs(faq_path, wipe=wipe_requested)
    print(f"Seeded {count} FAQ vectors into '{settings.CACHE_COLLECTION}'.")
