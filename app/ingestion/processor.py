import os
import sys
import uuid
import json
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding

from app.config import settings
from app.services.retrieval.embedding import (
    embed_texts,
    get_embedding_dim,
)
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_text


logfire.configure(service_name="enterprise-ingestion-service")


# Local folder for processed chunk metadata
PROCESSED_DATA_DIR = "processed_data"

# Sparse embedding model
SPARSE_MODEL_NAME = "Qdrant/bm25"
_sparse_model = None

# Qdrant client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def get_sparse_model():
    """Initialize the sparse BM25 model lazily."""
    global _sparse_model

    if _sparse_model is None:
        logfire.info(
            "Initializing FastEmbed sparse model",
            model=SPARSE_MODEL_NAME,
        )
        _sparse_model = SparseTextEmbedding(
            model_name=SPARSE_MODEL_NAME
        )

    return _sparse_model


def embed_sparse_texts(texts: list[str]) -> list[dict]:
    """
    Generate sparse vectors for document chunks.

    Each vector contains token indices and their weights.
    """
    if not texts:
        return []

    model = get_sparse_model()
    embeddings = list(model.embed(texts))

    return [
        {
            "indices": embedding.indices.tolist(),
            "values": embedding.values.tolist(),
        }
        for embedding in embeddings
    ]


def save_processed_locally(
    data: dict,
    source_type: str,
    filename: str,
) -> str:
    """Save parsed and chunked metadata locally."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)

    # Avoid directory traversal and nested paths in filenames.
    safe_filename = os.path.basename(filename)
    dest = os.path.join(folder, f"{safe_filename}.json")

    with open(dest, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return dest


def create_hybrid_collection():
    """
    Create a Qdrant collection with named dense and sparse vectors.

    Dense vector: Gemini or the configured fallback embedding model.
    Sparse vector: FastEmbed BM25.
    """
    if qdrant_client.collection_exists(
        settings.QDRANT_COLLECTION
    ):
        return

    dimension = get_embedding_dim()

    qdrant_client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False
                ),
                modifier=models.Modifier.IDF,
            )
        },
    )

    logfire.info(
        "Created hybrid Qdrant collection",
        collection=settings.QDRANT_COLLECTION,
        dense_dimension=dimension,
        sparse_vector="bm25",
    )


def process_file(
    file_path: str,
    filename: str,
    source_type: str,
):
    """Parse → chunk → save → embed → index dense and sparse vectors."""
    with logfire.span(
        "Processing File",
        file=filename,
        source=source_type,
    ):
        try:
            # 1. Extract text
            extension = filename.lower().rsplit(".", 1)[-1]

            if extension == "pdf":
                full_text = parse_pdf(file_path)

            elif extension in ("html", "htm"):
                full_text = parse_html(file_path)

            elif extension == "txt":
                full_text = parse_text(file_path)

            elif extension in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)

            else:
                logfire.warning(
                    "Skipping unsupported file type",
                    filename=filename,
                )
                return

            if not full_text or not full_text.strip():
                logfire.warning(
                    "No text extracted; skipping file",
                    filename=filename,
                )
                return

            # 2. Chunk the document
            chunks = chunk_text(full_text)

            if not chunks:
                logfire.warning(
                    "No chunks generated",
                    filename=filename,
                )
                return

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }

            local_path = save_processed_locally(
                processed_data,
                source_type,
                filename,
            )

            logfire.info(
                "Saved processed data",
                path=local_path,
            )

            # 4. Generate dense and sparse representations
            with logfire.span("Vectorizing & Indexing"):
                dense_embeddings = embed_texts(chunks)
                sparse_embeddings = embed_sparse_texts(chunks)

                if not (
                    len(chunks)
                    == len(dense_embeddings)
                    == len(sparse_embeddings)
                ):
                    raise ValueError(
                        "Chunk, dense embedding, and sparse "
                        "embedding counts do not match."
                    )

                # 5. Build Qdrant points
                points = []

                for chunk, dense_vector, sparse_vector in zip(
                    chunks,
                    dense_embeddings,
                    sparse_embeddings,
                ):
                    point = models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": dense_vector,
                            "bm25": models.SparseVector(
                                indices=sparse_vector["indices"],
                                values=sparse_vector["values"],
                            ),
                        },
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                        },
                    )

                    points.append(point)

                # 6. Upsert into Qdrant
                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                    wait=True,
                )

                logfire.info(
                    "Indexed hybrid document chunks",
                    filename=filename,
                    chunk_count=len(points),
                )

        except Exception:
            logfire.exception(
                "Failed to process file",
                filename=filename,
            )


def process_directory(
    dir_path: str,
    source_type: str,
):
    """Process all files directly inside a directory."""
    with logfire.span(
        "Scanning Directory",
        path=dir_path,
        source=source_type,
    ):
        files = [
            filename
            for filename in os.listdir(dir_path)
            if os.path.isfile(
                os.path.join(dir_path, filename)
            )
        ]

        logfire.info(
            "Files discovered",
            directory=dir_path,
            count=len(files),
        )

        for filename in files:
            process_file(
                os.path.join(dir_path, filename),
                filename,
                source_type,
            )


def run_universal_ingestion(
    base_dir: str,
    explicit_source_type: str = None,
    wipe: bool = False,
):
    """
    Scan a directory and ingest its documents into Qdrant.

    --wipe deletes the configured collection and recreates it
    with dense and sparse vector configurations.
    """
    if not os.path.isdir(base_dir):
        raise NotADirectoryError(
            f"Input directory does not exist: {base_dir}"
        )

    with logfire.span(
        "Universal Ingestion Started",
        base_directory=base_dir,
    ):
        # 1. Optionally rebuild the collection
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(
                    settings.QDRANT_COLLECTION
                ):
                    qdrant_client.delete_collection(
                        settings.QDRANT_COLLECTION
                    )

                    logfire.info(
                        "Deleted Qdrant collection",
                        collection=settings.QDRANT_COLLECTION,
                    )

        # 2. Ensure the hybrid collection exists
        create_hybrid_collection()

        # 3. Route subdirectories to source types
        subdirs = [
            directory
            for directory in os.listdir(base_dir)
            if os.path.isdir(
                os.path.join(base_dir, directory)
            )
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(
                    os.path.normpath(base_dir)
                ).lower()

                source_type = (
                    "true"
                    if "true" in base_name
                    else "noisy"
                    if "noisy" in base_name
                    else "general"
                )

            logfire.info(
                "Processing input directory",
                directory=base_dir,
                source_type=source_type,
            )

            process_directory(base_dir, source_type)

        else:
            for subdir in subdirs:
                source_type = (
                    "true"
                    if "true" in subdir.lower()
                    else "noisy"
                    if "noisy" in subdir.lower()
                    else subdir
                )

                process_directory(
                    os.path.join(base_dir, subdir),
                    source_type,
                )


if __name__ == "__main__":
    # Examples:
    # python -m app.ingestion.processor DATA --wipe
    # python -m app.ingestion.processor DATA/true_data true

    wipe_requested = "--wipe" in sys.argv
    clean_args = [
        argument
        for argument in sys.argv
        if argument != "--wipe"
    ]

    target_dir = (
        clean_args[1]
        if len(clean_args) > 1
        else "DATA"
    )

    explicit_type = (
        clean_args[2]
        if len(clean_args) > 2
        else None
    )

    if not os.path.isdir(target_dir):
        print(
            f"Error: directory '{target_dir}' does not exist."
        )
        sys.exit(1)

    run_universal_ingestion(
        target_dir,
        explicit_source_type=explicit_type,
        wipe=wipe_requested,
    )

    logfire.info("Ingestion job completed.")
