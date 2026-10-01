import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

BATCH_SIZE = 8
BATCH_PAUSE_SECONDS = 2.0
_MAX_RETRIES = 8
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  # all-mpnet-base-v2

_active_model = None
_model_type: str | None = None  # "gemini" or "fallback"


# ── Model initialisation ───────────────────────────────────────────────────────

def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query("probe")
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
        return model
    except Exception as e:
        logfire.warning(f"Gemini probe failed: {e}. Will use sentence-transformers fallback.")
        return None


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


# ── Batch embedding with retry ─────────────────────────────────────────────────

def _is_rate_limit(exc: Exception) -> bool:
    err = str(exc).lower()
    return any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))


def _backoff_seconds(attempt: int) -> float:
    # 5, 10, 20, 40, ... capped at 60s
    return min(60.0, 5.0 * (2 ** attempt))


def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        for attempt in range(_MAX_RETRIES):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                if _is_rate_limit(e) and attempt < _MAX_RETRIES - 1:
                    wait = _backoff_seconds(attempt)
                    logfire.warning(
                        f"Gemini rate limit hit — retrying in {wait:.0f}s "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES})."
                    )
                    time.sleep(wait)
                    continue
                logfire.error(f"Gemini embedding failed: {e}")
                raise
        raise RuntimeError(f"Gemini rate limit persisted after {_MAX_RETRIES} attempts.")
    return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type != "gemini":
        return _active_model.encode([query])[0].tolist()
    for attempt in range(_MAX_RETRIES):
        try:
            return _active_model.embed_query(query)
        except Exception as e:
            if _is_rate_limit(e) and attempt < _MAX_RETRIES - 1:
                wait = _backoff_seconds(attempt)
                logfire.warning(
                    f"Gemini query embed rate-limited — retrying in {wait:.0f}s "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})."
                )
                time.sleep(wait)
                continue
            raise


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    batches = list(range(0, len(texts), BATCH_SIZE))
    for n, i in enumerate(batches):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
        if _model_type == "gemini" and n < len(batches) - 1:
            time.sleep(BATCH_PAUSE_SECONDS)
    return all_embeddings
