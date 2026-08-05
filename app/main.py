# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, Response
from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard
from app.cache import get_cache

from pydantic import BaseModel
from typing import Optional


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")


@app.on_event("startup")
def startup_event():
    initialize_rails()

class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"
    
    
@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    
    Flow:
    1. Guardrails check (blocks malicious/off-topic queries)
    2. Semantic cache lookup (bypasses RAG pipeline on hit)
    3. LangGraph RAG pipeline (on cache miss)
    4. Cache storage (after successful RAG generation)
    """
    q = request.q
    thread_id = request.thread_id
    
    # Get semantic cache instance
    cache = get_cache()

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }
    
    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": []
            }

        # Gate 2: Semantic Cache — checks for similar cached responses
        # This runs BEFORE the expensive RAG pipeline to reduce latency and token costs
        with logfire.span("Cache Check"):
            cache_result = cache.lookup(q)
            
            if cache_result.hit:
                # Cache hit - return cached answer immediately
                logfire.info(
                    f"💾 Cache hit: similarity={cache_result.similarity_score:.4f}, "
                    f"latency_saved={cache_result.latency_saved_ms:.0f}ms"
                )
                return {
                    "question": q,
                    "answer": cache_result.answer,
                    "thought_process": [
                        "Intent: Cache Hit",
                        f"Similarity: {cache_result.similarity_score:.4f}",
                        f"Latency Saved: {cache_result.latency_saved_ms:.0f}ms"
                    ],
                    "status": "Answer retrieved from cache.",
                    "sources": [],
                    "cache_metadata": {
                        "similarity_score": cache_result.similarity_score,
                        "latency_saved_ms": cache_result.latency_saved_ms,
                        "model_name": cache_result.entry.model_name if cache_result.entry else None,
                        "cached_at": cache_result.entry.timestamp.isoformat() if cache_result.entry else None
                    }
                }
            else:
                logfire.info("💾 Cache miss - proceeding with RAG pipeline")

        # Gate 3: LangGraph RAG pipeline (on cache miss)
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)
        
        final_answer = final_output.get("final_answer")
        
        # Store the result in cache for future queries
        # This happens after successful RAG pipeline execution
        with logfire.span("Cache Storage"):
            cache.store(
                question=q,
                answer=final_answer,
                model_name="llama-3.3-70b-versatile",
                document_version="v1"
            )

        return {
            "question": q,
            "answer": final_answer,
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }
