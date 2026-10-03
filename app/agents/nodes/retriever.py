import logfire

from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import (
    search_enterprise_knowledge,
)
from app.services.retrieval.ranking_service import (
    rerank_documents,
)


def retrieve_node(state: AgentState):
    """
    Hybrid retrieval followed by FlashRank reranking.
    Returns only content chunks, without citation formatting.
    """
    query = state["current_query"]

    with logfire.span("Knowledge Retrieval"):
        logfire.info(f"Hybrid search for: {query}")

        # Dense + sparse retrieval with RRF fusion
        raw_results = search_enterprise_knowledge(
            query,
            limit=15,
            candidates_per_retriever=20,
        )

        logfire.info(
            "Retrieved fused candidates",
            candidates=len(raw_results),
        )

        doc_contents = [
            doc["content"]
            for doc in raw_results
            if doc.get("content")
        ]

        if not doc_contents:
            return {
                "documents": [],
                "status": "No relevant technical context found.",
                "plan": state["plan"] + ["Context Retrieval Empty"],
            }

        # FlashRank reranks the fused candidates
        with logfire.span("FlashRank Reranking"):
            reranked_contents = rerank_documents(
                query=query,
                documents=doc_contents,
                top_n=5,
            )

        # No source labels or citation formatting
        formatted_docs = [
            f"CONTENT: {doc}"
            for doc in reranked_contents
        ]

        logfire.info(
            "Reranking completed",
            selected_documents=len(formatted_docs),
        )

    return {
        "documents": formatted_docs,
        "status": "Hybrid retrieval and reranking completed.",
        "plan": state["plan"] + ["Hybrid Context Retrieved"],
    }
