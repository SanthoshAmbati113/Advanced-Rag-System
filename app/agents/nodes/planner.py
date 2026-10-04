import logfire
from app.agents.state import AgentState
from langchain_groq import ChatGroq
from app.config import settings
# from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file

# Direct Groq LLM integration
llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model="openai/gpt-oss-safeguard-20b",
    temperature=0
)
# model=HuggingFaceEndpoint(repo_id='Qwen/Qwen3-4B-Instruct-2507',huggingfacehub_api_token=os.getenv("HF_TOKEN"))
# llm=ChatHuggingFace(llm=model,temperature=0)


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
You are a Routing Agent for a Kubernetes RAG Assistant.
The assistant that answers the user DOES NOT HAVE ACCESS TO YOUR KNOWLEDGE.

The ONLY source of technical information is the Kubernetes knowledge base.

Therefore, every Kubernetes-related question MUST be routed to retrieval, even if you personally know the answer.

Never use your own knowledge to decide whether retrieval is needed.

Your ONLY responsibility is to decide whether the user's latest message should:
1. Be answered directly from the conversation history.
2. Search the Kubernetes knowledge base.

IMPORTANT:
- Do NOT answer the user's question.
- Do NOT use your own knowledge to determine whether Kubernetes information is "known".
- If the user is asking about ANY Kubernetes concept, ALWAYS choose retrieval.
- The knowledge base contains Kubernetes documentation, so Kubernetes-related questions should always be routed there.

CONVERSATION HISTORY:
{history}

LATEST MESSAGE:
"{user_message}"

Rules:

Return exactly "CONVERSATIONAL" ONLY if the latest message is:
- a greeting (hi, hello, hey)
- thanks
- goodbye
- casual conversation
- a follow-up that can be answered completely using the conversation history
- a question like "What is my name?" where the answer exists in the conversation history

For ALL Kubernetes-related questions, return ONLY a concise search query.

Examples:

User: Can you explain that with an example?
Output:
CONVERSATIONAL

User: What are the advantages of this approach?
Output:
CONVERSATIONAL

User: What is a Kubernetes Pod?
Output:
Kubernetes Pod

User: Explain Deployments
Output:
Kubernetes Deployment

User: What is a ReplicaSet?
Output:
Kubernetes ReplicaSet

User: What is kubectl?
Output:
kubectl

User: Explain ConfigMaps
Output:
Kubernetes ConfigMap

Return ONLY:
- CONVERSATIONAL
OR
- a concise search query

Do not include explanations.
"""

    with logfire.span("🧠 Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")

    if decision.strip().upper().startswith("CONVERSATIONAL"):
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
