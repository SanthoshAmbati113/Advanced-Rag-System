import os
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv


# Load local environment variables for development.
# Render environment variables are read automatically from os.environ.
env_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".env")
)
load_dotenv(dotenv_path=env_path)


# --- BACKEND CONFIGURATION ---
BACKEND_URL = os.getenv("BACKEND_API_URL", "").rstrip("/")


# --- LOGFIRE INITIALIZATION ---
try:
    token = os.getenv("LOGFIRE_TOKEN")

    if token:
        logfire.configure(token=token)
        LOGFIRE_STATUS = "Connected & Tracing"
    else:
        LOGFIRE_STATUS = "Disabled (No token configured)"

except Exception as e:
    print(f"Logfire initialization error in UI: {e}")
    LOGFIRE_STATUS = "Standby"


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)


# --- AVATARS ---
AI_AVATAR = "🤖"
USER_AVATAR = "👤"


# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

    try:
        logfire.info(
            "New user session created",
            session_id=st.session_state.session_id,
        )
    except Exception:
        pass

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")

    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")

    if st.button(
        "🗑️ Clear History & Memory",
        width="stretch",
        type="primary",
    ):
        try:
            logfire.warning(
                "Memory wipe triggered",
                session_id=st.session_state.session_id,
            )
        except Exception:
            pass

        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# --- MAIN CHAT ---
st.title("🤖 Enterprise Agentic Assistant")


# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    avatar = (
        AI_AVATAR
        if message["role"] == "assistant"
        else USER_AVATAR
    )

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# --- CHAT INPUT ---
if prompt := st.chat_input("Ask about your documentation..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AI_AVATAR):
        answer_placeholder = st.empty()

        try:
            # Validate backend configuration
            if not BACKEND_URL:
                raise ValueError(
                    "BACKEND_API_URL is missing. "
                    "Configure it in your Render frontend "
                    "service environment variables."
                )

            with st.status(
                "🔍 Agent is thinking...",
                expanded=True,
            ) as status:

                # --- CALL FASTAPI BACKEND ---
                with logfire.span("Calling RAG Backend"):
                    url = f"{BACKEND_URL}/query"

                    payload = {
                        "q": prompt,
                        "thread_id": st.session_state.session_id,
                    }

                    response = requests.post(
                        url,
                        json=payload,
                        timeout=120,
                    )

                    response.raise_for_status()
                    data = response.json()

                # --- SHOW REASONING STEPS ---
                steps = data.get("thought_process", [])

                for step in steps:
                    st.write(f"⚙️ {step}")

                status.update(
                    label="✅ Answer received",
                    state="complete",
                    expanded=False,
                )

                # --- OPTIONAL RETRIEVED CONTEXT ---
                sources = data.get("sources", [])

                if sources:
                    with st.expander(
                        "📄 View Retrieved Context"
                    ):
                        for i, source in enumerate(sources):
                            source_text = str(source)
                            preview = (
                                source_text[:100]
                                .replace("\n", " ")
                            )

                            if len(source_text) > 100:
                                preview += "..."

                            with st.expander(
                                f"Chunk {i + 1}: {preview}"
                            ):
                                st.info(source_text)

            # --- GET FINAL ANSWER ---
            full_answer = data.get(
                "answer",
                "No response received from the backend.",
            )

            if not isinstance(full_answer, str):
                full_answer = str(full_answer)

            # --- STREAMING DISPLAY EFFECT ---
            curr_text = ""

            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(
                    curr_text + "▌"
                )
                time.sleep(0.005)

            answer_placeholder.markdown(full_answer)

            # Save assistant response to chat history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_answer,
                }
            )

            try:
                logfire.info("Chat cycle completed successfully")
            except Exception:
                pass

        except requests.exceptions.Timeout:
            answer_placeholder.error(
                "The backend took too long to respond. "
                "Please try again in a moment."
            )

        except requests.exceptions.HTTPError as e:
            response_text = ""

            try:
                response_text = e.response.text[:1000]
            except Exception:
                pass

            answer_placeholder.error(
                f"The backend returned an HTTP error: {e}"
            )

            if response_text:
                with st.expander("Error details"):
                    st.code(response_text)

            try:
                logfire.exception("Backend HTTP error")
            except Exception:
                pass

        except requests.exceptions.RequestException as e:
            answer_placeholder.error(
                f"Could not connect to the backend: {e}"
            )

            try:
                logfire.exception("Backend connection failed")
            except Exception:
                pass

        except ValueError as e:
            answer_placeholder.error(str(e))

        except Exception as e:
            answer_placeholder.error(
                f"An unexpected error occurred: {e}"
            )

            try:
                logfire.exception("Unexpected UI error")
            except Exception:
                pass
