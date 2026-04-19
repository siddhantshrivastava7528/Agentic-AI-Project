"""
capstone_streamlit.py — AI Study Buddy Agent UI
Run: streamlit run capstone_streamlit.py
"""

import uuid
import os
import streamlit as st
from dotenv import load_dotenv

# ── LOAD ENV VARIABLES ───────────────────────────────────────────────────────
load_dotenv()

# ── PAGE CONFIG (MUST BE FIRST STREAMLIT CALL) ───────────────────────────────
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CHECK GROQ API KEY ───────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not set.\n\n👉 Create a `.env` file and add:\n\nGROQ_API_KEY=your_key_here")
    st.stop()

# ── IMPORT BACKEND (AFTER ENV IS READY) ──────────────────────────────────────
from agent import get_app, get_embedder, get_collection, get_llm, DOCUMENTS


# ══════════════════════════════════════════════════════════════════════════════
# 🔁 CACHED INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def initialize_resources():
    embedder   = get_embedder()
    collection = get_collection()
    llm        = get_llm()
    app        = get_app()
    return embedder, collection, llm, app


embedder, collection, llm, app = initialize_resources()


# ══════════════════════════════════════════════════════════════════════════════
# 🧠 SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []


# ══════════════════════════════════════════════════════════════════════════════
# 🎯 HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def score_badge(score: float) -> str:
    if score >= 0.85:
        return f"🟢 {score:.2f}"
    elif score >= 0.7:
        return f"🟡 {score:.2f}"
    return f"🔴 {score:.2f}"


def render_metadata(entry: dict):
    cols = st.columns(4)

    with cols[0]:
        st.caption(f"**Score:** {score_badge(entry.get('eval_score', 0.0))}")

    with cols[1]:
        route = entry.get("route", "—")
        icon = {
            "retrieve": "🔍",
            "tool": "🔧",
            "memory_only": "💬"
        }.get(route, "❓")
        st.caption(f"**Route:** {icon} {route}")

    with cols[2]:
        retries = entry.get("retry_count", 0)
        st.caption(f"**Retries:** {'🔁'*retries}{retries}")

    with cols[3]:
        sources = entry.get("sources", [])
        if sources:
            st.caption(f"**Sources:** {', '.join(sources[:2])}")
        else:
            st.caption("**Sources:** —")


# ══════════════════════════════════════════════════════════════════════════════
# 📚 SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📚 AI Study Buddy")
    st.caption("RAG + Tools + Agentic Reasoning")

    st.divider()

    st.subheader("📖 Knowledge Base")
    for doc in DOCUMENTS:
        st.markdown(f"- {doc.get('title', 'Untitled')}")

    st.divider()

    st.subheader("🛠 Tools")
    st.markdown(
        "- 🔢 Calculator\n"
        "- 🕒 Datetime\n"
        "- 🌐 Web Search"
    )

    st.divider()

    st.subheader("🔀 Routing")
    st.markdown(
        "| Route | Purpose |\n"
        "|------|--------|\n"
        "| retrieve | KB search |\n"
        "| tool | external tools |\n"
        "| memory_only | casual queries |"
    )

    st.divider()

    st.subheader("⚙️ Session")
    st.code(f"Thread: {st.session_state.thread_id[:12]}…")
    st.caption(f"Messages: {len(st.session_state.lc_messages)}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 New"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.lc_messages = []
            st.rerun()

    with col2:
        if st.button("🗑 Clear"):
            st.session_state.chat_history = []
            st.session_state.lc_messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 💬 MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("📚 AI Study Buddy Agent")

st.markdown(
    """
    > Ask questions, solve problems, or explore concepts.  
    > Powered by **RAG + Tools + Self-Evaluation**
    """
)

# ── SAMPLE PROMPTS ────────────────────────────────────────────────────────────

with st.expander("💡 Sample Queries"):
    samples = [
        "Explain Newton's second law",
        "Solve 2x^2 + 5x - 3 = 0",
        "What is today's date?",
        "Explain overfitting in ML",
        "Search latest AI research trends"
    ]

    for s in samples:
        if st.button(s):
            st.session_state["prefill"] = s
            st.rerun()


# ── CHAT HISTORY ──────────────────────────────────────────────────────────────

for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["query"])

    with st.chat_message("assistant"):
        st.write(entry["answer"])
        render_metadata(entry)

        if entry.get("sources"):
            with st.expander("📄 Sources"):
                for s in entry["sources"]:
                    st.markdown(f"- {s}")

        if entry.get("tool_output"):
            with st.expander("🔧 Tool Output"):
                st.code(entry["tool_output"])


# ── INPUT ─────────────────────────────────────────────────────────────────────

prefill = st.session_state.pop("prefill", "")
query = st.chat_input("Ask anything...") or prefill


if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        status = st.empty()
        status.status("🔀 Processing...")

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }

        initial_state = {
        "messages": st.session_state.lc_messages,
        "question": query,   # ✅ FIXED
        "retrieved": "",
        "sources": [],
        "answer": "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "route": "",
        "tool_result": "",
        "search_results": ""
}

        with st.spinner("Thinking..."):
            result = app.invoke(initial_state, config=config)

        status.empty()

        answer = result.get("answer", "No response generated.")
        sources = result.get("sources", [])
        score = result.get("eval_score", 0.0)
        retries = result.get("retry_count", 0)
        route = result.get("route", "")
        tool_output = result.get("tool_output", "")

        # ⚠️ Confidence feedback
        if score < 0.7:
            st.warning(f"⚠️ Low confidence ({score:.2f}). Verify results.")
        elif score < 0.85:
            st.info(f"ℹ️ Moderate confidence ({score:.2f}).")

        st.write(answer)

        render_metadata({
            "eval_score": score,
            "route": route,
            "retry_count": retries,
            "sources": sources,
            "tool_output": tool_output
        })

        if tool_output:
            with st.expander("🔧 Tool Output"):
                st.code(tool_output)

        if sources:
            with st.expander("📄 Sources"):
                for s in sources:
                    st.markdown(f"- **{s}**")

        # ── UPDATE MEMORY ──────────────────────────────────────────────────────
        st.session_state.lc_messages = result.get(
            "messages",
            st.session_state.lc_messages
        )

        st.session_state.chat_history.append({
            "query": query,
            "answer": answer,
            "sources": sources,
            "eval_score": score,
            "retry_count": retries,
            "route": route,
            "tool_output": tool_output
        })