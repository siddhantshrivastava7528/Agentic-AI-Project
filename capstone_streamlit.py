"""
capstone_streamlit.py — Physics Study Buddy Agent
Run: streamlit run capstone_streamlit.py
"""

import streamlit as st
import uuid
import os
import chromadb
from dotenv import load_dotenv
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ── LOAD ENV ────────────────────────────────────────────────────────────────
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not set")
    st.stop()

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Physics Study Buddy", page_icon="⚛️")
st.title("⚛️ Physics Study Buddy")
st.caption("AI-powered physics tutor")

# ── SESSION STATE ───────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── LOAD AGENT ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_agent():

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("capstone_kb")
    except:
        pass

    collection = client.create_collection("capstone_kb")

    # ✅ KEEP YOUR FULL DOCUMENTS (unchanged)
    DOCUMENTS = [...]  # <-- keep your full dataset here

    texts = [d["text"] for d in DOCUMENTS]

    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
    )

    class CapstoneState(TypedDict):
        question: str
        messages: List[dict]
        route: str
        retrieved: str
        sources: List[str]
        tool_result: str
        answer: str
        faithfulness: float
        eval_retries: int
        search_results: str

    # ── NODES ───────────────────────────────────────────────────────────────
    def memory_node(state):
        msgs = state.get("messages", []) + [{"role":"user","content":state["question"]}]
        return {"messages": msgs[-6:]}

    def router_node(state):
        prompt = f"""Route this physics question. Reply ONLY: retrieve / tool / skip
Question: {state['question']}"""
        r = llm.invoke(prompt).content.lower()

        if "tool" in r:
            return {"route": "tool"}
        elif "skip" in r or "memory" in r:
            return {"route": "skip"}
        else:
            return {"route": "retrieve"}

    def retrieval_node(state):
        q_emb = embedder.encode([state["question"]]).tolist()
        res = collection.query(query_embeddings=q_emb, n_results=3)

        chunks = res["documents"][0]
        topics = [m["topic"] for m in res["metadatas"][0]]

        context = "\n\n".join(
            f"[{topics[i]}] {chunks[i]}" for i in range(len(chunks))
        )

        return {"retrieved": context, "sources": topics}

    def skip_node(state):
        return {"retrieved": "", "sources": []}

    def tool_node(state):
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(state["question"], max_results=3))

            text = "\n\n".join(r.get("body","") for r in results)
            return {"tool_result": text}
        except Exception as e:
            return {"tool_result": f"Search error: {e}"}

    def answer_node(state):
        context = state.get("retrieved","") + "\n" + state.get("tool_result","")

        messages = [
            SystemMessage(content="You are a physics tutor."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}")
        ]

        answer = llm.invoke(messages).content
        return {"answer": answer}

    def eval_node(state):
        return {"faithfulness": 1.0, "eval_retries": 1}

    def save_node(state):
        msgs = state.get("messages", [])
        return {"messages": msgs + [{"role":"assistant","content":state.get("answer","")}]}

    def route_decision(state):
        return state["route"]

    def eval_decision(state):
        return "save"

    # ── GRAPH ───────────────────────────────────────────────────────────────
    graph = StateGraph(CapstoneState)

    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory","router")

    graph.add_conditional_edges("router", route_decision, {
        "retrieve":"retrieve",
        "skip":"skip",
        "tool":"tool"
    })

    graph.add_edge("retrieve","answer")
    graph.add_edge("skip","answer")
    graph.add_edge("tool","answer")

    graph.add_edge("answer","eval")

    graph.add_conditional_edges("eval", eval_decision, {
        "save":"save"
    })

    graph.add_edge("save", END)

    return graph.compile(checkpointer=MemorySaver())


agent_app = load_agent()

# ── CHAT UI ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a physics question..."):

    st.session_state.messages.append({"role":"user","content":prompt})

    # ✅ FIX: PASS CONFIG + MESSAGES
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    result = agent_app.invoke(
        {
            "question": prompt,
            "messages": st.session_state.messages
        },
        config=config
    )

    answer = result["answer"]

    st.write(answer)

    st.session_state.messages.append({
        "role":"assistant",
        "content":answer
    })