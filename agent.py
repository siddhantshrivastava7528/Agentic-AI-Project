"""
agent.py — Physics Study Buddy Backend
"""

import chromadb
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ─────────────────────────────────────────────────────────────────────────────
# 📚 KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Newton's Laws",
        "text": "Newton's laws describe motion. F = ma is the second law."
    },
    {
        "id": "doc_002",
        "topic": "Kinematics",
        "text": "Equations: v=u+at, s=ut+1/2at², v²=u²+2as."
    },
    {
        "id": "doc_003",
        "topic": "Energy",
        "text": "Kinetic energy = 1/2mv², Potential energy = mgh."
    },
    {
        "id": "doc_004",
        "topic": "Electricity",
        "text": "Coulomb law: F = kq1q2/r²."
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# 🧠 STATE
# ─────────────────────────────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str


# ─────────────────────────────────────────────────────────────────────────────
# 🔧 COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def get_collection(embedder):
    client = chromadb.EphemeralClient()

    try:
        client.delete_collection("kb")
    except:
        pass

    col = client.create_collection("kb")

    texts = [d["text"] for d in DOCUMENTS]

    col.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
    )

    return col


# ─────────────────────────────────────────────────────────────────────────────
# 🚀 MAIN AGENT GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def get_app():

    embedder = get_embedder()
    llm = get_llm()
    collection = get_collection(embedder)

    # ── MEMORY ───────────────────────────────────────────────────────────────
    def memory_node(state):
        msgs = state.get("messages", []) + [
            {"role": "user", "content": state["question"]}
        ]
        return {"messages": msgs[-6:]}

    # ── ROUTER ───────────────────────────────────────────────────────────────
    def router_node(state):
        prompt = f"""Classify query into: retrieve / tool / skip

Question: {state['question']}"""

        res = llm.invoke(prompt).content.lower()

        if "tool" in res:
            return {"route": "tool"}
        elif "skip" in res or "memory" in res:
            return {"route": "skip"}
        else:
            return {"route": "retrieve"}

    # ── RETRIEVAL ────────────────────────────────────────────────────────────
    def retrieval_node(state):
        q_emb = embedder.encode([state["question"]]).tolist()

        res = collection.query(query_embeddings=q_emb, n_results=2)

        docs = res["documents"][0]
        topics = [m["topic"] for m in res["metadatas"][0]]

        context = "\n\n".join(
            f"[{topics[i]}] {docs[i]}" for i in range(len(docs))
        )

        return {"retrieved": context, "sources": topics}

    def skip_node(state):
        return {"retrieved": "", "sources": []}

    # ── TOOL ─────────────────────────────────────────────────────────────────
    def tool_node(state):
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(state["question"], max_results=2))

            text = "\n\n".join(r.get("body", "") for r in results)

            return {"tool_result": text}

        except Exception as e:
            return {"tool_result": f"Search error: {e}"}

    # ── ANSWER ───────────────────────────────────────────────────────────────
    def answer_node(state):

        context = state.get("retrieved", "") + "\n" + state.get("tool_result", "")

        msgs = [SystemMessage(content="You are a helpful physics tutor.")]

        for m in state.get("messages", [])[:-1]:
            if m["role"] == "user":
                msgs.append(HumanMessage(content=m["content"]))
            else:
                msgs.append(AIMessage(content=m["content"]))

        msgs.append(
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion: {state['question']}"
            )
        )

        response = llm.invoke(msgs)

        return {"answer": response.content}

    # ── SAVE ────────────────────────────────────────────────────────────────
    def save_node(state):
        msgs = state.get("messages", []) + [
            {"role": "assistant", "content": state["answer"]}
        ]
        return {"messages": msgs}

    # ── DECISIONS ────────────────────────────────────────────────────────────
    def route_decision(state):
        return state["route"]

    # ── GRAPH ───────────────────────────────────────────────────────────────
    graph = StateGraph(CapstoneState)

    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory", "router")

    graph.add_conditional_edges("router", route_decision, {
        "retrieve": "retrieve",
        "skip": "skip",
        "tool": "tool"
    })

    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("tool", "answer")

    graph.add_edge("answer", "save")
    graph.add_edge("save", END)

    return graph.compile(checkpointer=MemorySaver())