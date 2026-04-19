"""
agent.py — Shared backend for Physics Study Buddy
"""

import os
import chromadb
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ─────────────────────────────────────────────────────────────────────────────
# 📚 DOCUMENTS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
DOCUMENTS = [
{
"id": "doc_001",
"topic": "Newton's Laws of Motion",
"text": "Newton's three laws of motion form the foundation of classical mechanics. "
"The First Law (Law of Inertia) states that an object at rest stays at rest and an object in motion stays in motion unless acted upon by an external force. "
"The Second Law states that force equals mass times acceleration: F = ma. "
"The Third Law states that for every action there is an equal and opposite reaction."
},

{
"id": "doc_002",
"topic": "Kinematics and Projectile Motion",
"text": "Kinematics describes motion without forces. Key equations: v = u + at, s = ut + 1/2 at^2, v^2 = u^2 + 2as. "
"In projectile motion, horizontal motion is constant while vertical motion has acceleration due to gravity."
},

{
"id": "doc_003",
"topic": "Work, Energy, and Power",
"text": "Work is done when force causes displacement: W = Fd cosθ. "
"Kinetic energy is KE = 1/2 mv^2. Potential energy is PE = mgh. "
"Power is the rate of doing work: P = W/t."
},

{
"id": "doc_004",
"topic": "Thermodynamics and Heat Transfer",
"text": "Thermodynamics deals with heat and energy. First law: ΔU = Q − W. "
"Second law: entropy increases. Heat transfer occurs via conduction, convection, and radiation."
},

{
"id": "doc_005",
"topic": "Electric Fields and Coulomb's Law",
"text": "Coulomb's Law: F = kq1q2/r^2. Electric field: E = F/q. "
"Electric potential: V = kQ/r. Charges create electric fields that exert forces."
},

{
"id": "doc_006",
"topic": "Magnetism and Electromagnetic Induction",
"text": "Moving charges create magnetic fields. Magnetic force: F = qvB sinθ. "
"Faraday's law states that changing magnetic flux induces EMF."
},

{
"id": "doc_007",
"topic": "Waves and Sound",
"text": "Wave speed is v = fλ. Sound travels as longitudinal waves. "
"Doppler effect describes change in frequency due to motion."
},

{
"id": "doc_008",
"topic": "Optics",
"text": "Reflection follows angle of incidence equals angle of reflection. "
"Refraction follows Snell's Law: n1 sinθ1 = n2 sinθ2. "
"Lenses follow: 1/f = 1/do + 1/di."
},

{
"id": "doc_009",
"topic": "Modern Physics",
"text": "Einstein's relativity: E = mc^2. Quantum mechanics: E = hf. "
"Heisenberg uncertainty principle: ΔxΔp ≥ ħ/2."
},

{
"id": "doc_010",
"topic": "Circular Motion and Gravitation",
"text": "Centripetal force: F = mv^2/r. Gravitational force: F = GMm/r^2. "
"Escape velocity: v = √(2GM/R)."
},

{
"id": "doc_011",
"topic": "Simple Harmonic Motion",
"text": "SHM follows F = -kx. Period of spring: T = 2π√(m/k). "
"Energy oscillates between kinetic and potential."
},

{
"id": "doc_012",
"topic": "Fluid Mechanics",
"text": "Pressure: P = F/A. Buoyant force equals weight of displaced fluid. "
"Bernoulli's principle: P + 1/2ρv^2 + ρgh = constant."
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
    faithfulness: float
    eval_retries: int
    search_results: str


FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2


# ─────────────────────────────────────────────────────────────────────────────
# 🔧 CORE COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def get_collection(embedder=None):
    if embedder is None:
        embedder = get_embedder()

    client = chromadb.EphemeralClient()

    try:
        client.delete_collection("capstone_kb")
    except Exception:
        pass

    col = client.create_collection("capstone_kb")

    texts = [d["text"] for d in DOCUMENTS]

    col.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
    )

    return col


# ─────────────────────────────────────────────────────────────────────────────
# 🚀 MAIN APP GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def get_app():
    embedder = get_embedder()
    collection = get_collection(embedder)
    llm = get_llm()

    # ── MEMORY NODE ──────────────────────────────────────────────────────────
    def memory_node(state):
        msgs = state.get("messages", []) + [{"role": "user", "content": state["question"]}]
        return {"messages": msgs[-6:]}

    # ── ROUTER NODE (FIXED) ──────────────────────────────────────────────────
    def router_node(state):
        question = state["question"]
        msgs = state.get("messages", [])

        recent = "; ".join(
            f"{m['role']}: {m['content'][:60]}" for m in msgs[-3:-1]
        ) or "none"

        router_prompt = f"""You are a routing agent for a Physics Study Buddy.

Classify the query into ONE of:
- retrieve → physics concepts, theory
- tool → calculations, numeric problems
- memory_only → casual chat

Recent conversation: {recent}

Question: {question}

Reply with ONLY one word: retrieve / tool / memory_only.
"""

        response = llm.invoke(router_prompt).content.strip().lower()

        if "memory" in response:
            return {"route": "memory_only"}
        elif "tool" in response:
            return {"route": "tool"}
        else:
            return {"route": "retrieve"}

    # ── RETRIEVAL NODE (FIXED STRINGS) ───────────────────────────────────────
    def retrieval_node(state):
        q_emb = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)

        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]

        context = "\n\n---\n\n".join(
            f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
        )

        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state):
        return {"retrieved": "", "sources": []}

    # ── TOOL NODE ────────────────────────────────────────────────────────────
    def tool_node(state):
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(f"physics {state['question']}", max_results=4))

            snippets = "\n\n".join(
                f"[{r.get('title', 'Source')}]\n{r.get('body', '')}" for r in results
            )

            return {"tool_result": snippets, "search_results": snippets}

        except Exception as e:
            return {"tool_result": f"Search error: {e}", "search_results": ""}

    # ── ANSWER NODE (FIXED STRINGS) ──────────────────────────────────────────
    def answer_node(state):
        question = state["question"]
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages = state.get("messages", [])
        retries = state.get("eval_retries", 0)

        parts = []
        if retrieved:
            parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result:
            parts.append(f"TOOL RESULT:\n{tool_result}")

        context = "\n\n".join(parts)

        retry_note = (
            "IMPORTANT: Improve accuracy.\n" if retries > 0 else ""
        )

        if context:
            system_text = f"""You are an expert Physics Study Buddy.
Answer ONLY using the context below.

{retry_note}

{context}
"""
        else:
            system_text = "You are a helpful assistant."

        lc_msgs = [SystemMessage(content=system_text)]

        for m in messages[:-1]:
            if m["role"] == "user":
                lc_msgs.append(HumanMessage(content=m["content"]))
            else:
                lc_msgs.append(AIMessage(content=m["content"]))

        lc_msgs.append(HumanMessage(content=question))

        response = llm.invoke(lc_msgs)

        return {"answer": response.content}

    # ── EVAL NODE (FIXED STRING) ─────────────────────────────────────────────
    def eval_node(state):
        answer = state.get("answer", "")
        context = state.get("retrieved", "")[:500]
        retries = state.get("eval_retries", 0)

        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        prompt = f"""Rate faithfulness from 0.0 to 1.0.

Context:
{context}

Answer:
{answer[:300]}

Return ONLY a number.
"""

        try:
            score = float(
                llm.invoke(prompt).content.strip().split()[0].replace(",", ".")
            )
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5

        return {"faithfulness": score, "eval_retries": retries + 1}

    # ── SAVE NODE ────────────────────────────────────────────────────────────
    def save_node(state):
        msgs = state.get("messages", []) + [
            {"role": "assistant", "content": state.get("answer", "")}
        ]
        return {"messages": msgs}

    # ── ROUTING LOGIC ────────────────────────────────────────────────────────
    def route_decision(state):
        r = state.get("route", "retrieve")
        if r == "tool":
            return "tool"
        if r == "memory_only":
            return "skip"
        return "retrieve"

    def eval_decision(state):
        if (
            state.get("faithfulness", 1.0) >= FAITHFULNESS_THRESHOLD
            or state.get("eval_retries", 0) >= MAX_EVAL_RETRIES
        ):
            return "save"
        return "answer"

    # ── GRAPH ────────────────────────────────────────────────────────────────
    graph = StateGraph(CapstoneState)

    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory", "router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
    )

    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("tool", "answer")

    graph.add_edge("answer", "eval")

    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {"answer": "answer", "save": "save"},
    )

    graph.add_edge("save", END)

    return graph.compile(checkpointer=MemorySaver())