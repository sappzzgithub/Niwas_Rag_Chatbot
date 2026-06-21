"""
app.py  —  Multimodal RAG Chat UI
===================================
Clean question-answer interface for the Niwas Housing Finance FY2025 RAG pipeline.

Usage
-----
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

# ── Project root on path ─────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from src.rag_engine import RAGEngine, RAGAnswer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Niwas FY2025 — Ask Anything",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Hide default streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Page background */
    .stApp { background-color: #f5f7fa; }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .app-header h1 { font-size: 1.6rem; margin: 0; color: white; }
    .app-header p  { font-size: 0.88rem; color: #a8dadc; margin: 0.3rem 0 0 0; }

    /* Answer card */
    .answer-card {
        background: white;
        border-radius: 10px;
        padding: 1.4rem 1.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
        border-left: 5px solid #457b9d;
        font-size: 0.95rem;
        line-height: 1.75;
        color: #1a1a2e;
    }

    /* Chunk card */
    .chunk-card {
        background: #fafafa;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid #457b9d;
        font-size: 0.84rem;
        color: #333;
    }
    .chunk-card.table { border-left-color: #2a9d8f; }
    .chunk-card.image { border-left-color: #e76f51; }

    .chunk-meta {
        font-size: 0.75rem;
        color: #888;
        margin-bottom: 0.3rem;
        display: flex;
        gap: 0.5rem;
        align-items: center;
        flex-wrap: wrap;
    }

    /* Badges */
    .badge {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-text  { background:#264653; color:#fff; }
    .badge-table { background:#2a9d8f; color:#fff; }
    .badge-image { background:#e76f51; color:#fff; }

    /* Citation */
    .citation {
        display: inline-block;
        background: #e8f4f8;
        color: #457b9d;
        border: 1px solid #b8d9ea;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin: 2px 3px 0 0;
    }

    /* History question label */
    .q-label {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }

    /* Timing */
    .timing {
        font-size: 0.75rem;
        color: #aaa;
        margin-top: 0.5rem;
    }

    /* Input area */
    .stTextArea textarea {
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        border: 1.5px solid #d0d7de !important;
    }
    .stTextArea textarea:focus {
        border-color: #457b9d !important;
        box-shadow: 0 0 0 3px #457b9d22 !important;
    }

    /* Ask button */
    .stButton > button {
        background: #457b9d;
        color: white;
        border-radius: 8px;
        border: none;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0.5rem 2rem;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #1d3557; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1a1a2e;
        color: white;
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stButton > button {
        background: #457b9d;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "engine" not in st.session_state:
    st.session_state.engine = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of {question, answer: RAGAnswer}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏠 Niwas RAG")
    st.markdown("Ask anything about the **FY2025 Annual Report**")
    st.divider()

    # Engine loader
    if st.session_state.engine is None:
        st.warning("Engine not loaded")
        if st.button("🚀 Load Engine", type="primary"):
            with st.spinner("Loading …"):
                try:
                    st.session_state.engine = RAGEngine()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    else:
        st.success("✅ Engine ready")
        if st.button("🔄 Reload"):
            st.session_state.engine = None
            st.rerun()

    st.divider()
    st.markdown("### Config")
    st.markdown(f"**Provider:** `{cfg.LLM_PROVIDER}`")
    st.markdown(f"**LLM:** `{cfg.GROQ_LLM_MODEL if cfg.LLM_PROVIDER == 'groq' else cfg.OLLAMA_LLM_MODEL}`")
    st.markdown(f"**Embedding:** `BAAI/bge-base-en-v1.5`")
    st.markdown(f"**Top-K:** `{cfg.TOP_K} chunks`")

    st.divider()

    # Chunk stats
    try:
        import json as _json
        _chunks = _json.loads(cfg.CHUNKS_JSON.read_text())
        st.markdown("### Index Stats")
        st.markdown(f"📄 Text: **{sum(1 for c in _chunks if c['content_type']=='text')}**")
        st.markdown(f"📊 Table: **{sum(1 for c in _chunks if c['content_type']=='table')}**")
        st.markdown(f"🖼 Image: **{sum(1 for c in _chunks if c['content_type']=='image')}**")
        st.markdown(f"Total: **{len(_chunks)} chunks**")
    except Exception:
        st.markdown("*Run pipeline first*")

    st.divider()
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <h1>🏠 Niwas Housing Finance FY2025 — Ask Anything</h1>
  <p>Answers are grounded in the 133-page annual report · text · tables · charts · citations included</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────

if st.session_state.engine is None:
    st.info("⬅️ Click **Load Engine** in the sidebar to get started.")

question = st.text_area(
    "Your question",
    height=100,
    placeholder=(
        "e.g. What is Niwas's total AUM as of March 2025?\n"
        "e.g. What are the key risks mentioned in the MD&A?\n"
        "e.g. What was the interest income for FY2025?"
    ),
    label_visibility="collapsed",
    disabled=st.session_state.engine is None,
)

ask_col, _ = st.columns([2, 5])
with ask_col:
    ask = st.button(
        "🔍 Ask",
        type="primary",
        disabled=st.session_state.engine is None or not question.strip(),
    )

# ─────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────

if ask and question.strip():
    with st.spinner("Searching document and generating answer …"):
        try:
            answer: RAGAnswer = st.session_state.engine.ask(question.strip())
            st.session_state.history.insert(0, {
                "question": question.strip(),
                "answer": answer,
            })
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────────
# RESULTS HISTORY
# ─────────────────────────────────────────────

for item in st.session_state.history:
    q: str       = item["question"]
    a: RAGAnswer = item["answer"]

    # Question label
    st.markdown(f'<div class="q-label">💬 {q}</div>', unsafe_allow_html=True)

    # Two columns: answer left, chunks right
    left, right = st.columns([5, 4])

    # ── Answer ────────────────────────────────────────────────────────
    with left:
        st.markdown("**Answer**")
        st.markdown(f'<div class="answer-card">{a.answer}</div>', unsafe_allow_html=True)

        # Citations
        if a.citations:
            cite_html = "".join(f'<span class="citation">{c}</span>' for c in a.citations)
            st.markdown(f"**Sources:** {cite_html}", unsafe_allow_html=True)

        st.markdown(
            f'<div class="timing">⏱ {a.elapsed_seconds:.2f}s &nbsp;·&nbsp; {a.provider}/{a.model}</div>',
            unsafe_allow_html=True,
        )

    # ── Retrieved Chunks ──────────────────────────────────────────────
    with right:
        st.markdown(f"**Retrieved Chunks** (Top-{cfg.TOP_K})")
        for c in a.retrieved_chunks:
            ct      = c["content_type"]
            icon    = {"text": "📄", "table": "📊", "image": "🖼"}.get(ct, "•")
            badge   = f'<span class="badge badge-{ct}">{icon} {ct}</span>'
            preview = c["text"][:200].replace("\n", " ")

            st.markdown(f"""
<div class="chunk-card {ct}">
  <div class="chunk-meta">
    {badge}
    <span>Rank {c['rank']}</span>
    <span>·</span>
    <span>Page {c['page_num']}</span>
    <span>·</span>
    <span>{c['section']}</span>
    <span>·</span>
    <span>score {c['score']:.4f}</span>
  </div>
  <div>{preview}…</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

# ─────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────

if not st.session_state.history and st.session_state.engine is not None:
    st.markdown("""
    ### Try asking:
    - *What are the key risks to Niwas's business?*
    - *What is the total loan book as at March 31, 2025?*
    - *What was the interest income for FY2025 compared to FY2024?*
    - *What does the AUM chart show about growth?*
    - *What is Niwas's capital adequacy ratio?*
    - *Who are the key management personnel?*
    - *What is the GNPA ratio for FY2025?*
    """)