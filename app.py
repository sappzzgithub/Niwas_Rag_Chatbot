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
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
    header     {visibility: hidden;}

    /* Full dark background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

    /* Sidebar */
    section[data_testid="stSidebar"],
    section[data-testid="stSidebar"] { background: #1a1a2e !important; }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div { color: #c9d1d9 !important; }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
    section[data-testid="stSidebar"] .stButton > button {
        background: #238636 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        width: 100% !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #2ea043 !important;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.8rem;
        border: 1px solid #30363d;
    }
    .app-header h1 { font-size: 1.5rem; margin: 0 0 0.3rem 0; color: #ffffff; }
    .app-header p  { font-size: 0.85rem; color: #8b949e; margin: 0; }

    /* Input label */
    .input-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #c9d1d9;
        margin-bottom: 0.4rem;
    }

    /* Text input overrides */
    .stTextInput > div > div > input {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border: 1.5px solid #30363d !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1rem !important;
        caret-color: #58a6ff !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 3px #58a6ff22 !important;
    }
    .stTextInput > div > div > input::placeholder { color: #484f58 !important; }

    /* Ask button */
    div[data-testid="column"]:last-child .stButton > button {
        background: #238636 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        height: 2.6rem !important;
        width: 100% !important;
        margin-top: 0.1rem !important;
    }
    div[data-testid="column"]:last-child .stButton > button:hover {
        background: #2ea043 !important;
    }

    /* Answer card */
    .answer-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        font-size: 0.92rem;
        line-height: 1.75;
        color: #e6edf3;
        margin-bottom: 0.75rem;
    }

    /* Chunk card */
    .chunk-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 3px solid #58a6ff;
        border-radius: 6px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
        color: #c9d1d9;
    }
    .chunk-card.table { border-left-color: #2a9d8f; }
    .chunk-card.image { border-left-color: #e76f51; }

    .chunk-meta {
        font-size: 0.73rem;
        color: #6e7681;
        margin-bottom: 0.35rem;
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        align-items: center;
    }

    /* Badges */
    .badge { padding: 1px 7px; border-radius: 3px; font-size: 0.7rem; font-weight: 700; }
    .badge-text  { background:#264653; color:#a8dadc; }
    .badge-table { background:#0d3d38; color:#2dd4bf; }
    .badge-image { background:#3d1a0d; color:#f97316; }

    /* Citations */
    .citation {
        display: inline-block;
        background: #1c2b3a;
        color: #58a6ff;
        border: 1px solid #1f4f8c;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.73rem;
        margin: 2px 3px 0 0;
    }

    .q-label {
        font-size: 1rem;
        font-weight: 600;
        color: #e6edf3;
        margin: 1rem 0 0.75rem 0;
        padding-left: 0.5rem;
        border-left: 3px solid #58a6ff;
    }

    .timing {
        font-size: 0.72rem;
        color: #484f58;
        margin-top: 0.5rem;
    }

    .section-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6e7681;
        margin-bottom: 0.5rem;
    }

    /* Divider */
    hr { border-color: #21262d !important; margin: 1.2rem 0 !important; }

    /* Suggested chips */
    .chip {
        display: inline-block;
        background: #21262d;
        color: #8b949e;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        margin: 3px 4px 3px 0;
        cursor: default;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "engine" not in st.session_state:
    st.session_state.engine = None
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏠 Niwas RAG")
    st.markdown("Annual Report FY2025 · 133 pages")
    st.divider()

    if st.session_state.engine is None:
        st.markdown("**Engine not loaded**")
        if st.button("Load Engine"):
            with st.spinner("Loading …"):
                try:
                    st.session_state.engine = RAGEngine()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    else:
        st.success("✅ Engine ready")
        if st.button("Reload Engine"):
            st.session_state.engine = None
            st.rerun()

    st.divider()
    st.markdown("### Config")
    st.markdown(f"**Provider:** `{cfg.LLM_PROVIDER}`")
    st.markdown(f"**LLM:** `{cfg.GROQ_LLM_MODEL if cfg.LLM_PROVIDER == 'groq' else cfg.OLLAMA_LLM_MODEL}`")
    st.markdown(f"**Embedding:** `bge-base-en-v1.5`")
    st.markdown(f"**Top-K:** `{cfg.TOP_K}`")

    st.divider()

    try:
        import json as _json
        _chunks = _json.loads(cfg.CHUNKS_JSON.read_text())
        st.markdown("### Index")
        st.markdown(f"📄 Text &nbsp; **{sum(1 for c in _chunks if c['content_type']=='text')}**")
        st.markdown(f"📊 Table **{sum(1 for c in _chunks if c['content_type']=='table')}**")
        st.markdown(f"🖼 Image **{sum(1 for c in _chunks if c['content_type']=='image')}**")
        st.markdown(f"Total &nbsp; **{len(_chunks)} chunks**")
    except Exception:
        st.markdown("*Run pipeline first*")

    st.divider()
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <h1>🏠 Niwas Housing Finance FY2025 — Document Q&amp;A</h1>
  <p>Ask any question about the annual report · answers are grounded with source citations</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT ROW
# ─────────────────────────────────────────────

st.markdown('<div class="input-label">Ask a question about the annual report</div>', unsafe_allow_html=True)

col_input, col_btn = st.columns([8, 1])

with col_input:
    question = st.text_input(
        label="question",
        placeholder="e.g. What was the total interest income for FY2025?",
        label_visibility="collapsed",
        disabled=st.session_state.engine is None,
        key="question_input",
    )

with col_btn:
    ask = st.button(
        "Ask →",
        type="primary",
        disabled=st.session_state.engine is None or not (question or "").strip(),
    )

if st.session_state.engine is None:
    st.caption("⬅ Load the engine from the sidebar to begin")

# ─────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────

if ask and question.strip():
    with st.spinner("Searching and generating answer …"):
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
# EMPTY STATE
# ─────────────────────────────────────────────

if not st.session_state.history and st.session_state.engine is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Try asking:**")
    st.markdown("""
<span class="chip">What are the key risks in the MD&A?</span>
<span class="chip">Total loan book March 2025 vs 2024?</span>
<span class="chip">Interest income FY2025?</span>
<span class="chip">AUM growth over the years?</span>
<span class="chip">What is the GNPA ratio?</span>
<span class="chip">Capital adequacy ratio FY2025?</span>
<span class="chip">Who are the key management personnel?</span>
<span class="chip">What is Niwas's strategy for expansion?</span>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

for item in st.session_state.history:
    q: str       = item["question"]
    a: RAGAnswer = item["answer"]

    st.markdown(f'<div class="q-label">💬 {q}</div>', unsafe_allow_html=True)

    left, right = st.columns([5, 4])

    with left:
        st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-card">{a.answer}</div>', unsafe_allow_html=True)

        if a.citations:
            cite_html = "".join(f'<span class="citation">{c}</span>' for c in a.citations)
            st.markdown(f"**Sources:** {cite_html}", unsafe_allow_html=True)

        st.markdown(
            f'<div class="timing">⏱ {a.elapsed_seconds:.2f}s · {a.provider}/{a.model}</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(f'<div class="section-label">Retrieved Chunks (Top-{cfg.TOP_K})</div>', unsafe_allow_html=True)
        for c in a.retrieved_chunks:
            ct      = c["content_type"]
            icon    = {"text": "📄", "table": "📊", "image": "🖼"}.get(ct, "•")
            badge   = f'<span class="badge badge-{ct}">{icon} {ct}</span>'
            preview = c["text"][:180].replace("\n", " ")

            st.markdown(f"""
<div class="chunk-card {ct}">
  <div class="chunk-meta">
    {badge}
    <span>Rank {c['rank']}</span>·
    <span>Page {c['page_num']}</span>·
    <span>{c['section']}</span>·
    <span>score {c['score']:.4f}</span>
  </div>
  {preview}…
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)