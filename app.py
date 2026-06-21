"""
app.py  —  Phase 10: Streamlit Chatbot UI
==========================================
Full-featured multimodal RAG chatbot interface.

Features
--------
• PDF upload → auto-triggers pipeline phases 1-7
• Chat interface with conversation history
• Retrieved chunks panel (page number, type, score, text preview)
• Page image viewer (shows rendered page for each retrieved chunk)
• Answer generation timer
• Provider / model badge
• One-click "Run required test questions" for assignment demo

Run
---
    streamlit run app.py
"""

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from PIL import Image

import config as cfg

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Niwas FY2025 — Multimodal RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# REQUIRED TEST QUESTIONS (assignment)
# ─────────────────────────────────────────────

REQUIRED_QUESTIONS = [
    "What are the key risks to Niwas's business highlighted in the MD&A section?",
    "What is Niwas's stated strategy for scaling its loan book as described in the Board's Report?",
    "What is the total loan book (Loans line) as at March 31, 2025 vs March 31, 2024?",
    "What was Niwas's total interest income for FY2025 and how did it compare to FY2024?",
    "Describe the trend or distribution shown in the geographic / state-wise loan portfolio figure.",
    "What does the AUM or disbursement chart in the Corporate Overview section show about growth over the years?",
]


# ─────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────

def init_session():
    defaults = {
        "chat_history":     [],   # list of {"role", "content", "metadata"}
        "engine":           None,
        "pipeline_done":    False,
        "pdf_uploaded":     False,
        "processing":       False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Chat bubbles */
    .user-bubble   { background:#1a1a2e; color:#e0e0e0; border-radius:12px;
                     padding:12px 16px; margin:6px 0; }
    .assist-bubble { background:#16213e; color:#e0e0e0; border-radius:12px;
                     padding:12px 16px; margin:6px 0; border-left:3px solid #0f3460; }
    /* Chunk cards */
    .chunk-card    { background:#0d1117; border:1px solid #21262d;
                     border-radius:8px; padding:10px 14px; margin:6px 0;
                     font-size:0.82rem; color:#c9d1d9; }
    .chunk-header  { font-weight:600; color:#58a6ff; margin-bottom:4px; }
    .chunk-meta    { color:#8b949e; font-size:0.75rem; }
    /* Badge */
    .badge { display:inline-block; background:#0f3460; color:#58a6ff;
             border-radius:20px; padding:2px 10px; font-size:0.72rem; }
    /* Timer */
    .timer { color:#3fb950; font-size:0.78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Multimodal RAG")
    st.caption("Niwas Housing Finance FY2025")
    st.divider()

    # ── PDF Upload ─────────────────────────────────────────────────
    st.subheader("1. Upload PDF")
    uploaded_file = st.file_uploader(
        "Annual Report PDF",
        type=["pdf"],
        help="Upload NHFPL FY2025 Annual Report",
    )

    if uploaded_file is not None and not st.session_state.pipeline_done:
        # Save PDF to data/raw/
        dest = cfg.RAW_DIR / uploaded_file.name
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved: {uploaded_file.name}")
        st.session_state.pdf_uploaded = True

    # ── Pipeline Controls ──────────────────────────────────────────
    st.subheader("2. Run Pipeline")

    col_phases, col_skip = st.columns(2)
    with col_phases:
        phases = st.multiselect(
            "Phases",
            options=[1, 2, 3, 4, 5, 6],
            default=[1, 2, 3, 5, 6],
            help="Phase 4 requires a vision API key",
        )
    with col_skip:
        st.caption("Deselect Phase 4\nif no vision API key")

    if st.button(
        "▶ Run Pipeline",
        disabled=not st.session_state.pdf_uploaded,
        use_container_width=True,
    ):
        _run_pipeline(phases)

    # ── Status ─────────────────────────────────────────────────────
    if cfg.CHUNKS_JSON.exists():
        import json as _json
        n = len(_json.loads(cfg.CHUNKS_JSON.read_text()))
        st.success(f"✓ Index ready ({n} chunks)")
        if st.session_state.engine is None:
            _load_engine()
    else:
        st.info("Run the pipeline to index the document.")

    st.divider()

    # ── Settings ───────────────────────────────────────────────────
    st.subheader("3. Settings")
    top_k = st.slider("Top-K chunks", 1, 10, cfg.TOP_K)
    st.caption(
        f"LLM: `{cfg.LLM_PROVIDER}/{cfg.GEMINI_LLM_MODEL if cfg.LLM_PROVIDER=='gemini' else cfg.GROQ_LLM_MODEL}`"
    )

    st.divider()

    # ── Required test questions ────────────────────────────────────
    st.subheader("4. Test Questions")
    if st.button("🧪 Run All 6 Required Questions", use_container_width=True):
        _run_batch_questions(REQUIRED_QUESTIONS, top_k)

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ─────────────────────────────────────────────
# PIPELINE + ENGINE LOADERS (defined before use)
# ─────────────────────────────────────────────

def _run_pipeline(phases: list[int]):
    """Execute selected pipeline phases with a progress bar."""
    st.session_state.processing = True
    progress = st.progress(0, text="Starting pipeline …")

    import importlib
    phase_names = {
        1: "PDF Parsing",
        2: "Image Extraction",
        3: "Vision Page Selection",
        4: "Vision Understanding",
        5: "Chunking",
        6: "Embedding + Indexing",
    }

    for i, phase_num in enumerate(sorted(phases)):
        label = phase_names.get(phase_num, f"Phase {phase_num}")
        progress.progress(
            int((i / len(phases)) * 100),
            text=f"Phase {phase_num}: {label} …",
        )
        try:
            mod_map = {
                1: "src.parser",
                2: "src.image_extractor",
                3: "src.vision_selector",
                4: "src.vision_analyzer",
                5: "src.chunker",
                6: "src.embedder",
            }
            module = importlib.import_module(mod_map[phase_num])
            module.run()
        except Exception as exc:
            st.error(f"Phase {phase_num} failed: {exc}")
            st.session_state.processing = False
            return

    progress.progress(100, text="Pipeline complete!")
    st.session_state.pipeline_done = True
    st.session_state.processing = False
    _load_engine()
    st.rerun()


def _load_engine():
    """Instantiate the RAGEngine and cache it in session state."""
    try:
        from src.rag_engine import RAGEngine
        st.session_state.engine = RAGEngine()
    except Exception as exc:
        st.warning(f"Could not load RAG engine: {exc}")


def _run_batch_questions(questions: list[str], top_k: int):
    """Ask all 6 required questions in sequence."""
    if st.session_state.engine is None:
        st.error("Pipeline not ready. Run the pipeline first.")
        return
    for q in questions:
        _ask_and_record(q, top_k)
    st.rerun()


def _ask_and_record(question: str, top_k: int):
    """Ask one question and append to chat history."""
    engine = st.session_state.engine
    if engine is None:
        st.error("Engine not loaded.")
        return

    # Record user message
    st.session_state.chat_history.append(
        {"role": "user", "content": question, "metadata": {}}
    )

    # Generate answer
    try:
        answer_obj = engine.ask(question, top_k=top_k)
    except Exception as exc:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"Error: {exc}",
                "metadata": {},
            }
        )
        return

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer_obj.answer,
            "metadata": answer_obj.to_dict(),
        }
    )


# ─────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────

st.title("Multimodal RAG — Niwas Housing Finance FY2025")
st.caption(
    "Ask questions about the annual report. "
    "Retrieved chunks, source pages, and related images are shown below each answer."
)

# ── Render chat history ─────────────────────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            meta = msg.get("metadata", {})

            # Answer text
            st.markdown(msg["content"])

            # Timer + model badge
            if meta.get("elapsed_seconds"):
                st.markdown(
                    f'<span class="timer">⏱ {meta["elapsed_seconds"]:.2f}s</span> '
                    f'&nbsp; <span class="badge">'
                    f'{meta.get("provider","")}/{meta.get("model","")}</span>',
                    unsafe_allow_html=True,
                )

            # Retrieved chunks + images
            chunks = meta.get("retrieved_chunks", [])
            if chunks:
                with st.expander(
                    f"📎 {len(chunks)} Retrieved Chunks (click to expand)", expanded=False
                ):
                    for c in chunks:
                        # Chunk card
                        st.markdown(
                            f'<div class="chunk-card">'
                            f'<div class="chunk-header">#{c["rank"]} — '
                            f'{c["content_type"].upper()} — {c["citation"]}</div>'
                            f'<div class="chunk-meta">Score: {c["score"]} | '
                            f'Section: {c["section"]}</div>'
                            f'<div style="margin-top:6px">{c["text"][:400]}…</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        # Page render thumbnail
                        render_rel = c.get("render_path", "")
                        if render_rel:
                            render_path = cfg.ROOT_DIR / render_rel
                            if render_path.exists():
                                img = Image.open(render_path)
                                st.image(
                                    img,
                                    caption=f"Page {c['page_num']} render",
                                    width=420,
                                )

# ── Chat input ──────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask a question about the annual report …",
    disabled=(st.session_state.engine is None),
):
    top_k = st.session_state.get("top_k", cfg.TOP_K) if False else cfg.TOP_K
    # Read top_k from sidebar widget (it's in local scope at render time)
    # We use the default from config; the slider value is captured below
    _ask_and_record(prompt, cfg.TOP_K)
    st.rerun()


# ── Empty state ─────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.info(
        "👈 Upload the PDF and run the pipeline in the sidebar to get started.\n\n"
        "Then use the **Run All 6 Required Questions** button to demo the assignment."
    )