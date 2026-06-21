"""
config.py
=========
Single source of truth for all configuration across the pipeline.
Modify values here; never hardcode paths or model names in business logic.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# ROOT PATHS
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"

RAW_DIR       = DATA_DIR / "raw"
PARSED_DIR    = DATA_DIR / "parsed"
IMAGES_DIR    = DATA_DIR / "images"
CHUNKS_DIR    = DATA_DIR / "chunks"
VECTORDB_DIR  = DATA_DIR / "vectordb"
OUTPUTS_DIR   = ROOT_DIR / "outputs"

for _d in [RAW_DIR, PARSED_DIR, IMAGES_DIR, CHUNKS_DIR, VECTORDB_DIR, OUTPUTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# DOCUMENT
# ─────────────────────────────────────────────
PDF_FILENAME  = "NHFPL_Annual_Report _FY2025.pdf"
PDF_PATH      = RAW_DIR / PDF_FILENAME
PDF_NAME      = Path(PDF_FILENAME).stem

# ─────────────────────────────────────────────
# PHASE 1 – PARSING
# ─────────────────────────────────────────────
PARSED_JSON        = PARSED_DIR / "parsed.json"
PAGES_MD_DIR       = PARSED_DIR / "pages_md"
PAGES_MD_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# PHASE 2 – IMAGE EXTRACTION
# ─────────────────────────────────────────────
IMAGE_METADATA_JSON = IMAGES_DIR / "image_metadata.json"
EMBEDDED_DIR        = IMAGES_DIR / "embedded"
PAGE_RENDERS_DIR    = IMAGES_DIR / "page_renders"
EMBEDDED_DIR.mkdir(parents=True, exist_ok=True)
PAGE_RENDERS_DIR.mkdir(parents=True, exist_ok=True)
PAGE_RENDER_DPI     = 150
MIN_IMAGE_SIZE_PX   = 50

# ─────────────────────────────────────────────
# PHASE 3 – VISION PAGE SELECTION
# ─────────────────────────────────────────────
VISION_PAGES_JSON  = IMAGES_DIR / "vision_pages.json"

VISUAL_KEYWORDS = [
    "chart", "graph", "figure", "fig.", "trend", "growth",
    "disbursement", "aum", "portfolio", "distribution", "breakdown",
    "overview", "highlight", "infographic", "map", "state-wise",
    "geographic", "composition", "mix", "%", "₹", "crore",
    "balance sheet", "profit", "income", "cash flow",
]

# ─────────────────────────────────────────────
# PHASE 4 – VISION UNDERSTANDING
# ─────────────────────────────────────────────
VISION_DESCRIPTIONS_JSON = IMAGES_DIR / "vision_descriptions.json"

# Supported providers: "groq" | "ollama"
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "groq")          # ← changed

GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
GROQ_VISION_MODEL   = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")  # ← changed

OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:13b")

VISION_PROMPT = """
You are a financial document analyst. This image is a page from an annual report.
Carefully examine and describe:
1. All charts or graphs: their type (bar/line/pie etc.), axes labels, data series, key trends, exact numeric values visible.
2. All tables: headers, rows, any totals or subtotals.
3. Infographics or highlighted KPIs: exact numbers, units (₹ crore, %, etc.).
4. Any geographic maps or state-wise distribution diagrams.
5. Text callouts, growth arrows, or annotations.

Return a structured, detailed description. Preserve ALL numeric values exactly as shown.
If the page contains no meaningful chart, table, or infographic, reply: NO_VISUAL_CONTENT.
"""

# ─────────────────────────────────────────────
# PHASE 5 – CHUNKING
# ─────────────────────────────────────────────
CHUNKS_JSON          = CHUNKS_DIR / "chunks.json"
CHUNK_SIZE           = 450
CHUNK_OVERLAP        = 60
CHUNK_SIZE_CHARS     = CHUNK_SIZE * 4
CHUNK_OVERLAP_CHARS  = CHUNK_OVERLAP * 4

# ─────────────────────────────────────────────
# PHASE 6 – EMBEDDINGS
# ─────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# ─────────────────────────────────────────────
# PHASE 7 – VECTOR DATABASE
# ─────────────────────────────────────────────
CHROMA_COLLECTION = PDF_NAME.replace(" ", "_").replace("-", "_")[:63]

# ─────────────────────────────────────────────
# PHASE 8 – RETRIEVAL
# ─────────────────────────────────────────────
TOP_K = 5

# ─────────────────────────────────────────────
# PHASE 9 – RAG / LLM
# ─────────────────────────────────────────────
# Supported providers: "groq" | "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")                # ← changed

GROQ_LLM_MODEL   = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")   # ← changed

OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "mistral:7b")
LLM_INTER_QUESTION_DELAY = int(os.getenv("LLM_INTER_QUESTION_DELAY", "30"))

RAG_SYSTEM_PROMPT = """
You are a precise financial analyst assistant with access to excerpts from the
Niwas Housing Finance Private Limited Annual Report FY2025.

RULES:
- Answer ONLY from the provided context chunks.
- Include a "Source:" citation for every factual claim, e.g. "Source: Balance Sheet, page 61".
- If multiple sources support a claim, cite all of them.
- If the answer cannot be found in the context, say: "The provided context does not contain enough information to answer this question."
- Never invent or extrapolate financial figures.
- Keep answers concise and structured; use bullet points for lists.
"""

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")