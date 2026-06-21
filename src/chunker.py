"""
src/chunker.py  —  Phase 5: Multimodal Chunk Builder
=====================================================
Merges three content streams into a unified set of retrievable chunks:

  Stream 1 — Markdown text  (from parsed.json)
  Stream 2 — GFM pipe tables (extracted from the Markdown)
  Stream 3 — Vision descriptions (from vision_descriptions.json)

Chunking strategy
-----------------
  • Text is split with RecursiveCharacterTextSplitter (LangChain).
  • Tables are kept whole (never split mid-row) and emitted as single chunks.
  • Vision descriptions are emitted as single chunks tagged content_type=image.
  • Every chunk carries metadata: page_num, content_type, section, source.

Output
------
  data/chunks/chunks.json  — list of chunk dicts ready for embedding
"""

import re
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

import config as cfg
from src.utils import get_logger, load_json, save_json, timer

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# TABLE EXTRACTION
# ─────────────────────────────────────────────

_TABLE_RE = re.compile(
    r"(\|.+\|\n(\|[-: |]+\|\n)(\|.+\|\n)*)",  # header | separator | rows
    re.MULTILINE,
)


def extract_tables_from_markdown(markdown: str) -> tuple[list[str], str]:
    """
    Find all GFM pipe tables in `markdown`.
    Returns:
      tables      — list of raw table strings (preserving rows intact)
      cleaned_md  — markdown with tables removed (to avoid double-indexing)
    """
    tables: list[str] = []
    cleaned = markdown

    for match in _TABLE_RE.finditer(markdown):
        table_str = match.group(0).strip()
        # Must have at least 2 data rows to be a real table (not a 1-row artefact)
        rows = [l for l in table_str.split("\n") if l.startswith("|") and "---" not in l]
        if len(rows) >= 2:
            tables.append(table_str)

    # Remove tables from the main text
    cleaned = _TABLE_RE.sub("\n", cleaned).strip()
    return tables, cleaned


# ─────────────────────────────────────────────
# CHUNK FACTORY
# ─────────────────────────────────────────────

def _make_chunk(
    text: str,
    page_num: int,
    content_type: str,   # "text" | "table" | "image"
    section: str,
    extra: dict | None = None,
) -> dict[str, Any]:
    """Construct a chunk dict with a stable UUID."""
    chunk: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "text": text.strip(),
        "page_num": page_num,
        "content_type": content_type,
        "section": section,
        "source": f"page {page_num}",
    }
    if extra:
        chunk.update(extra)
    return chunk


# ─────────────────────────────────────────────
# CHUNKER CLASS
# ─────────────────────────────────────────────

class MultimodalChunker:
    """
    Builds the final chunk list from parsed.json + vision_descriptions.json.

    Processing order per page:
      1. Extract tables → whole-table chunks
      2. Split remaining text → text chunks
      3. If page has a vision description → one image chunk
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.CHUNK_SIZE_CHARS,
            chunk_overlap=cfg.CHUNK_OVERLAP_CHARS,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    def build(self) -> list[dict[str, Any]]:
        """Build and persist chunks.json. Returns the chunk list."""
        logger.info("Loading parsed.json …")
        parsed_pages: list[dict] = load_json(cfg.PARSED_JSON)

        # Vision descriptions are optional (Phase 4 may not have run yet)
        vision_descriptions: dict[str, Any] = {}
        if cfg.VISION_DESCRIPTIONS_JSON.exists():
            vision_descriptions = load_json(cfg.VISION_DESCRIPTIONS_JSON)
            logger.info(
                f"Loaded {len(vision_descriptions)} vision descriptions"
            )
        else:
            logger.warning(
                "vision_descriptions.json not found — "
                "image chunks will be omitted. Run Phase 4 first."
            )

        chunks: list[dict[str, Any]] = []

        with timer("Chunk building", logger):
            for page_record in parsed_pages:
                page_num = page_record["page_num"]
                markdown = page_record.get("markdown", "")
                section = page_record.get("section", "")

                if not markdown.strip():
                    continue

                # ── Tables ────────────────────────────────────────────
                tables, text_without_tables = extract_tables_from_markdown(
                    markdown
                )
                for table_str in tables:
                    chunk = _make_chunk(
                        text=table_str,
                        page_num=page_num,
                        content_type="table",
                        section=section,
                    )
                    chunks.append(chunk)

                # ── Text ──────────────────────────────────────────────
                if text_without_tables.strip():
                    splits = self.splitter.split_text(text_without_tables)
                    for split in splits:
                        if len(split.strip()) < 30:  # skip near-empty splits
                            continue
                        chunk = _make_chunk(
                            text=split,
                            page_num=page_num,
                            content_type="text",
                            section=section,
                        )
                        chunks.append(chunk)

                # ── Vision / Image ────────────────────────────────────
                vision_data = vision_descriptions.get(str(page_num))
                if vision_data:
                    desc = vision_data.get("description", "")
                    if desc and desc != "NO_VISUAL_CONTENT" and not desc.startswith("ERROR"):
                        chunk = _make_chunk(
                            text=f"[Visual content on page {page_num}]\n{desc}",
                            page_num=page_num,
                            content_type="image",
                            section=section,
                            extra={
                                "render_path": vision_data.get("render_path", ""),
                            },
                        )
                        chunks.append(chunk)

        logger.info(f"Total chunks: {len(chunks)}")
        logger.info(
            f"  text={sum(1 for c in chunks if c['content_type']=='text')} | "
            f"  table={sum(1 for c in chunks if c['content_type']=='table')} | "
            f"  image={sum(1 for c in chunks if c['content_type']=='image')}"
        )

        save_json(chunks, cfg.CHUNKS_JSON)
        logger.info(f"Saved chunks.json ({len(chunks)} chunks)")
        return chunks


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run() -> list[dict]:
    chunker = MultimodalChunker()
    return chunker.build()


if __name__ == "__main__":
    chunks = run()
    print(f"\n✓ {len(chunks)} chunks built.")
    sample = chunks[0]
    print(f"  Sample chunk:\n"
          f"    id={sample['id']}\n"
          f"    page={sample['page_num']}  type={sample['content_type']}\n"
          f"    text={sample['text'][:200]}")