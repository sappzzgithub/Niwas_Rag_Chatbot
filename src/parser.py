"""
src/parser.py  —  Phase 1: PDF Parsing
=======================================
Uses pymupdf4llm to extract page-wise Markdown that preserves:
  • Flowing body text (paragraphs, headers)
  • Tables rendered as GitHub-Flavoured Markdown pipe tables
  • Inline formatting from the PDF's structure

Outputs
-------
  data/parsed/parsed.json          — list of page dicts
  data/parsed/pages_md/page_NNN.md — one Markdown file per page

Why pymupdf4llm?
  ✓ Understands PDF layout order better than raw fitz text extraction
  ✓ Converts tables to GFM pipe format — parseable downstream
  ✓ Retains heading hierarchy via # markers
  ✓ Zero API cost — fully local

Page splitting strategy (in order of preference):
  1. Form-feed character (\f) — pymupdf4llm's native separator
  2. fitz get_text("markdown") — per-page markdown via PyMuPDF directly
  3. fitz get_text("blocks")   — plain text blocks fallback
"""

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF — for page count, metadata, and fallback extraction
import pymupdf4llm

import config as cfg
from src.utils import get_logger, save_json, timer

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _detect_content_types(markdown: str) -> list[str]:
    """
    Heuristically classify what content types appear on a page.
    Returns a list such as ['text'], ['text', 'table'], etc.
    """
    types: list[str] = ["text"]
    # GFM pipe table: at least one | ... | row
    if re.search(r"^\|.+\|", markdown, re.MULTILINE):
        types.append("table")
    return types


def _extract_section_hint(markdown: str, page_num: int) -> str:
    """
    Return a short section label based on headings found on the page,
    or a page-range hint if no heading is detected.
    """
    # Find the first Markdown heading
    match = re.search(r"^#{1,3}\s+(.+)$", markdown, re.MULTILINE)
    if match:
        return match.group(1).strip()[:80]

    # Fall back to document-structure hints based on page number
    if 1 <= page_num <= 11:
        return "Corporate Overview"
    elif 12 <= page_num <= 18:
        return "Management Discussion & Analysis"
    elif 19 <= page_num <= 49:
        return "Board's Report & Governance"
    elif 61 <= page_num <= 65:
        return "Financial Statements"
    elif 66 <= page_num <= 133:
        return "Notes to Accounts"
    else:
        return "General"


def _count_words(text: str) -> int:
    return len(text.split())


# ─────────────────────────────────────────────
# CORE CLASS
# ─────────────────────────────────────────────

class PDFParser:
    """
    Parses a PDF using pymupdf4llm and emits structured page-level records.

    Each record contains:
      page_num       : 1-based page number
      markdown       : full GFM Markdown for the page
      content_types  : ['text'] | ['text', 'table']
      section        : inferred section label
      word_count     : rough size indicator
      has_content    : False for blank/near-blank pages
    """

    def __init__(self, pdf_path: Path = cfg.PDF_PATH):
        self.pdf_path = pdf_path
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ------------------------------------------------------------------
    def parse(self) -> list[dict[str, Any]]:
        """
        Run the full parse.  Returns the list of page records AND writes
        parsed.json + per-page .md files to disk.
        """
        logger.info(f"Parsing PDF: {self.pdf_path.name}")

        # ── Step 1: Get total page count from fitz ────────────────────
        doc = fitz.open(str(self.pdf_path))
        total_pages = doc.page_count
        doc.close()
        logger.info(f"PDF has {total_pages} pages total")

        # ── Step 2: Try pymupdf4llm full-doc extraction ───────────────
        pages_md = self._try_pymupdf4llm_split(total_pages)

        # ── Step 3: Validate — if we didn't get the right page count,
        #            fall back to fitz page-by-page ────────────────────
        if len(pages_md) != total_pages:
            logger.warning(
                f"pymupdf4llm returned {len(pages_md)} pages "
                f"(expected {total_pages}). Falling back to fitz extraction."
            )
            pages_md = self._extract_via_fitz(total_pages)

        logger.info(f"Final page count: {len(pages_md)}")

        # ── Step 4: Build records and persist ────────────────────────
        records: list[dict[str, Any]] = []
        for idx, md in enumerate(pages_md):
            page_num = idx + 1
            record = {
                "page_num": page_num,
                "markdown": md,
                "content_types": _detect_content_types(md),
                "section": _extract_section_hint(md, page_num),
                "word_count": _count_words(md),
                "has_content": _count_words(md) > 5,
            }
            records.append(record)
            self._write_page_md(page_num, md)

        # ── Step 5: Persist JSON ──────────────────────────────────────
        save_json(records, cfg.PARSED_JSON)
        logger.info(f"Saved parsed.json  ({len(records)} pages)")
        logger.info(
            f"Pages with tables : "
            f"{sum(1 for r in records if 'table' in r['content_types'])}"
        )
        logger.info(
            f"Blank/near-blank  : "
            f"{sum(1 for r in records if not r['has_content'])}"
        )
        return records

    # ------------------------------------------------------------------
    def _try_pymupdf4llm_split(self, total_pages: int) -> list[str]:
        """
        Attempt full-document extraction with pymupdf4llm, then split
        on the form-feed (\f) character that separates pages.
        Returns a list of per-page markdown strings.
        """
        try:
            with timer("pymupdf4llm full-doc extraction", logger):
                full_md: str = pymupdf4llm.to_markdown(
                    str(self.pdf_path),
                    show_progress=True,
                    table_strategy="lines_strict",
                )

            # Strategy 1: form-feed separator (pymupdf4llm native)
            if "\f" in full_md:
                pages = full_md.split("\f")
                pages = [p.strip() for p in pages]
                # Remove empty trailing pages
                while pages and not pages[-1]:
                    pages.pop()
                logger.info(
                    f"Split via \\f separator → {len(pages)} pages"
                )
                if len(pages) == total_pages:
                    return pages
                logger.warning(
                    f"\\f split gave {len(pages)} pages, expected {total_pages}"
                )
                return pages  # still return; outer logic will fallback if wrong

            # Strategy 2: horizontal-rule separator (some pymupdf4llm versions)
            if "\n---\n" in full_md:
                pages = full_md.split("\n---\n")
                pages = [p.strip() for p in pages if p.strip()]
                logger.info(
                    f"Split via '---' separator → {len(pages)} pages"
                )
                return pages

            # Strategy 3: no separator found — return whole doc as single page
            logger.warning(
                "No page separator found in pymupdf4llm output. "
                "Will use fitz fallback."
            )
            return [full_md.strip()]

        except Exception as exc:
            logger.warning(f"pymupdf4llm extraction failed: {exc}. Using fitz fallback.")
            return []

    # ------------------------------------------------------------------
    def _extract_via_fitz(self, total_pages: int) -> list[str]:
        """
        Reliable page-by-page extraction using PyMuPDF (fitz) directly.
        Tries get_text("markdown") first (PyMuPDF ≥ 1.24), then falls
        back to assembling text from blocks.

        Returns a list of per-page markdown strings (one per page).
        """
        logger.info(
            f"Extracting {total_pages} pages via fitz (page-by-page) …"
        )
        doc = fitz.open(str(self.pdf_path))
        pages: list[str] = []

        # Detect whether fitz supports markdown output (PyMuPDF ≥ 1.24)
        supports_markdown = hasattr(fitz, "TOOLS") and fitz.version[0] >= "1.24"
        try:
            # Quick probe on page 0
            doc[0].get_text("markdown")
            supports_markdown = True
        except Exception:
            supports_markdown = False

        logger.info(
            f"fitz markdown mode: {'enabled' if supports_markdown else 'disabled (using blocks)'}"
        )

        with timer("fitz page-by-page extraction", logger):
            for page_index in range(total_pages):
                page = doc[page_index]
                md = self._extract_page_text(page, supports_markdown)
                pages.append(md)

                if (page_index + 1) % 20 == 0:
                    logger.info(
                        f"  … extracted page {page_index + 1}/{total_pages}"
                    )

        doc.close()
        logger.info(f"fitz extraction complete: {len(pages)} pages")
        return pages

    # ------------------------------------------------------------------
    def _extract_page_text(self, page: fitz.Page, supports_markdown: bool) -> str:
        """
        Extract text from a single fitz page.
        Tries markdown mode first, then dict/blocks mode.
        """
        # ── Method 1: native markdown (PyMuPDF ≥ 1.24) ───────────────
        if supports_markdown:
            try:
                text = page.get_text("markdown")
                if text and text.strip():
                    return text.strip()
            except Exception:
                pass

        # ── Method 2: structured blocks → reconstruct text ───────────
        try:
            blocks = page.get_text("blocks")
            # blocks: list of (x0, y0, x1, y1, text, block_no, block_type)
            # block_type 0 = text, 1 = image
            text_blocks = [
                b[4].strip()
                for b in sorted(blocks, key=lambda b: (b[1], b[0]))  # sort top→bottom, left→right
                if b[6] == 0 and b[4].strip()
            ]
            if text_blocks:
                return "\n\n".join(text_blocks)
        except Exception:
            pass

        # ── Method 3: plain text fallback ────────────────────────────
        try:
            text = page.get_text("text")
            if text and text.strip():
                return text.strip()
        except Exception:
            pass

        return ""  # completely blank page

    # ------------------------------------------------------------------
    def _write_page_md(self, page_num: int, markdown: str) -> None:
        """Write a single page's Markdown to pages_md/page_NNN.md."""
        path = cfg.PAGES_MD_DIR / f"page_{page_num:03d}.md"
        path.write_text(markdown, encoding="utf-8")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run() -> list[dict]:
    """Public entry point — called by pipeline runner and notebooks."""
    parser = PDFParser()
    return parser.parse()


if __name__ == "__main__":
    records = run()
    print(f"\n✓ Parsed {len(records)} pages.")
    print(f"  Sample page 1 preview:\n{records[0]['markdown'][:300]}")
    print(f"  Sample page 61 preview:\n{records[60]['markdown'][:300]}")