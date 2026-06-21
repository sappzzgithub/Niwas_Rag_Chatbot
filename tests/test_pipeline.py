"""
tests/test_pipeline.py
======================
Unit and integration tests for the Multimodal RAG pipeline.

Run with: pytest tests/ -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import config as cfg
from src.utils import save_json, load_json, chunk_list, truncate
from src.chunker import extract_tables_from_markdown, MultimodalChunker


# ─────────────────────────────────────────────
# UTILS TESTS
# ─────────────────────────────────────────────

class TestUtils:
    def test_save_and_load_json(self, tmp_path):
        data = {"key": "value", "nums": [1, 2, 3]}
        p = tmp_path / "test.json"
        save_json(data, p)
        assert p.exists()
        loaded = load_json(p)
        assert loaded == data

    def test_load_json_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")

    def test_chunk_list(self):
        items = list(range(10))
        chunks = chunk_list(items, 3)
        assert chunks == [[0,1,2],[3,4,5],[6,7,8],[9]]

    def test_truncate(self):
        long_text = "a" * 300
        result = truncate(long_text, max_chars=100)
        assert result.endswith("…")
        assert len(result) == 101

        short_text = "hello"
        assert truncate(short_text) == "hello"


# ─────────────────────────────────────────────
# TABLE EXTRACTION TESTS
# ─────────────────────────────────────────────

class TestTableExtraction:
    SAMPLE_MARKDOWN_WITH_TABLE = """
## Financial Highlights

Some text here.

| Metric         | FY2025     | FY2024     |
|----------------|-----------|-----------|
| Total Income   | ₹100 Cr   | ₹80 Cr    |
| Net Profit     | ₹20 Cr    | ₹15 Cr    |
| Loan Book      | ₹500 Cr   | ₹400 Cr   |

More text after the table.
"""

    def test_extracts_table(self):
        tables, cleaned = extract_tables_from_markdown(
            self.SAMPLE_MARKDOWN_WITH_TABLE
        )
        assert len(tables) == 1
        assert "Total Income" in tables[0]
        assert "₹100 Cr" in tables[0]

    def test_removes_table_from_cleaned_text(self):
        tables, cleaned = extract_tables_from_markdown(
            self.SAMPLE_MARKDOWN_WITH_TABLE
        )
        # Table should be gone from cleaned
        assert "| Metric" not in cleaned
        # Other text should remain
        assert "Financial Highlights" in cleaned
        assert "More text after" in cleaned

    def test_no_table(self):
        plain = "Just a paragraph of text with no tables whatsoever."
        tables, cleaned = extract_tables_from_markdown(plain)
        assert tables == []
        assert cleaned.strip() == plain.strip()

    def test_single_row_not_counted_as_table(self):
        # Only header + separator, no data rows → not a real table
        single_row = "| Col1 | Col2 |\n|------|------|\n"
        tables, _ = extract_tables_from_markdown(single_row)
        # Should be excluded (less than 2 data rows)
        assert len(tables) == 0


# ─────────────────────────────────────────────
# CHUNKER TESTS
# ─────────────────────────────────────────────

class TestChunker:
    def test_chunk_text_basic(self, tmp_path, monkeypatch):
        """Chunker should emit text chunks from parsed.json."""
        # Fake parsed.json
        fake_parsed = [
            {
                "page_num": 1,
                "markdown": "This is a long paragraph. " * 50,
                "content_types": ["text"],
                "section": "Test Section",
                "word_count": 100,
                "has_content": True,
            }
        ]
        parsed_path = tmp_path / "parsed.json"
        save_json(fake_parsed, parsed_path)
        chunks_path = tmp_path / "chunks.json"

        # Patch config paths
        monkeypatch.setattr(cfg, "PARSED_JSON", parsed_path)
        monkeypatch.setattr(cfg, "CHUNKS_JSON", chunks_path)
        monkeypatch.setattr(cfg, "VISION_DESCRIPTIONS_JSON", tmp_path / "vd.json")

        from src.chunker import MultimodalChunker
        chunker = MultimodalChunker()
        chunks = chunker.build()

        assert len(chunks) >= 1
        for c in chunks:
            assert "id" in c
            assert "text" in c
            assert "page_num" in c
            assert "content_type" in c
            assert c["content_type"] in {"text", "table", "image"}

    def test_table_chunk_not_split(self, tmp_path, monkeypatch):
        """Tables must not be split across chunks."""
        big_table = "| Col1 | Col2 |\n|------|------|\n"
        # Add enough rows to exceed chunk size if split
        for i in range(30):
            big_table += f"| Row{i} value long text here | Another value {i} |\n"

        fake_parsed = [
            {
                "page_num": 61,
                "markdown": big_table,
                "content_types": ["table"],
                "section": "Financial Statements",
                "word_count": 200,
                "has_content": True,
            }
        ]
        parsed_path = tmp_path / "parsed.json"
        save_json(fake_parsed, parsed_path)
        chunks_path = tmp_path / "chunks.json"

        monkeypatch.setattr(cfg, "PARSED_JSON", parsed_path)
        monkeypatch.setattr(cfg, "CHUNKS_JSON", chunks_path)
        monkeypatch.setattr(cfg, "VISION_DESCRIPTIONS_JSON", tmp_path / "vd.json")

        from src.chunker import MultimodalChunker
        chunker = MultimodalChunker()
        chunks = chunker.build()

        table_chunks = [c for c in chunks if c["content_type"] == "table"]
        assert len(table_chunks) == 1, "Table should be a single unsplit chunk"
        # Verify the chunk contains multiple rows
        assert "Row0" in table_chunks[0]["text"]
        assert "Row29" in table_chunks[0]["text"]

    def test_vision_chunk_included(self, tmp_path, monkeypatch):
        """Vision descriptions should produce image-type chunks."""
        fake_parsed = [
            {
                "page_num": 5,
                "markdown": "Some text on this page.",
                "content_types": ["text"],
                "section": "Corporate Overview",
                "word_count": 10,
                "has_content": True,
            }
        ]
        fake_vision = {
            "5": {
                "page_num": 5,
                "section": "Corporate Overview",
                "description": "Bar chart showing AUM growth from ₹100 Cr in FY2021 to ₹500 Cr in FY2025.",
                "score": 4,
                "reasons": ["In required section"],
                "render_path": "data/images/page_renders/page_005.png",
            }
        }
        parsed_path = tmp_path / "parsed.json"
        vision_path = tmp_path / "vd.json"
        chunks_path = tmp_path / "chunks.json"
        save_json(fake_parsed, parsed_path)
        save_json(fake_vision, vision_path)

        monkeypatch.setattr(cfg, "PARSED_JSON", parsed_path)
        monkeypatch.setattr(cfg, "CHUNKS_JSON", chunks_path)
        monkeypatch.setattr(cfg, "VISION_DESCRIPTIONS_JSON", vision_path)

        from src.chunker import MultimodalChunker
        chunker = MultimodalChunker()
        chunks = chunker.build()

        image_chunks = [c for c in chunks if c["content_type"] == "image"]
        assert len(image_chunks) == 1
        assert "AUM growth" in image_chunks[0]["text"]
        assert image_chunks[0]["page_num"] == 5


# ─────────────────────────────────────────────
# VISION SELECTOR TESTS
# ─────────────────────────────────────────────

class TestVisionSelector:
    def test_forced_pages_always_selected(self, tmp_path, monkeypatch):
        """Pages 1-11, 12-18, 61-65 must always be selected."""
        from src.vision_selector import VisionPageSelector, _page_in_forced_range

        assert _page_in_forced_range(1) is True
        assert _page_in_forced_range(11) is True
        assert _page_in_forced_range(15) is True
        assert _page_in_forced_range(61) is True
        assert _page_in_forced_range(65) is True
        assert _page_in_forced_range(50) is False
        assert _page_in_forced_range(133) is False

    def test_keyword_scoring(self, tmp_path, monkeypatch):
        """Pages with visual keywords get a higher score."""
        from src.vision_selector import _score_page

        page_with_keywords = {
            "page_num": 20,
            "markdown": "The AUM growth chart shows disbursement trends with ₹ crore breakdown.",
            "word_count": 50,
        }
        image_page_empty = {"embedded_images": []}
        keyword_set = {"aum", "chart", "disbursement", "₹", "crore"}

        score, reasons = _score_page(page_with_keywords, image_page_empty, keyword_set)
        assert score >= 2  # should be selected
        assert len(reasons) >= 1

    def test_images_boost_score(self, tmp_path, monkeypatch):
        from src.vision_selector import _score_page

        page = {"page_num": 30, "markdown": "Minimal text.", "word_count": 10}
        image_page_with_imgs = {
            "embedded_images": [{"xref": 1}, {"xref": 2}]
        }
        score, reasons = _score_page(page, image_page_with_imgs, set())
        # Low text + 2 images → score should reflect images
        assert score >= 2


# ─────────────────────────────────────────────
# RAG ENGINE (mocked LLM)
# ─────────────────────────────────────────────

class TestRAGEngine:
    def test_build_rag_prompt(self):
        """Prompt should include chunk text and question."""
        from src.rag_engine import build_rag_prompt
        from src.retriever import RetrievedChunk

        chunks = [
            RetrievedChunk(
                id="abc",
                text="Total loans were ₹500 Cr as at March 2025.",
                score=0.1,
                page_num=61,
                content_type="table",
                section="Financial Statements",
                source="page 61",
                rank=1,
            )
        ]
        question = "What is the total loan book?"
        prompt = build_rag_prompt(question, chunks)

        assert "₹500 Cr" in prompt
        assert "CHUNK 1" in prompt
        assert question in prompt
        assert "Page 61" in prompt

    def test_retrieved_chunk_citation(self):
        from src.retriever import RetrievedChunk

        c = RetrievedChunk(
            id="x", text="", score=0.2, page_num=61,
            content_type="table", section="Financial Statements",
            source="page 61", rank=1,
        )
        assert "page 61" in c.citation()
        assert "Table" in c.citation()

        c2 = RetrievedChunk(
            id="y", text="", score=0.3, page_num=15,
            content_type="image", section="MD&A",
            source="page 15", rank=2,
        )
        assert "Figure" in c2.citation()
        assert "page 15" in c2.citation()