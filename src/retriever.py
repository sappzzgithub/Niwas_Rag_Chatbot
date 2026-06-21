"""
src/retriever.py  —  Phase 8: Retrieval
=========================================
Given a natural-language question, embeds the query and returns the top-k
most relevant chunks from ChromaDB, including all metadata.

Design notes
------------
• BGE models recommend a query prefix "Represent this sentence for searching
  relevant passages: " — this module applies it automatically for queries.
• Results are returned sorted by relevance score (lower cosine distance = better).
• An optional content_type filter lets callers restrict to text/table/image.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from src.utils import get_logger

logger = get_logger(__name__)

# BGE query prefix for asymmetric retrieval
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """One retrieved chunk with its score and metadata."""
    id:           str
    text:         str
    score:        float          # cosine distance (lower = more similar)
    page_num:     int
    content_type: str
    section:      str
    source:       str
    render_path:  str = ""
    rank:         int = 0

    def citation(self) -> str:
        """Human-readable citation string."""
        parts = [self.section] if self.section else []
        parts.append(f"page {self.page_num}")
        if self.content_type == "table":
            parts.insert(0, "Table from")
        elif self.content_type == "image":
            parts.insert(0, "Figure/Chart from")
        return "Source: " + ", ".join(parts)


# ─────────────────────────────────────────────
# RETRIEVER CLASS
# ─────────────────────────────────────────────

class Retriever:
    """
    Semantic retriever over the ChromaDB collection.

    Usage
    -----
        retriever = Retriever()
        chunks = retriever.retrieve("What is the total loan book?", top_k=5)
    """

    def __init__(self):
        from src.embedder import get_embedding_model, get_or_create_collection
        self.collection = get_or_create_collection()
        self.model = get_embedding_model()
        logger.info(
            f"Retriever ready — collection has {self.collection.count()} docs"
        )

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = cfg.TOP_K,
        content_type_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Embed the query and return the top-k most relevant chunks.

        Parameters
        ----------
        query               : natural-language question
        top_k               : number of results (default from config)
        content_type_filter : if set, only return chunks of this type
                              ("text" | "table" | "image")
        """
        if self.collection.count() == 0:
            raise RuntimeError(
                "ChromaDB collection is empty. Run Phase 7 (embedder.py) first."
            )

        # Apply BGE query prefix for better retrieval quality
        query_text = (
            _BGE_QUERY_PREFIX + query
            if "bge" in cfg.EMBEDDING_MODEL.lower()
            else query
        )
        logger.debug(f"Embedding query: {query[:80]} …")
        q_vec = self.model.encode(
            [query_text],
            normalize_embeddings=True,
        ).tolist()

        # Build optional where-filter for ChromaDB
        where: dict | None = None
        if content_type_filter:
            where = {"content_type": {"$eq": content_type_filter}}

        results = self.collection.query(
            query_embeddings=q_vec,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        for rank, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            start=1,
        ):
            chunks.append(
                RetrievedChunk(
                    id=results["ids"][0][rank - 1],
                    text=doc,
                    score=float(dist),
                    page_num=int(meta.get("page_num", 0)),
                    content_type=str(meta.get("content_type", "text")),
                    section=str(meta.get("section", "")),
                    source=str(meta.get("source", "")),
                    render_path=str(meta.get("render_path", "")),
                    rank=rank,
                )
            )

        logger.debug(
            f"Retrieved {len(chunks)} chunks for query: '{query[:60]}'"
        )
        return chunks

    # ------------------------------------------------------------------
    def retrieve_for_display(
        self, query: str, top_k: int = cfg.TOP_K
    ) -> list[dict[str, Any]]:
        """
        Convenience wrapper that returns plain dicts (for Streamlit / JSON).
        """
        chunks = self.retrieve(query, top_k=top_k)
        return [
            {
                "rank":         c.rank,
                "page_num":     c.page_num,
                "content_type": c.content_type,
                "section":      c.section,
                "score":        round(c.score, 4),
                "citation":     c.citation(),
                "text":         c.text,
                "render_path":  c.render_path,
            }
            for c in chunks
        ]


# ─────────────────────────────────────────────
# ENTRY POINT  (quick smoke test)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    TEST_QUERY = "What is the total loan book as at March 2025?"
    r = Retriever()
    results = r.retrieve(TEST_QUERY)
    print(f"\n✓ Top-{len(results)} results for: '{TEST_QUERY}'\n")
    for c in results:
        print(f"  [{c.rank}] p.{c.page_num} ({c.content_type})  dist={c.score:.4f}")
        print(f"       {c.text[:120]} …\n")