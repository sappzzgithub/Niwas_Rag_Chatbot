# """
# src/retriever.py  —  Phase 8: Hybrid Retrieval
# ===============================================
# Combines dense vector search (ChromaDB + BGE embeddings) with sparse
# keyword search (BM25) using Weighted Reciprocal Rank Fusion (RRF) to
# produce a single ranked list of the most relevant chunks.

# Why hybrid?
# -----------
# - Dense search excels at semantic similarity ("loan book growth")
# - BM25 excels at exact keyword matches ("AUM", "Tamil Nadu", "31,512.30")
# - Weighted RRF merges both — BM25 gets higher weight for financial docs
#   because annual reports are keyword-dense (exact numbers, section names)

# Architecture
# ------------
#   Query
#     │
#     ├─► BGE embed ──► ChromaDB query ──► dense_results (top 50)
#     │
#     └─► BM25 tokenise ──► sparse_results (top 50)
#                                 │
#                       Weighted RRF (dense=1.0, sparse=1.5)
#                                 │
#                            top_k chunks
# """

# from __future__ import annotations

# import sys
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any

# sys.path.insert(0, str(Path(__file__).parent.parent))

# import config as cfg
# from src.utils import get_logger, load_json

# logger = get_logger(__name__)

# # BGE query prefix for asymmetric retrieval
# _BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# # RRF constant — 60 is standard; higher = smoother rank blending
# _RRF_K = 60

# # RRF weights — BM25 gets higher weight for financial keyword-dense documents.
# # Annual reports contain exact numbers (31,512.30), section names, and
# # structured financial terms that BM25 matches precisely.
# # Configurable via config.py / .env.
# _DENSE_WEIGHT  = float(getattr(cfg, "RRF_DENSE_WEIGHT",  1.0))
# _SPARSE_WEIGHT = float(getattr(cfg, "RRF_SPARSE_WEIGHT", 1.5))


# # ─────────────────────────────────────────────
# # DATA CLASS
# # ─────────────────────────────────────────────

# @dataclass
# class RetrievedChunk:
#     """One retrieved chunk with its score and metadata."""
#     id:           str
#     text:         str
#     score:        float
#     page_num:     int
#     content_type: str
#     section:      str
#     source:       str
#     render_path:  str = ""
#     rank:         int = 0

#     def citation(self) -> str:
#         """Human-readable citation string."""
#         parts = [self.section] if self.section else []
#         parts.append(f"page {self.page_num}")
#         if self.content_type == "table":
#             parts.insert(0, "Table from")
#         elif self.content_type == "image":
#             parts.insert(0, "Figure/Chart from")
#         return "Source: " + ", ".join(parts)


# # ─────────────────────────────────────────────
# # BM25 INDEX  (built once, cached in memory)
# # ─────────────────────────────────────────────

# class BM25Index:
#     """
#     Lightweight BM25 index over all chunks.
#     Built from chunks.json on first use and cached for the process lifetime.
#     """

#     def __init__(self, chunks: list[dict]):
#         try:
#             from rank_bm25 import BM25Okapi
#         except ImportError:
#             raise ImportError(
#                 "pip install rank-bm25  — required for hybrid retrieval"
#             )

#         self.chunks = chunks
#         # Tokenise: lowercase, split on whitespace.
#         # Keeps numbers and special chars intact — critical for financials
#         corpus = [self._tokenise(c["text"]) for c in chunks]
#         self.bm25 = BM25Okapi(corpus)
#         logger.info(f"BM25 index built over {len(chunks)} chunks")

#     @staticmethod
#     def _tokenise(text: str) -> list[str]:
#         """Simple whitespace tokeniser — consistent with query tokenisation."""
#         return text.lower().split()

#     def query(self, query: str, top_k: int) -> list[tuple[int, float]]:
#         """
#         Return list of (chunk_index, bm25_score) sorted descending by score.
#         Only returns chunks with score > 0.
#         """
#         tokens = self._tokenise(query)
#         scores = self.bm25.get_scores(tokens)
#         ranked = sorted(
#             enumerate(scores), key=lambda x: x[1], reverse=True
#         )
#         return [(idx, score) for idx, score in ranked[:top_k] if score > 0]


# # ─────────────────────────────────────────────
# # WEIGHTED RECIPROCAL RANK FUSION
# # ─────────────────────────────────────────────

# def _rrf_score(rank: int, k: int = _RRF_K) -> float:
#     """RRF score for a chunk at position `rank` (1-indexed)."""
#     return 1.0 / (k + rank)


# def _reciprocal_rank_fusion(
#     dense_ids: list[str],
#     sparse_ids: list[str],
#     dense_weight: float = _DENSE_WEIGHT,
#     sparse_weight: float = _SPARSE_WEIGHT,
# ) -> list[tuple[str, float]]:
#     """
#     Merge two ranked lists using Weighted Reciprocal Rank Fusion.

#     sparse_weight > dense_weight boosts BM25 results, appropriate for
#     financial document retrieval where exact keyword matches are reliable.

#     Configurable via config.py:
#       RRF_DENSE_WEIGHT  = 1.0  (default)
#       RRF_SPARSE_WEIGHT = 1.5  (default)
#     """
#     scores: dict[str, float] = {}

#     for rank, chunk_id in enumerate(dense_ids, start=1):
#         scores[chunk_id] = (
#             scores.get(chunk_id, 0.0) + dense_weight * _rrf_score(rank)
#         )

#     for rank, chunk_id in enumerate(sparse_ids, start=1):
#         scores[chunk_id] = (
#             scores.get(chunk_id, 0.0) + sparse_weight * _rrf_score(rank)
#         )

#     return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# # ─────────────────────────────────────────────
# # RETRIEVER CLASS
# # ─────────────────────────────────────────────

# class Retriever:
#     """
#     Hybrid retriever: BM25 + dense vector search fused with weighted RRF.

#     Usage
#     -----
#         retriever = Retriever()
#         chunks = retriever.retrieve("What is the total loan book?", top_k=5)
#     """

#     # Pool size for each retriever before fusion
#     _POOL_SIZE = 50

#     def __init__(self):
#         from src.embedder import get_embedding_model, get_or_create_collection

#         self.collection = get_or_create_collection()
#         self.model = get_embedding_model()

#         if not cfg.CHUNKS_JSON.exists():
#             raise FileNotFoundError(
#                 f"chunks.json not found at {cfg.CHUNKS_JSON}. "
#                 "Run Phase 5 (chunker.py) first."
#             )
#         all_chunks = load_json(cfg.CHUNKS_JSON)
#         self._chunks_by_id: dict[str, dict] = {
#             c["id"]: c for c in all_chunks
#         }
#         self._bm25 = BM25Index(all_chunks)
#         self._all_chunk_ids: list[str] = [c["id"] for c in all_chunks]

#         logger.info(
#             f"Retriever ready — collection has {self.collection.count()} docs"
#         )

#     # ------------------------------------------------------------------
#     def retrieve(
#         self,
#         query: str,
#         top_k: int = cfg.TOP_K,
#         content_type_filter: str | None = None,
#     ) -> list[RetrievedChunk]:
#         """
#         Hybrid retrieve: BM25 + dense → weighted RRF → top_k chunks.

#         Parameters
#         ----------
#         query               : natural-language question
#         top_k               : number of results to return
#         content_type_filter : optional — restrict to "text" | "table" | "image"
#         """
#         if self.collection.count() == 0:
#             raise RuntimeError(
#                 "ChromaDB collection is empty. Run Phase 7 (embedder.py) first."
#             )

#         pool = self._POOL_SIZE

#         # ── Dense retrieval ───────────────────────────────────────────
#         dense_ids = self._dense_retrieve(query, top_k=pool)

#         # ── Sparse (BM25) retrieval ───────────────────────────────────
#         sparse_ids = self._sparse_retrieve(query, top_k=pool)

#         # ── Weighted Reciprocal Rank Fusion ───────────────────────────
#         fused = _reciprocal_rank_fusion(
#             dense_ids,
#             sparse_ids,
#             dense_weight=_DENSE_WEIGHT,
#             sparse_weight=_SPARSE_WEIGHT,
#         )

#         # ── Build RetrievedChunk objects ──────────────────────────────
#         chunks: list[RetrievedChunk] = []
#         rank = 1

#         for chunk_id, rrf_score in fused:
#             if rank > top_k * 3:  # safety limit
#                 break

#             chunk_data = self._chunks_by_id.get(chunk_id)
#             if chunk_data is None:
#                 continue

#             if content_type_filter:
#                 if chunk_data.get("content_type") != content_type_filter:
#                     continue

#             chunks.append(
#                 RetrievedChunk(
#                     id=chunk_id,
#                     text=chunk_data["text"],
#                     score=rrf_score,
#                     page_num=int(chunk_data.get("page_num", 0)),
#                     content_type=str(chunk_data.get("content_type", "text")),
#                     section=str(chunk_data.get("section", "")),
#                     source=str(chunk_data.get("source", "")),
#                     render_path=str(chunk_data.get("render_path", "")),
#                     rank=rank,
#                 )
#             )
#             rank += 1

#             if len(chunks) == top_k:
#                 break

#         logger.info(
#             f"Hybrid retrieve: {len(dense_ids)} dense + "
#             f"{len(sparse_ids)} BM25 → {len(chunks)} chunks after RRF "
#             f"(dense={_DENSE_WEIGHT}, sparse={_SPARSE_WEIGHT})"
#         )
#         return chunks

#     # ------------------------------------------------------------------
#     def _dense_retrieve(self, query: str, top_k: int) -> list[str]:
#         """
#         Query ChromaDB with the embedded query vector.
#         Returns ordered list of chunk IDs (best first).
#         """
#         query_text = (
#             _BGE_QUERY_PREFIX + query
#             if "bge" in cfg.EMBEDDING_MODEL.lower()
#             else query
#         )
#         q_vec = self.model.encode(
#             [query_text],
#             normalize_embeddings=True,
#         ).tolist()

#         results = self.collection.query(
#             query_embeddings=q_vec,
#             n_results=min(top_k, self.collection.count()),
#             include=["distances"],
#         )
#         return results["ids"][0]

#     # ------------------------------------------------------------------
#     def _sparse_retrieve(self, query: str, top_k: int) -> list[str]:
#         """
#         BM25 keyword search over all chunks.
#         Returns ordered list of chunk IDs (best first).
#         """
#         ranked = self._bm25.query(query, top_k=top_k)
#         return [self._all_chunk_ids[idx] for idx, _ in ranked]

#     # ------------------------------------------------------------------
#     def retrieve_for_display(
#         self, query: str, top_k: int = cfg.TOP_K
#     ) -> list[dict[str, Any]]:
#         """Convenience wrapper returning plain dicts for Streamlit / JSON."""
#         chunks = self.retrieve(query, top_k=top_k)
#         return [
#             {
#                 "rank":         c.rank,
#                 "page_num":     c.page_num,
#                 "content_type": c.content_type,
#                 "section":      c.section,
#                 "score":        round(c.score, 4),
#                 "citation":     c.citation(),
#                 "text":         c.text,
#                 "render_path":  c.render_path,
#             }
#             for c in chunks
#         ]


# # ─────────────────────────────────────────────
# # ENTRY POINT  (quick smoke test)
# # ─────────────────────────────────────────────

# if __name__ == "__main__":
#     TEST_QUERIES = [
#         "What is the total loan book as at March 2025?",
#         "AUM disbursement growth chart Corporate Overview",
#         "geographic state-wise loan portfolio distribution",
#         "key risks MD&A credit market liquidity",
#         "strategy for scaling loan book Board's Report",
#         "total interest income FY2025 FY2024 comparison",
#     ]
#     r = Retriever()
#     for q in TEST_QUERIES:
#         print(f"\n{'='*60}")
#         print(f"Query: {q}")
#         results = r.retrieve(q, top_k=5)
#         for c in results:
#             print(
#                 f"  [{c.rank}] p.{c.page_num} ({c.content_type}) "
#                 f"rrf={c.score:.4f}  {c.text[:80]} …"
#             )
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