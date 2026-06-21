from __future__ import annotations
"""
src/embedder.py  —  Phase 6 & 7: Embeddings + ChromaDB Indexing
================================================================
Phase 6: Generates embeddings for every chunk using a Sentence Transformers
         model (BAAI/bge-base-en-v1.5 by default — strong on retrieval tasks).

Phase 7: Stores chunks, embeddings, and all metadata into a persistent
         ChromaDB collection on disk.

Why BAAI/bge-base-en-v1.5?
  ✓ Top-ranked on MTEB retrieval benchmarks at this size class
  ✓ 768-dim embeddings, runs well on CPU
  ✓ Optimised for asymmetric retrieval (query ≠ passage length)

ChromaDB notes:
  • The collection is keyed by PDF_NAME from config — safe to re-run.
  • Existing documents with the same ID are upserted (not duplicated).
  • Metadata stored: page_num, content_type, section, source, render_path.
"""

import sys
from pathlib import Path
from typing import Any
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import config as cfg
from src.utils import get_logger, load_json, chunk_list, timer

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# EMBEDDING MODEL (singleton within a process)
# ─────────────────────────────────────────────

_model_cache: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        logger.info(f"Loading embedding model: {cfg.EMBEDDING_MODEL}")
        _model_cache = SentenceTransformer(cfg.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model_cache


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings; returns list of float vectors."""
    model = get_embedding_model()
    # BGE models benefit from a query prefix — for passages we use none
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via dot product
        convert_to_numpy=True,
    )
    return vectors.tolist()


# ─────────────────────────────────────────────
# CHROMA CLIENT (singleton within a process)
# ─────────────────────────────────────────────

_chroma_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(cfg.VECTORDB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_or_create_collection() -> chromadb.Collection:
    """
    Return the ChromaDB collection for this PDF.
    Uses cosine distance — compatible with normalised BGE embeddings.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=cfg.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ─────────────────────────────────────────────
# INDEXER CLASS
# ─────────────────────────────────────────────

class VectorIndexer:
    """
    Loads chunks.json, generates embeddings in batches, and upserts
    everything into the ChromaDB collection.
    """

    BATCH_SIZE = 64   # chunks per embedding + upsert batch

    def index(self) -> chromadb.Collection:
        """Run indexing pipeline. Returns the ChromaDB collection."""
        chunks: list[dict[str, Any]] = load_json(cfg.CHUNKS_JSON)
        logger.info(f"Indexing {len(chunks)} chunks into ChromaDB …")
        logger.info(f"  Collection : {cfg.CHROMA_COLLECTION}")
        logger.info(f"  Store path : {cfg.VECTORDB_DIR}")

        collection = get_or_create_collection()

        # Check what's already indexed
        existing_count = collection.count()
        logger.info(f"  Existing documents in collection: {existing_count}")

        with timer("Embedding + indexing", logger):
            for batch in chunk_list(chunks, self.BATCH_SIZE):
                ids     = [c["id"]   for c in batch]
                texts   = [c["text"] for c in batch]
                metas   = [self._build_metadata(c) for c in batch]

                logger.debug(f"Embedding batch of {len(batch)} chunks …")
                embeddings = embed_texts(texts)

                collection.upsert(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metas,
                )

        final_count = collection.count()
        logger.info(
            f"Indexing complete. Collection size: {final_count} documents."
        )
        return collection

    # ------------------------------------------------------------------
    @staticmethod
    def _build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
        """
        Build the metadata dict for a chunk.
        ChromaDB metadata values must be str | int | float | bool.
        """
        return {
            "page_num":     int(chunk.get("page_num", 0)),
            "content_type": str(chunk.get("content_type", "text")),
            "section":      str(chunk.get("section", "")),
            "source":       str(chunk.get("source", "")),
            "render_path":  str(chunk.get("render_path", "")),
        }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run() -> chromadb.Collection:
    indexer = VectorIndexer()
    return indexer.index()


if __name__ == "__main__":
    col = run()
    print(f"\n✓ Collection '{cfg.CHROMA_COLLECTION}' has {col.count()} documents.")