"""
src/rag_engine.py  —  Phase 9: RAG Answer Engine
==================================================
Ties retrieval to LLM generation:

  1. Retrieve top-k chunks (Phase 8).
  2. Build a structured prompt containing all chunk text + metadata.
  3. Send to the configured LLM provider.
  4. Return a structured answer dict with the answer text, citations,
     retrieved chunks, and timing information.

Provider abstraction mirrors vision_analyzer.py:
  GeminiLLMProvider | GroqLLMProvider | OllamaLLMProvider
  → selected by cfg.LLM_PROVIDER
"""

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from src.retriever import Retriever, RetrievedChunk
from src.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# ANSWER DATACLASS
# ─────────────────────────────────────────────

@dataclass
class RAGAnswer:
    question:       str
    answer:         str
    citations:      list[str]
    retrieved_chunks: list[dict[str, Any]]
    elapsed_seconds: float
    provider:       str
    model:          str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question":        self.question,
            "answer":          self.answer,
            "citations":       self.citations,
            "retrieved_chunks": self.retrieved_chunks,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "provider":        self.provider,
            "model":           self.model,
        }


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Construct the user-turn prompt.
    Each chunk is presented with its citation metadata so the LLM can
    reference it accurately.
    """
    context_parts: list[str] = []
    for c in chunks:
        header = (
            f"--- CHUNK {c.rank} | {c.content_type.upper()} | "
            f"Page {c.page_num} | {c.section} ---"
        )
        context_parts.append(f"{header}\n{c.text}")

    context_block = "\n\n".join(context_parts)

    prompt = (
        f"CONTEXT (retrieved from the Niwas Housing Finance FY2025 Annual Report):\n\n"
        f"{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"Provide a precise, well-structured answer using ONLY the context above. "
        f"Include 'Source: <section>, page <N>' citations for every factual claim."
    )
    return prompt


# ─────────────────────────────────────────────
# LLM PROVIDERS
# ─────────────────────────────────────────────

class LLMProviderBase(ABC):
    model_name: str = ""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class GeminiLLMProvider(LLMProviderBase):
    def __init__(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")
        if not cfg.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env")
        genai.configure(api_key=cfg.GEMINI_API_KEY)
        self.model_name = cfg.GEMINI_LLM_MODEL
        self._model = genai.GenerativeModel(
            self.model_name,
            system_instruction=cfg.RAG_SYSTEM_PROMPT,
        )
        logger.info(f"Gemini LLM: {self.model_name}")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # system_prompt already injected via system_instruction above
        response = self._model.generate_content(
            user_prompt,
            generation_config={"temperature": 0.1, "max_output_tokens": 2048},
        )
        return response.text.strip()


class GroqLLMProvider(LLMProviderBase):
    def __init__(self):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("pip install groq")
        if not cfg.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=cfg.GROQ_API_KEY)
        self.model_name = cfg.GROQ_LLM_MODEL
        logger.info(f"Groq LLM: {self.model_name}")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=800,  # reduced from 2048 to stay under TPM limit
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                err = str(exc)
                if "429" in err or "rate_limit" in err.lower():
                    # Parse wait time from error message if available
                    wait = 2 ** (attempt + 2)  # 4s, 8s, 16s, 32s
                    import re
                    match = re.search(r"try again in (\d+\.?\d*)s", err)
                    if match:
                        wait = float(match.group(1)) + 2
                    logger.warning(
                        f"Rate limit hit (attempt {attempt+1}/4). "
                        f"Waiting {wait:.1f}s …"
                    )
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Groq rate limit: all 4 retries exhausted")


class OllamaLLMProvider(LLMProviderBase):
    def __init__(self):
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")
        self._requests = requests
        self.model_name = cfg.OLLAMA_LLM_MODEL
        self.base_url = cfg.OLLAMA_BASE_URL
        logger.info(f"Ollama LLM: {self.model_name} @ {self.base_url}")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 2048},
        }
        resp = self._requests.post(
            f"{self.base_url}/api/generate", json=payload, timeout=180
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


def get_llm_provider() -> LLMProviderBase:
    provider_map = {
        "gemini": GeminiLLMProvider,
        "groq":   GroqLLMProvider,
        "ollama": OllamaLLMProvider,
    }
    key = cfg.LLM_PROVIDER.lower()
    if key not in provider_map:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{key}'. Choose from: {list(provider_map)}"
        )
    return provider_map[key]()


# ─────────────────────────────────────────────
# RAG ENGINE
# ─────────────────────────────────────────────

class RAGEngine:
    """
    End-to-end RAG: retrieve → prompt → generate → structured answer.

    Usage
    -----
        engine = RAGEngine()
        answer = engine.ask("What is Niwas's total loan book?")
        print(answer.answer)
        for cite in answer.citations:
            print(cite)
    """

    def __init__(self):
        self.retriever = Retriever()
        self.llm = get_llm_provider()

    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        top_k: int = cfg.TOP_K,
    ) -> RAGAnswer:
        """
        Answer a question using RAG.

        Parameters
        ----------
        question : str  — natural-language question
        top_k    : int  — number of chunks to retrieve

        Returns
        -------
        RAGAnswer dataclass with answer, citations, and retrieved chunks.
        """
        t0 = time.perf_counter()
        logger.info(f"Question: {question}")

        # ── Retrieve ──────────────────────────────────────────────────
        chunks: list[RetrievedChunk] = self.retriever.retrieve(
            question, top_k=top_k
        )
        logger.info(
            f"Retrieved {len(chunks)} chunks from pages: "
            f"{[c.page_num for c in chunks]}"
        )

        # ── Build prompt ──────────────────────────────────────────────
        user_prompt = build_rag_prompt(question, chunks)

        # ── Generate ──────────────────────────────────────────────────
        answer_text = self.llm.complete(cfg.RAG_SYSTEM_PROMPT, user_prompt)

        elapsed = time.perf_counter() - t0
        logger.info(f"Answer generated in {elapsed:.2f}s")

        # ── Extract citations (unique, ordered) ───────────────────────
        citations = list(
            dict.fromkeys(c.citation() for c in chunks)
        )

        # Serialisable chunk list for downstream display
        chunks_display = [
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

        return RAGAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_chunks=chunks_display,
            elapsed_seconds=elapsed,
            provider=cfg.LLM_PROVIDER,
            model=self.llm.model_name,
        )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    engine = RAGEngine()
    q = "What are the key risks to Niwas's business highlighted in the MD&A section?"
    ans = engine.ask(q)
    print(f"\nQ: {ans.question}")
    print(f"\nA: {ans.answer}")
    print(f"\nCitations:")
    for cite in ans.citations:
        print(f"  {cite}")
    print(f"\n⏱  {ans.elapsed_seconds:.2f}s  |  {ans.provider}/{ans.model}")