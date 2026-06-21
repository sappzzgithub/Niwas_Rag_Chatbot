"""
demo.py  —  Assignment Demo Script
====================================
Answers all 6 required test questions and prints:
  (a) Retrieved chunks (rank, page, type, score, text preview)
  (b) Final answer with source citations
  (c) Elapsed time per question

Usage
-----
    python demo.py                    # all 6 questions
    python demo.py --question 3       # single question by index (1-based)
    python demo.py --output results/  # also save JSON results to a directory
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from src.rag_engine import RAGEngine, RAGAnswer
from src.utils import get_logger

logger = get_logger("demo")


# ─────────────────────────────────────────────
# REQUIRED TEST QUESTIONS
# ─────────────────────────────────────────────

QUESTIONS: list[dict] = [
    {
        "id": 1,
        "type": "TEXT",
        "question": "What are the key risks to Niwas's business highlighted in the MD&A section?",
        "source_hint": "MD&A, pages 12-18",
    },
    {
        "id": 2,
        "type": "TEXT",
        "question": "What is Niwas's stated strategy for scaling its loan book as described in the Board's Report?",
        "source_hint": "Board's Report, pages 19-24",
    },
    {
        "id": 3,
        "type": "TABLE",
        "question": "What is the total loan book (Loans line) as at March 31, 2025 vs March 31, 2024?",
        "source_hint": "Balance Sheet, page 61",
    },
    {
        "id": 4,
        "type": "TABLE",
        "question": "What was Niwas's total interest income for FY2025 and how did it compare to FY2024?",
        "source_hint": "P&L, page 62",
    },
    {
        "id": 5,
        "type": "IMAGE",
        "question": "Describe the trend or distribution shown in the geographic / state-wise loan portfolio figure.",
        "source_hint": "pages 15-16",
    },
    {
        "id": 6,
        "type": "IMAGE",
        "question": "What does the AUM or disbursement chart in the Corporate Overview section show about growth over the years?",
        "source_hint": "Corporate Overview, pages 3-9",
    },
]

DIVIDER = "=" * 72
THIN    = "─" * 72


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────

def print_question_header(q: dict) -> None:
    print(f"\n{DIVIDER}")
    print(f"  Q{q['id']} [{q['type']}]  {q['question']}")
    print(f"  Source hint: {q['source_hint']}")
    print(DIVIDER)


def print_retrieved_chunks(chunks: list[dict]) -> None:
    print(f"\n📎 Retrieved Chunks ({len(chunks)} total):")
    print(THIN)
    for c in chunks:
        type_icon = {"text": "📄", "table": "📊", "image": "🖼"}.get(
            c["content_type"], "•"
        )
        print(
            f"  [{c['rank']}] {type_icon} Page {c['page_num']} | "
            f"{c['content_type'].upper()} | score={c['score']:.4f}"
        )
        print(f"       Section  : {c['section']}")
        print(f"       Citation : {c['citation']}")
        preview = c["text"][:250].replace("\n", " ")
        print(f"       Preview  : {preview}…")
        print()


def print_answer(answer: RAGAnswer) -> None:
    print(f"\n💡 Answer:")
    print(THIN)
    print(answer.answer)
    print()
    print(f"📚 Citations:")
    for cite in answer.citations:
        print(f"   • {cite}")
    print()
    print(
        f"⏱  {answer.elapsed_seconds:.2f}s  |  "
        f"Provider: {answer.provider}/{answer.model}"
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_demo(
    question_ids: list[int] | None = None,
    output_dir: Path | None = None,
) -> list[dict]:
    """
    Run the demo and return a list of result dicts.

    Parameters
    ----------
    question_ids : which questions to run (1-based). None = all.
    output_dir   : if set, save JSON results here.
    """
    print(f"\n{'#' * 72}")
    print("  MULTIMODAL RAG DEMO — Niwas Housing Finance FY2025")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 72}")

    # Show active provider
    print(f"\n  LLM Provider : {cfg.LLM_PROVIDER}")
    if cfg.LLM_PROVIDER == "groq":
        print(f"  Model        : {cfg.GROQ_LLM_MODEL}")
        print(f"  Question delay: {cfg.LLM_INTER_QUESTION_DELAY}s (TPM rate limit buffer)")
    else:
        print(f"  Model        : {cfg.OLLAMA_LLM_MODEL}")
        print(f"  Question delay: none (local model)")

    engine = RAGEngine()

    qs = QUESTIONS
    if question_ids:
        qs = [q for q in QUESTIONS if q["id"] in question_ids]

    results: list[dict] = []
    total_start = time.perf_counter()

    for i, q_meta in enumerate(qs):
        # ── Inter-question delay (Groq TPM rate limit protection) ─────
        if i > 0 and cfg.LLM_PROVIDER == "groq":
            delay = cfg.LLM_INTER_QUESTION_DELAY
            logger.info(
                f"Waiting {delay}s before Q{q_meta['id']} "
                f"(Groq TPM limit protection) …"
            )
            time.sleep(delay)

        print_question_header(q_meta)
        answer: RAGAnswer = engine.ask(q_meta["question"])

        print_retrieved_chunks(answer.retrieved_chunks)
        print_answer(answer)

        results.append(
            {
                "question_id":   q_meta["id"],
                "type":          q_meta["type"],
                "source_hint":   q_meta["source_hint"],
                **answer.to_dict(),
            }
        )

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{DIVIDER}")
    print(f"  All {len(qs)} question(s) answered in {total_elapsed:.1f}s")
    print(DIVIDER)

    # ── Save JSON output ──────────────────────────────────────────────
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = output_dir / f"demo_results_{ts}.json"
        out_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n✓ Results saved to: {out_file}")

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal RAG Demo")
    parser.add_argument(
        "--question",
        type=int,
        nargs="+",
        default=None,
        help="Question ID(s) to run (1-6). Default: all.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Directory to save JSON results (default: outputs/)",
    )
    args = parser.parse_args()

    run_demo(
        question_ids=args.question,
        output_dir=Path(args.output),
    )