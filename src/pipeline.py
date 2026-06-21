"""
src/pipeline.py  —  Master Pipeline Runner
==========================================
Runs all 9 processing phases in order (Phase 10 is the Streamlit app,
launched separately).

Usage
-----
    # Run all phases
    python src/pipeline.py

    # Run specific phases only
    python src/pipeline.py --phases 1 2 3

    # Skip vision (useful if no API key yet)
    python src/pipeline.py --skip 4
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger, timer

logger = get_logger("pipeline")


PHASE_REGISTRY: dict[int, dict] = {
    1: {"name": "PDF Parsing",          "module": "src.parser"},
    2: {"name": "Image Extraction",     "module": "src.image_extractor"},
    3: {"name": "Vision Page Selection","module": "src.vision_selector"},
    4: {"name": "Vision Understanding", "module": "src.vision_analyzer"},
    5: {"name": "Multimodal Chunking",  "module": "src.chunker"},
    6: {"name": "Embedding + Indexing", "module": "src.embedder"},   # phases 6+7
}


def run_phase(phase_num: int) -> None:
    phase = PHASE_REGISTRY[phase_num]
    logger.info(f"{'='*60}")
    logger.info(f"  PHASE {phase_num}: {phase['name']}")
    logger.info(f"{'='*60}")

    import importlib
    module = importlib.import_module(phase["module"])

    t0 = time.perf_counter()
    module.run()
    elapsed = time.perf_counter() - t0
    logger.info(f"  → Phase {phase_num} complete in {elapsed:.1f}s\n")


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal RAG Pipeline Runner"
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        type=int,
        default=list(PHASE_REGISTRY.keys()),
        help="Which phases to run (default: all)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        type=int,
        default=[],
        help="Phases to skip",
    )
    args = parser.parse_args()

    phases_to_run = sorted(
        [p for p in args.phases if p not in args.skip]
    )

    logger.info(f"Phases to run: {phases_to_run}")
    total_start = time.perf_counter()

    for phase_num in phases_to_run:
        if phase_num not in PHASE_REGISTRY:
            logger.warning(f"Phase {phase_num} not in registry — skipping")
            continue
        run_phase(phase_num)

    total_elapsed = time.perf_counter() - total_start
    logger.info(f"Pipeline finished in {total_elapsed:.1f}s")
    logger.info("Next: run `streamlit run app.py` to launch the chatbot UI.")


if __name__ == "__main__":
    main()