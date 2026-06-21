"""
src/utils.py
============
Shared utilities: logging setup, timing decorator, JSON I/O helpers.
Imported by every other module — keep this dependency-light.
"""

import json
import logging
import time
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import config as cfg


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, cfg.LOG_LEVEL, logging.INFO))
        logger.propagate = False
    return logger


# ─────────────────────────────────────────────
# TIMING
# ─────────────────────────────────────────────

@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    """Context manager that prints elapsed time for a block."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    msg = f"{label} completed in {elapsed:.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


# ─────────────────────────────────────────────
# JSON HELPERS
# ─────────────────────────────────────────────

def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Atomically write JSON to path, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    tmp.replace(path)


def load_json(path: Path) -> Any:
    """Load and return JSON from path; raises FileNotFoundError with a clear message."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run the preceding pipeline phase first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# MISC
# ─────────────────────────────────────────────

def chunk_list(lst: list, size: int) -> list[list]:
    """Split a list into sublists of at most `size` items."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def truncate(text: str, max_chars: int = 200) -> str:
    """Return a truncated preview of text for logging."""
    return text[:max_chars] + "…" if len(text) > max_chars else text