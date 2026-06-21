"""
src/vision_selector.py  —  Phase 3: Vision Page Selection
==========================================================
Decides which pages are worth sending to the expensive vision model.
Selection is driven by two complementary signals:

  Signal A — Text heuristics (from parsed.json):
    • Page contains visual keywords (chart, graph, AUM, %, ₹ crore …)
    • Page has low word count but many embedded images
    • Page is in a known visual-rich section (Corporate Overview p.1-11,
      MD&A p.12-18, Financial Statements p.61-65)

  Signal B — Image presence (from image_metadata.json):
    • Page has ≥ 1 embedded image above the minimum size threshold

Pages scoring above a threshold (or matching forced ranges) are marked
for vision analysis and written to vision_pages.json.

Output
------
  data/images/vision_pages.json — list of page dicts with selection reason
"""

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from src.utils import get_logger, load_json, save_json, timer

logger = get_logger(__name__)


# Pages that MUST be analysed regardless of score (assignment requirement)
FORCED_PAGE_RANGES: list[tuple[int, int]] = [
    (1, 11),    # Corporate Overview — AUM/disbursement charts
    (12, 18),   # MD&A — geographic/state-wise portfolio
    (61, 65),   # Financial Statements — Balance Sheet, P&L
]


def _page_in_forced_range(page_num: int) -> bool:
    return any(lo <= page_num <= hi for lo, hi in FORCED_PAGE_RANGES)


def _score_page(
    page_record: dict,
    image_page: dict,
    keyword_set: set[str],
) -> tuple[int, list[str]]:
    """
    Return (score, reasons) for a page.
    score ≥ 2  →  selected for vision analysis.
    """
    reasons: list[str] = []
    score = 0
    md = page_record.get("markdown", "").lower()

    # ── Keyword hits ───────────────────────────────────────────────
    hits = [kw for kw in keyword_set if kw in md]
    if hits:
        score += min(len(hits), 3)   # cap contribution at 3
        reasons.append(f"Keywords: {', '.join(hits[:5])}")

    # ── Embedded images present ───────────────────────────────────
    n_images = len(image_page.get("embedded_images", []))
    if n_images > 0:
        score += 2
        reasons.append(f"{n_images} embedded image(s)")

    # ── Low word-count + images → probably an image-heavy page ───
    word_count = page_record.get("word_count", 999)
    if word_count < 100 and n_images > 0:
        score += 1
        reasons.append("Low text density + images")

    # ── Forced section ─────────────────────────────────────────────
    if _page_in_forced_range(page_record["page_num"]):
        score += 2
        reasons.append("In required section range")

    return score, reasons


class VisionPageSelector:
    """
    Reads parsed.json and image_metadata.json and produces vision_pages.json.
    """

    SCORE_THRESHOLD = 2

    def __init__(self):
        self.parsed_pages: list[dict] = []
        self.image_meta: dict = {}
        self.keyword_set = {kw.lower() for kw in cfg.VISUAL_KEYWORDS}

    def select(self) -> list[dict[str, Any]]:
        """Run selection and return list of selected page records."""
        logger.info("Loading parsed.json and image_metadata.json …")
        self.parsed_pages = load_json(cfg.PARSED_JSON)
        self.image_meta = load_json(cfg.IMAGE_METADATA_JSON)

        selected: list[dict[str, Any]] = []

        with timer("Vision page scoring", logger):
            for page_record in self.parsed_pages:
                page_num = page_record["page_num"]
                image_page = self.image_meta["pages"].get(str(page_num), {})

                score, reasons = _score_page(
                    page_record, image_page, self.keyword_set
                )

                if score >= self.SCORE_THRESHOLD:
                    selected.append(
                        {
                            "page_num": page_num,
                            "score": score,
                            "reasons": reasons,
                            "section": page_record.get("section", ""),
                            "page_render": image_page.get("page_render", ""),
                            "embedded_images": image_page.get(
                                "embedded_images", []
                            ),
                            "word_count": page_record.get("word_count", 0),
                        }
                    )

        selected.sort(key=lambda x: x["page_num"])

        save_json(selected, cfg.VISION_PAGES_JSON)
        logger.info(
            f"Selected {len(selected)} / {len(self.parsed_pages)} pages "
            f"for vision analysis"
        )
        # Log page number summary
        page_nums = [s["page_num"] for s in selected]
        logger.info(f"Selected pages: {page_nums}")
        return selected


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run() -> list[dict]:
    selector = VisionPageSelector()
    return selector.select()


if __name__ == "__main__":
    pages = run()
    print(f"\n✓ {len(pages)} pages selected for vision.")
    for p in pages[:5]:
        print(f"  p.{p['page_num']:>3}  score={p['score']}  reasons={p['reasons']}")