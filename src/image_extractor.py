"""
src/image_extractor.py  —  Phase 2: Image Extraction
=====================================================
Uses PyMuPDF (fitz) to:
  1. Extract every embedded image from the PDF (logos, photos, chart bitmaps).
  2. Render each full page as a PNG at configurable DPI — used downstream
     for vision analysis even when no embedded image exists on that page.

Outputs
-------
  data/images/embedded/page_NNN_img_M.png   — raw embedded images
  data/images/page_renders/page_NNN.png     — full-page renders
  data/images/image_metadata.json           — index of everything above

Why PyMuPDF (fitz) for images?
  ✓ Fastest PDF renderer available in Python
  ✓ Access to the raw xref image objects (no re-compression artefacts)
  ✓ Controllable DPI for renders — 150 DPI gives crisp text + small files
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF

import config as cfg
from src.utils import get_logger, save_json, timer

logger = get_logger(__name__)


class ImageExtractor:
    """
    Extracts embedded images and renders full-page PNGs from a PDF.

    Attributes
    ----------
    pdf_path : Path   — source PDF
    metadata : dict   — accumulated image_metadata (populated after extract())
    """

    def __init__(self, pdf_path: Path = cfg.PDF_PATH):
        self.pdf_path = pdf_path
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        self.metadata: dict[str, Any] = {
            "pdf": str(pdf_path.name),
            "pages": {},
        }

    # ------------------------------------------------------------------
    def extract(self) -> dict[str, Any]:
        """
        Run extraction and rendering for every page.
        Returns the metadata dict and saves image_metadata.json.
        """
        doc = fitz.open(str(self.pdf_path))
        total = doc.page_count
        logger.info(f"Extracting images from {total} pages — DPI={cfg.PAGE_RENDER_DPI}")

        with timer("Image extraction + rendering", logger):
            for page_index in range(total):
                page_num = page_index + 1
                page = doc[page_index]

                page_record: dict[str, Any] = {
                    "page_num": page_num,
                    "embedded_images": [],
                    "page_render": None,
                }

                # ── 1. Render full page ──────────────────────────────
                render_path = self._render_page(page, page_num)
                page_record["page_render"] = str(render_path.relative_to(cfg.ROOT_DIR))

                # ── 2. Extract embedded images ───────────────────────
                image_list = page.get_images(full=True)
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    save_path = self._save_embedded_image(doc, xref, page_num, img_idx)
                    if save_path is None:
                        continue  # too small or unsupported
                    page_record["embedded_images"].append(
                        {
                            "xref": xref,
                            "index": img_idx,
                            "path": str(save_path.relative_to(cfg.ROOT_DIR)),
                        }
                    )

                self.metadata["pages"][str(page_num)] = page_record

                if page_num % 20 == 0:
                    logger.info(f"  … processed page {page_num}/{total}")

        doc.close()
        save_json(self.metadata, cfg.IMAGE_METADATA_JSON)
        logger.info(f"Saved image_metadata.json")
        total_embedded = sum(
            len(v["embedded_images"]) for v in self.metadata["pages"].values()
        )
        logger.info(
            f"Total embedded images : {total_embedded} | "
            f"Page renders : {total}"
        )
        return self.metadata

    # ------------------------------------------------------------------
    def _render_page(self, page: fitz.Page, page_num: int) -> Path:
        """Render a page to PNG at configured DPI and return the path."""
        mat = fitz.Matrix(cfg.PAGE_RENDER_DPI / 72, cfg.PAGE_RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = cfg.PAGE_RENDERS_DIR / f"page_{page_num:03d}.png"
        pix.save(str(out_path))
        return out_path

    # ------------------------------------------------------------------
    def _save_embedded_image(
        self,
        doc: fitz.Document,
        xref: int,
        page_num: int,
        img_idx: int,
    ) -> Path | None:
        """
        Extract an embedded image by xref, skip if it is too small.
        Returns the save path or None if skipped.
        """
        try:
            base_image = doc.extract_image(xref)
        except Exception as exc:
            logger.debug(f"Could not extract xref {xref}: {exc}")
            return None

        img_bytes = base_image["image"]
        width = base_image.get("width", 0)
        height = base_image.get("height", 0)
        ext = base_image.get("ext", "png")

        # Skip decorative / tiny images
        if width < cfg.MIN_IMAGE_SIZE_PX or height < cfg.MIN_IMAGE_SIZE_PX:
            return None

        out_path = (
            cfg.EMBEDDED_DIR / f"page_{page_num:03d}_img_{img_idx:02d}.{ext}"
        )
        out_path.write_bytes(img_bytes)
        return out_path


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run() -> dict:
    extractor = ImageExtractor()
    return extractor.extract()


if __name__ == "__main__":
    meta = run()
    pages_with_imgs = sum(
        1 for v in meta["pages"].values() if v["embedded_images"]
    )
    print(f"\n✓ Done. Pages with embedded images: {pages_with_imgs}")