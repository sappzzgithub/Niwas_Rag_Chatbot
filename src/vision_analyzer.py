"""
src/vision_analyzer.py  —  Phase 4: Vision Understanding
=========================================================
Sends each selected page render to a vision LLM and collects
structured descriptions of charts, tables, infographics, and KPIs.

Provider abstraction
--------------------
  VisionProviderBase   — abstract interface
  GroqVisionProvider   — Groq Llama 4 Scout (meta-llama/llama-4-scout-17b-16e-instruct)
  OllamaVisionProvider — local Ollama (llava:13b etc.)

The active provider is chosen by cfg.VISION_PROVIDER.

Output
------
  data/images/vision_descriptions.json
  — dict keyed by page_num with raw description text + metadata
"""

import base64
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from src.utils import get_logger, load_json, save_json, timer

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _encode_image(image_path: Path) -> str:
    """Base64-encode an image file for API transmission."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _load_image_path(relative_path: str) -> Path:
    """Resolve a relative path (stored in JSON) to an absolute Path."""
    return cfg.ROOT_DIR / relative_path


# ─────────────────────────────────────────────
# PROVIDER INTERFACE
# ─────────────────────────────────────────────

class VisionProviderBase(ABC):
    """All vision providers must implement describe_page()."""

    @abstractmethod
    def describe_page(self, image_path: Path, prompt: str) -> str:
        """
        Send the image at image_path to the vision model with the given prompt.
        Return the model's text response.
        """
        ...


# ─────────────────────────────────────────────
# GROQ PROVIDER  (Llama 4 Scout — natively multimodal)
# ─────────────────────────────────────────────

class GroqVisionProvider(VisionProviderBase):
    """
    Uses Groq API with Llama 4 Scout (meta-llama/llama-4-scout-17b-16e-instruct).
    Llama 4 Scout is natively multimodal — accepts base64 image_url directly.
    Free tier: 30 req/min, 6000 tokens/min.
    """

    def __init__(self):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("pip install groq")

        if not cfg.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")

        self.client = Groq(api_key=cfg.GROQ_API_KEY)
        self.model = cfg.GROQ_VISION_MODEL
        logger.info(f"Groq Vision model: {self.model}")

    def describe_page(self, image_path: Path, prompt: str) -> str:
        b64 = _encode_image(image_path)
        ext = image_path.suffix.lstrip(".").lower()
        # Llama 4 Scout supports png, jpeg, webp, gif
        mime = f"image/{ext}" if ext in {"png", "jpeg", "jpg", "webp", "gif"} else "image/png"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens=1500,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# OLLAMA PROVIDER  (local fallback)
# ─────────────────────────────────────────────

class OllamaVisionProvider(VisionProviderBase):
    """Uses local Ollama server for vision models (llava, bakllava, etc.)."""

    def __init__(self):
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")

        self.base_url = cfg.OLLAMA_BASE_URL
        self.model = cfg.OLLAMA_VISION_MODEL
        self._requests = requests
        logger.info(f"Ollama Vision model: {self.model} @ {self.base_url}")

    def describe_page(self, image_path: Path, prompt: str) -> str:
        b64 = _encode_image(image_path)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1500},
        }
        resp = self._requests.post(
            f"{self.base_url}/api/generate", json=payload, timeout=120
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


# ─────────────────────────────────────────────
# PROVIDER FACTORY
# ─────────────────────────────────────────────

def get_vision_provider() -> VisionProviderBase:
    provider_map = {
        "groq":   GroqVisionProvider,
        "ollama": OllamaVisionProvider,
    }
    key = cfg.VISION_PROVIDER.lower()
    if key not in provider_map:
        raise ValueError(
            f"Unknown VISION_PROVIDER '{key}'. "
            f"Choose from: {list(provider_map)}"
        )
    return provider_map[key]()


# ─────────────────────────────────────────────
# ANALYZER CLASS
# ─────────────────────────────────────────────

class VisionAnalyzer:
    """
    Iterates over vision_pages.json, sends each page render to the
    vision provider, and accumulates descriptions in vision_descriptions.json.

    Supports incremental runs — pages already described are skipped.
    """

    RATE_LIMIT_DELAY = 4.0   # seconds between API calls — Groq free tier is 30 req/min

    def __init__(self):
        self.provider = get_vision_provider()

    def analyze(self) -> dict[str, Any]:
        """Run vision analysis. Returns the descriptions dict."""
        vision_pages: list[dict] = load_json(cfg.VISION_PAGES_JSON)
        logger.info(
            f"Vision analysis: {len(vision_pages)} pages using "
            f"provider={cfg.VISION_PROVIDER} model={cfg.GROQ_VISION_MODEL}"
        )

        # Load any existing partial results for incremental runs
        descriptions: dict[str, Any] = {}
        if cfg.VISION_DESCRIPTIONS_JSON.exists():
            descriptions = load_json(cfg.VISION_DESCRIPTIONS_JSON)
            logger.info(
                f"Resuming — {len(descriptions)} pages already described"
            )

        with timer("Vision analysis", logger):
            for i, page_info in enumerate(vision_pages):
                page_num = page_info["page_num"]
                key = str(page_num)

                if key in descriptions:
                    logger.debug(f"Page {page_num}: already described, skipping")
                    continue

                render_path_rel = page_info.get("page_render", "")
                if not render_path_rel:
                    logger.warning(f"Page {page_num}: no render path, skipping")
                    continue

                image_path = _load_image_path(render_path_rel)
                if not image_path.exists():
                    logger.warning(f"Page {page_num}: render not found at {image_path}")
                    continue

                logger.info(
                    f"[{i+1}/{len(vision_pages)}] Analysing page {page_num} "
                    f"(section: {page_info.get('section', '?')})"
                )

                description = self._safe_describe(image_path, page_num)

                descriptions[key] = {
                    "page_num": page_num,
                    "section": page_info.get("section", ""),
                    "description": description,
                    "score": page_info.get("score", 0),
                    "reasons": page_info.get("reasons", []),
                    "render_path": render_path_rel,
                }

                # Save incrementally after each page — safe to Ctrl+C and resume
                save_json(descriptions, cfg.VISION_DESCRIPTIONS_JSON)

                # Rate limiting — Groq free tier: 30 req/min
                if i < len(vision_pages) - 1:
                    time.sleep(self.RATE_LIMIT_DELAY)

        logger.info(
            f"Vision descriptions saved: {len(descriptions)} pages → "
            f"{cfg.VISION_DESCRIPTIONS_JSON}"
        )
        return descriptions

    def _safe_describe(self, image_path: Path, page_num: int) -> str:
        """Call the provider with exponential backoff retries."""
        for attempt in range(3):
            try:
                desc = self.provider.describe_page(image_path, cfg.VISION_PROMPT)
                if desc == "NO_VISUAL_CONTENT":
                    logger.debug(f"Page {page_num}: no visual content detected")
                return desc
            except Exception as exc:
                wait = 2 ** (attempt + 1)   # 2s, 4s, 8s
                logger.warning(
                    f"Page {page_num} attempt {attempt+1} failed: {exc}. "
                    f"Retrying in {wait}s …"
                )
                time.sleep(wait)
        return f"ERROR: Could not analyse page {page_num}"


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run() -> dict:
    analyzer = VisionAnalyzer()
    return analyzer.analyze()


if __name__ == "__main__":
    descs = run()
    sample_key = next(iter(descs))
    print(f"\n✓ {len(descs)} pages described.")
    print(f"  Sample (page {sample_key}):\n"
          f"  {descs[sample_key]['description'][:400]}")