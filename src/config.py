"""Shared configuration: taxonomy, paths, and model settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = ROOT / "outputs"

# Any LLM that speaks the OpenAI-style chat API works. Pick one with LLM_PROVIDER (or --provider):
#   openai : OpenAI API            needs OPENAI_API_KEY
#   gemini : Google Gemini API     needs GEMINI_API_KEY (has a free tier)
#   ollama : local open models     needs Ollama running locally, no key, no cost
# Model names change over time: check the provider's docs and override with LLM_MODEL or --model.
PROVIDERS = {
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY", "default_model": "gpt-4o-mini"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY", "default_model": "gemini-2.0-flash"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None, "default_model": "llama3.1"},
}
PROVIDER = os.getenv("LLM_PROVIDER", "openai")
MODEL = os.getenv("LLM_MODEL") or PROVIDERS[PROVIDER]["default_model"]

# Issue taxonomy (kept deliberately small so hand-labelling 50 reviews is realistic).
THEMES = [
    "product_defect",       # stopped working, faulty, broke, dead on arrival
    "performance_quality",  # works, but battery/picture/sound/speed/cooling is poor
    "delivery_logistics",   # late, wrong item, missing item, courier problems
    "packaging_damage",     # arrived damaged or poorly packed
    "after_sales_service",  # warranty, repair, installation, support, returns/refunds
    "price_value",          # overpriced, not worth it, great value
    "usability_design",     # hard to use, heavy, bulky, bad design
    "no_issue",             # positive or neutral review with no complaint
    "other",
]
SENTIMENTS = ["positive", "neutral", "negative"]
SEVERITY = {1: "minor annoyance", 2: "significant problem", 3: "critical (unusable / safety / money lost)"}
