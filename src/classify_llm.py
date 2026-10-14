"""Classify every review with an LLM and save structured labels.

Each review is sent with a strict instruction to return JSON only:
    {"theme": ..., "sentiment": ..., "severity": 0-3, "reason": "<short>"}

Features: response cache (re-runs cost nothing), retries with backoff, resumable, small thread pool.

Usage (pick a provider; see src/config.py):
    export OPENAI_API_KEY=...            # or GEMINI_API_KEY=..., or run Ollama locally (no key)
    python -m src.classify_llm --input data/reviews_sample.csv --provider openai --limit 20   # try a few first
    python -m src.classify_llm --input data/reviews_sample.csv --provider openai              # full run
    python -m src.classify_llm --input data/reviews_sample.csv --dry-run                      # print a prompt, no API call
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .config import MODEL, OUT_DIR, PROVIDER, PROVIDERS, SENTIMENTS, THEMES

PROMPT_VERSION = "v1"


SYSTEM_PROMPT = f"""You are an analyst who tags e-commerce product reviews for an after-sales operations team.
For the review you are given, return ONLY a JSON object (no markdown, no commentary) with these keys:

"theme": the single MAIN issue, one of {THEMES}
  - product_defect: stopped working, faulty, broke, dead on arrival
  - performance_quality: works but is poor (battery, picture, sound, speed, cooling, build quality)
  - delivery_logistics: late, wrong or missing item, courier problems
  - packaging_damage: arrived damaged or badly packed
  - after_sales_service: warranty, repair, installation, support, returns or refunds
  - price_value: overpriced or not worth it
  - usability_design: hard to use, heavy, bulky, confusing design
  - no_issue: positive or neutral review with no complaint
  - other: a complaint that fits none of the above
"sentiment": one of {SENTIMENTS}
"severity": integer. 0 = no issue; 1 = minor annoyance; 2 = significant problem; 3 = critical (unusable, safety risk, or money lost)
"reason": at most 15 words explaining your choice

If a review contains several complaints, choose the one that matters most to the customer.
Judge only what the review says. Do not invent details."""


def build_user_prompt(review: str, category: str) -> str:
    return f"Product category: {category}\nReview: {review}"


def parse_response(text: str) -> dict:
    """Extract and validate the JSON object; raise ValueError if it is unusable."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in: {text[:120]!r}")
    obj = json.loads(match.group(0))
    theme = str(obj.get("theme", "")).strip().lower()
    sentiment = str(obj.get("sentiment", "")).strip().lower()
    if theme not in THEMES:
        raise ValueError(f"Invalid theme: {theme!r}")
    if sentiment not in SENTIMENTS:
        raise ValueError(f"Invalid sentiment: {sentiment!r}")
    severity = int(obj.get("severity", 0))
    if severity not in (0, 1, 2, 3):
        raise ValueError(f"Invalid severity: {severity}")
    if theme == "no_issue":
        severity = 0
    return {"theme": theme, "sentiment": sentiment, "severity": severity, "reason": str(obj.get("reason", ""))[:200]}


def call_model(client, review: str, category: str, model: str, retries: int = 4) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(review, category)},
                ],
            )
            return parse_response(resp.choices[0].message.content or "")
        except Exception as e:  # network errors, rate limits, bad JSON: back off and retry
            last_err = e
            time.sleep(min(2 ** attempt, 15))
    return {"theme": "error", "sentiment": "error", "severity": -1, "reason": f"failed: {last_err}"[:200]}


def load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["key"]] = rec["result"]
    return cache


def make_client(provider: str):
    """Build an OpenAI-compatible client for the chosen provider (imported lazily so tests need no key)."""
    import os

    from openai import OpenAI

    if provider not in PROVIDERS:
        raise SystemExit(f"Unknown provider {provider!r}. Choose from {list(PROVIDERS)}")
    cfg = PROVIDERS[provider]
    key = os.getenv(cfg["key_env"]) if cfg["key_env"] else "ollama"
    if not key:
        raise SystemExit(f"Set the {cfg['key_env']} environment variable first.")
    return OpenAI(api_key=key, base_url=cfg["base_url"])


def classify_dataframe(df: pd.DataFrame, client, model: str = MODEL, cache_path: Path = OUT_DIR / "llm_cache.jsonl",
                       workers: int = 4) -> pd.DataFrame:
    cache = load_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    key = lambda rid: f"{PROMPT_VERSION}|{model}|{rid}"
    todo = [r for r in df.itertuples() if key(r.review_id) not in cache]
    print(f"{len(df)} reviews | {len(df) - len(todo)} cached | {len(todo)} to classify with {model}")

    def work(r):
        return r.review_id, call_model(client, r.review, r.category, model)

    with open(cache_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (rid, res) in enumerate(ex.map(work, todo), 1):
            if res["theme"] != "error":  # never cache failures, so a re-run retries them
                cache[key(rid)] = res
                f.write(json.dumps({"key": key(rid), "result": res}) + "\n")
                f.flush()
            else:
                cache[key(rid)] = res
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")

    rows = [{"review_id": r.review_id, **cache[key(r.review_id)]} for r in df.itertuples()]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=OUT_DIR / "llm_labels.csv")
    ap.add_argument("--limit", type=int, default=None, help="only classify the first N reviews (cheap test)")
    ap.add_argument("--provider", default=PROVIDER, choices=list(PROVIDERS))
    ap.add_argument("--model", default=None, help="defaults to the provider's default model (or LLM_MODEL)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    if args.dry_run:
        r = df.iloc[0]
        print("=== SYSTEM ===\n" + SYSTEM_PROMPT + "\n\n=== USER ===\n" + build_user_prompt(r.review, r.category))
        return

    model = args.model or (MODEL if args.provider == PROVIDER else PROVIDERS[args.provider]["default_model"])
    client = make_client(args.provider)
    out = classify_dataframe(df, client, model=model, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    errors = (out["theme"] == "error").sum()
    print(f"Wrote {len(out)} labels -> {args.output}  ({errors} errors)")


if __name__ == "__main__":
    main()
