"""Load a raw review CSV, clean it, draw a balanced sample, and create a hand-labelling template.

Usage:
    python -m src.prepare_data --input data/raw/amazon_reviews.csv --n 800
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from .config import DATA_DIR

# The raw file's column names may differ slightly between sources, so map flexibly.
CANDIDATES = {
    "review_id": ["Unique_ID", "unique_id", "review_id", "id", "ID"],
    "header": ["Review_Header", "review_header", "title", "summary", "Title"],
    "text": ["Review_Text", "review_text", "text", "Review", "review", "reviewText", "Text"],
    "rating": ["Rating", "rating", "overall", "stars", "Stars"],
    "category": ["Category", "category", "product_category", "Product_Category"],
}


def _pick(df: pd.DataFrame, key: str):
    for col in CANDIDATES[key]:
        if col in df.columns:
            return col
    return None


def read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def load_and_clean(path: Path, min_chars: int = 20, max_chars: int = 1500) -> pd.DataFrame:
    raw = read_csv_any(path)
    cols = {k: _pick(raw, k) for k in CANDIDATES}
    if cols["text"] is None:
        raise ValueError(f"No review-text column found. Columns present: {list(raw.columns)}")

    df = pd.DataFrame()
    df["review_id"] = raw[cols["review_id"]].astype(str) if cols["review_id"] else [f"r{i}" for i in range(len(raw))]
    df["category"] = raw[cols["category"]].astype(str) if cols["category"] else "unknown"
    df["rating"] = pd.to_numeric(raw[cols["rating"]], errors="coerce") if cols["rating"] else pd.NA
    header = raw[cols["header"]].fillna("").astype(str) if cols["header"] else ""
    body = raw[cols["text"]].fillna("").astype(str)
    df["review"] = (header + ". " + body if cols["header"] else body).str.replace(r"\s+", " ", regex=True).str.strip(" .")

    df = df[df["review"].str.len().between(min_chars, max_chars)]
    df = df.drop_duplicates(subset="review").reset_index(drop=True)
    # make ids unique even if the source ids repeat
    if df["review_id"].duplicated().any():
        df["review_id"] = [f"r{i}" for i in range(len(df))]
    return df


def balanced_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Equal-sized random draw per category (so one big category does not dominate)."""
    cats = df["category"].unique()
    per_cat = max(1, n // len(cats))
    parts = [g.sample(min(len(g), per_cat), random_state=seed) for _, g in df.groupby("category")]
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out.head(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--n", type=int, default=800, help="reviews to analyse")
    ap.add_argument("--gold-n", type=int, default=50, help="reviews to hand-label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "reviews_sample.csv")
    args = ap.parse_args()

    df = load_and_clean(args.input)
    print(f"Clean reviews: {len(df):,}  | categories: {df['category'].value_counts().to_dict()}")
    sample = balanced_sample(df, args.n, args.seed)
    sample.to_csv(args.out, index=False)
    print(f"Wrote {len(sample)} reviews -> {args.out}")

    gold = sample.sample(min(args.gold_n, len(sample)), random_state=args.seed)[["review_id", "category", "rating", "review"]].copy()
    gold["theme"] = ""
    gold["sentiment"] = ""
    gold["severity"] = ""
    gold_path = args.out.parent / "gold_template.csv"
    gold.to_csv(gold_path, index=False)
    print(f"Wrote hand-labelling template ({len(gold)} rows) -> {gold_path}")
    print("Fill theme / sentiment / severity, then save it as data/gold_labels.csv (see README).")


if __name__ == "__main__":
    main()
