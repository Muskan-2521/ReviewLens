"""Measure how accurate the labels are against YOUR hand labels (the 'gold' set).

Compares: (a) LLM vs gold, (b) keyword baseline vs gold.

Usage:
    python -m src.evaluate --gold data/gold_labels.csv --sample data/reviews_sample.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .baseline import classify_df
from .config import OUT_DIR, THEMES


def score(gold: pd.Series, pred: pd.Series) -> dict:
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("data/gold_labels.csv"))
    ap.add_argument("--sample", type=Path, default=Path("data/reviews_sample.csv"))
    ap.add_argument("--llm", type=Path, default=OUT_DIR / "llm_labels.csv")
    args = ap.parse_args()

    gold = pd.read_csv(args.gold)
    gold = gold[gold["theme"].notna() & (gold["theme"].astype(str).str.strip() != "")].copy()
    if gold.empty:
        raise SystemExit("Gold file has no labelled rows. Fill the theme/sentiment/severity columns first.")
    bad = set(gold["theme"]) - set(THEMES)
    if bad:
        raise SystemExit(f"Unknown themes in gold file: {bad}. Allowed: {THEMES}")

    sample = pd.read_csv(args.sample)
    llm = pd.read_csv(args.llm)
    base = classify_df(sample)

    m = gold[["review_id", "theme", "sentiment", "severity"]].rename(columns=lambda c: f"gold_{c}" if c != "review_id" else c)
    m = m.merge(llm[["review_id", "theme", "sentiment", "severity"]].rename(columns=lambda c: f"llm_{c}" if c != "review_id" else c), on="review_id", how="inner")
    m = m.merge(base.rename(columns=lambda c: f"base_{c}" if c != "review_id" else c), on="review_id", how="inner")
    m = m[m["llm_theme"] != "error"]
    n = len(m)
    if n == 0:
        raise SystemExit("No overlap between gold labels and LLM labels (check review_id values).")

    llm_t, base_t = score(m["gold_theme"], m["llm_theme"]), score(m["gold_theme"], m["base_theme"])
    llm_s = accuracy_score(m["gold_sentiment"], m["llm_sentiment"])
    base_s = accuracy_score(m["gold_sentiment"], m["base_sentiment"])
    sev_exact = (m["gold_severity"].astype(int) == m["llm_severity"].astype(int)).mean()
    sev_within1 = ((m["gold_severity"].astype(int) - m["llm_severity"].astype(int)).abs() <= 1).mean()

    lines = [
        f"# Evaluation on {n} hand-labelled reviews\n",
        "| Metric | Keyword baseline | LLM |",
        "|---|---|---|",
        f"| Theme accuracy | {base_t['accuracy']:.1%} | {llm_t['accuracy']:.1%} |",
        f"| Theme macro-F1 | {base_t['macro_f1']:.2f} | {llm_t['macro_f1']:.2f} |",
        f"| Sentiment accuracy | {base_s:.1%} | {llm_s:.1%} |",
        f"| Severity exact match | - | {sev_exact:.1%} |",
        f"| Severity within +/-1 | - | {sev_within1:.1%} |",
        "",
        f"With only {n} labelled reviews these numbers are noisy (roughly +/-{1.96 * (0.25 / n) ** 0.5:.0%} at 95% confidence). Treat them as indicative, not exact.",
        "",
        "## Per-theme report (LLM)\n```",
        classification_report(m["gold_theme"], m["llm_theme"], zero_division=0),
        "```",
    ]
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    m.to_csv(OUT_DIR / "evaluation_detail.csv", index=False)

    labels = sorted(set(m["gold_theme"]) | set(m["llm_theme"]))
    cm = confusion_matrix(m["gold_theme"], m["llm_theme"], labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xlabel("LLM")
    ax.set_ylabel("Hand label (gold)")
    ax.set_title("Theme confusion matrix")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)

    print("\n".join(lines[:11]))
    print(f"\nSaved outputs/evaluation.md, evaluation_detail.csv, confusion_matrix.png")
    print("Tip: open evaluation_detail.csv and read the rows where gold_theme != llm_theme. "
          "Sometimes your label is the wrong one.")


if __name__ == "__main__":
    main()
