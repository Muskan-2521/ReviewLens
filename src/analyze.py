"""Turn labels into business insight: ranked issues, charts, and example quotes.

Priority score = number of reviews with that issue x average severity.
It ranks issues by how often AND how badly they hurt customers.

Usage:
    python -m src.analyze --sample data/reviews_sample.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import OUT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, default=Path("data/reviews_sample.csv"))
    ap.add_argument("--llm", type=Path, default=OUT_DIR / "llm_labels.csv")
    args = ap.parse_args()

    sample = pd.read_csv(args.sample)
    labels = pd.read_csv(args.llm)
    df = sample.merge(labels, on="review_id", how="inner")
    df = df[df["theme"] != "error"]
    total = len(df)
    OUT_DIR.mkdir(exist_ok=True)

    issues = df[df["theme"] != "no_issue"]
    summary = (
        issues.groupby("theme")
        .agg(reviews=("review_id", "count"), avg_severity=("severity", "mean"))
        .assign(share_of_all_reviews=lambda t: t["reviews"] / total)
        .assign(priority_score=lambda t: t["reviews"] * t["avg_severity"])
        .sort_values("priority_score", ascending=False)
    )
    summary.to_csv(OUT_DIR / "issue_summary.csv")

    by_cat = pd.crosstab(issues["category"], issues["theme"])
    by_cat.to_csv(OUT_DIR / "issues_by_category.csv")

    # Chart 1: priority ranking
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = summary.sort_values("priority_score")
    ax.barh(s.index, s["priority_score"], color="#2b6cb0")
    ax.set_xlabel("Priority score (reviews x avg severity)")
    ax.set_title(f"Which issues matter most? ({total} reviews)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_issues.png", dpi=150)

    # Chart 2: issue mix by product category
    if len(by_cat):
        pct = by_cat.div(by_cat.sum(axis=1), axis=0)
        ax = pct.plot(kind="bar", stacked=True, figsize=(9, 5), colormap="tab20")
        ax.set_ylabel("Share of issues")
        ax.set_title("Issue mix by product category")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "issues_by_category.png", dpi=150)

    # Markdown summary with real example quotes
    md = [f"# Issue summary ({total} reviews analysed)\n",
          f"Reviews with a complaint: {len(issues)} ({len(issues) / total:.0%}).\n",
          "| Rank | Issue | Reviews | Share of all | Avg severity | Priority |", "|---|---|---|---|---|---|"]
    for i, (theme, r) in enumerate(summary.iterrows(), 1):
        md.append(f"| {i} | {theme} | {int(r.reviews)} | {r.share_of_all_reviews:.0%} | {r.avg_severity:.1f} | {r.priority_score:.0f} |")
    md.append("\n## Example reviews for the top issues\n")
    for theme in summary.head(3).index:
        md.append(f"**{theme}**\n")
        ex = issues[issues["theme"] == theme].sort_values("severity", ascending=False).head(3)
        for r in ex.itertuples():
            md.append(f"- ({r.category}, severity {r.severity}) \"{r.review[:220]}\"")
        md.append("")
    (OUT_DIR / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:14]))
    print("\nSaved outputs/summary.md, issue_summary.csv, issues_by_category.csv, top_issues.png, issues_by_category.png")


if __name__ == "__main__":
    main()
