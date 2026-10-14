"""Keyword-rule baseline. The LLM has to beat this to justify its cost - report both."""
import re

import pandas as pd

RULES = {
    "packaging_damage": r"\b(packag\w*|box\w* (was )?(damaged|crushed|torn|broken)|arrived (damaged|broken|cracked)|dent(ed)?|scratched on arrival)\b",
    "delivery_logistics": r"\b(deliver\w*|courier|late|delay\w*|wrong (item|product|colou?r)|missing|never (came|arrived)|not received|shipping)\b",
    "after_sales_service": r"\b(warranty|service (center|centre)|customer (care|service|support)|repair\w*|install\w*|technician|refund|return(ed)? (policy|request)|replacement|no response)\b",
    "product_defect": r"\b(stopped working|not working|doesn'?t work|does not work|defect\w*|faulty|dead on arrival|\bdead\b|broke\w*|malfunction\w*|stopped charging|not turning on|won'?t turn on)\b",
    "performance_quality": r"\b(battery|lag\w*|slow|hang\w*|heat\w*|overheat\w*|noise|noisy|cooling|picture quality|sound quality|poor quality|blurry|display)\b",
    "price_value": r"\b(overpriced|expensive|waste of money|not worth|value for money|worth (the )?(price|money)|costly)\b",
    "usability_design": r"\b(heavy|bulky|difficult to use|hard to use|complicated|uncomfortable|design|small font|confusing)\b",
}
POSITIVE = r"\b(good|great|excellent|love\w*|perfect|amazing|nice|awesome|best|happy|satisfied|recommend\w*|superb)\b"
NEGATIVE = r"\b(bad|poor|worst|terrible|disappoint\w*|waste|useless|horrible|pathetic|hate|regret|never buy|don'?t buy|cheap quality)\b"
CRITICAL = r"\b(unusable|stopped working|dead|fire|burn\w*|shock|refund|fraud|fake|scam|waste of money)\b"


def classify_one(text: str, rating=None) -> dict:
    t = text.lower()
    # priority order matters: first match wins (specific, actionable themes first)
    theme = next((name for name, pat in RULES.items() if re.search(pat, t)), None)

    neg = len(re.findall(NEGATIVE, t))
    pos = len(re.findall(POSITIVE, t))
    if pd.notna(rating):
        sentiment = "negative" if rating <= 2 else "neutral" if rating == 3 else "positive"
    else:
        sentiment = "negative" if neg > pos else "positive" if pos > neg else "neutral"

    if theme is None:
        theme = "other" if sentiment == "negative" else "no_issue"
    if sentiment == "positive" and theme != "no_issue" and neg == 0:
        theme = "no_issue"  # praise that merely mentions "delivery" or "battery"

    if theme == "no_issue":
        severity = 0
    elif re.search(CRITICAL, t):
        severity = 3
    else:
        severity = 2 if sentiment == "negative" else 1
    return {"theme": theme, "sentiment": sentiment, "severity": severity}


def classify_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = [classify_one(r.review, r.rating) for r in df.itertuples()]
    return pd.concat([df[["review_id"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
