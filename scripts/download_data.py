"""Download the public Amazon customer-review CSV from Zenodo into data/raw/.

Dataset: "Amazon Customer Review Data" (Zenodo record 3549704) - ~60k reviews, 2013-2019,
categories: Mobiles, Smart TV, Books, Mobile Accessories, Refrigerator.
Check the record page for the licence and citation before you publish results:
    https://zenodo.org/record/3549704

Usage:
    python scripts/download_data.py
If this fails (network or API change), download the CSV manually from the page above and save it
as data/raw/amazon_reviews.csv.
"""
import sys
from pathlib import Path

import requests

RECORD = "3549704"
DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "amazon_reviews.csv"


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    meta = requests.get(f"https://zenodo.org/api/records/{RECORD}", timeout=30)
    meta.raise_for_status()
    files = meta.json().get("files", [])
    csvs = [f for f in files if f.get("key", "").lower().endswith(".csv")]
    if not csvs:
        sys.exit("No CSV found in the Zenodo record. Download it manually (see docstring).")
    f = max(csvs, key=lambda x: x.get("size", 0))  # the big one is the review file
    url = f["links"]["self"]
    print(f"Downloading {f['key']} ({f.get('size', 0) / 1e6:.1f} MB) ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(DEST, "wb") as out:
            for chunk in r.iter_content(chunk_size=1 << 20):
                out.write(chunk)
    print(f"Saved -> {DEST}")


if __name__ == "__main__":
    main()
