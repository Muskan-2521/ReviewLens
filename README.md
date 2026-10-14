# Data

| File | What it is |
|---|---|
| `demo_reviews_SYNTHETIC.csv` | 40 short reviews **written by hand for testing the code only**. They are not real customer reviews. Never report results from this file. |
| `raw/amazon_reviews.csv` | The real dataset. Created by `python scripts/download_data.py` (git-ignored because it is large). |
| `reviews_sample.csv` | Balanced sample drawn from the real data by `src/prepare_data.py`. |
| `gold_template.csv` | 50 reviews from the sample, with empty label columns for you to fill in by hand. |
| `gold_labels.csv` | Your filled-in version of the template (the ground truth used for evaluation). |

## Real dataset

"Amazon Customer Review Data", Zenodo record 3549704 (Dublin Business School / Dublin City University):
https://zenodo.org/record/3549704

About 60,000 reviews from 2013 to 2019 across Mobiles, Smart TV, Books, Mobile Accessories and Refrigerator.
Check the licence and citation requirements on that page before publishing, and cite it in your README.

If `scripts/download_data.py` fails, download the CSV from that page and save it as `data/raw/amazon_reviews.csv`.
The loader finds columns by common names (`Review_Text`, `Review_Header`, `Rating`, `Category`, `Unique_ID`), and
raises a clear error listing the columns it found if none match. If your file uses other names, add them to
`CANDIDATES` in `src/prepare_data.py`.

### Alternatives
Any CSV with a review-text column works, for example Flipkart or other Amazon review exports from Kaggle.
