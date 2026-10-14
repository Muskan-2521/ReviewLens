# Review Insights with GenAI

Turn thousands of unstructured product reviews into a **ranked list of business issues** using a large language model (LLM),
and **measure how accurate the AI actually is** against hand-labelled data.

> **Business question:** Which customer problems (product defects, delivery, packaging, after-sales service, price) hurt
> customers most, and what should an operations team fix first?

## Why this project

Reading reviews by hand does not scale. An LLM can tag each review with a theme, sentiment and severity in seconds, but
AI output is only useful if you can show it is **reliable**. This project therefore includes:

1. a simple **keyword baseline** (the LLM must beat this to be worth its cost),
2. a **50-review hand-labelled gold set** to measure accuracy,
3. a **priority score** (frequency x severity) that turns tags into decisions,
4. a one-page **business note** with recommendations.

## Pipeline

```
raw reviews CSV
   |  prepare_data.py   clean, de-duplicate, balanced sample + hand-labelling template
   v
reviews_sample.csv ---> classify_llm.py   LLM returns JSON {theme, sentiment, severity, reason}
   |                        (cached, retries, resumable)
   |---> baseline.py       keyword rules for comparison
   v
evaluate.py   accuracy / macro-F1 / confusion matrix vs YOUR hand labels
analyze.py    ranked issues, charts, example quotes, summary.md
   v
docs/business_note_template.md   one-page recommendations
```

## Taxonomy

| Theme | Meaning |
|---|---|
| `product_defect` | stopped working, faulty, dead on arrival |
| `performance_quality` | works but poor: battery, picture, sound, speed, cooling |
| `delivery_logistics` | late, wrong or missing item |
| `packaging_damage` | arrived damaged or badly packed |
| `after_sales_service` | warranty, repair, installation, support, refunds |
| `price_value` | overpriced or not worth it |
| `usability_design` | hard to use, bulky, confusing |
| `no_issue` / `other` | no complaint / fits none of the above |

Severity: 0 none, 1 minor, 2 significant, 3 critical (unusable, safety risk, money lost).

## Quick start

```bash
git clone <your-repo-url> && cd review-insights-genai
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                                # offline tests, no API key needed

# 1. Get the real data (see data/README.md for source and licence)
python scripts/download_data.py

# 2. Clean, sample 800 reviews, create a 50-row hand-labelling template
python -m src.prepare_data --input data/raw/amazon_reviews.csv --n 800

# 3. Hand-label data/gold_template.csv (see docs/labeling_guide.md)
#    and save it as data/gold_labels.csv  -- do this BEFORE looking at model output

# 4. Classify with an LLM (choose ONE provider, see "Choosing an LLM provider" below)
export OPENAI_API_KEY=...                                # Windows PowerShell: $env:OPENAI_API_KEY="..."
python -m src.classify_llm --input data/reviews_sample.csv --provider openai --limit 20   # cheap trial first
python -m src.classify_llm --input data/reviews_sample.csv --provider openai              # full run

# 5. Measure accuracy and produce insights
python -m src.evaluate --gold data/gold_labels.csv --sample data/reviews_sample.csv
python -m src.analyze  --sample data/reviews_sample.csv
```

Outputs appear in `outputs/`: `evaluation.md`, `confusion_matrix.png`, `summary.md`, `top_issues.png`,
`issues_by_category.png`, `issue_summary.csv`.

Use `--dry-run` on `classify_llm` to see the exact prompt without calling any API.

### Choosing an LLM provider

The classifier talks to any provider with an OpenAI-style chat API, so the same code runs on all of these:

| Provider | `--provider` | Setup | Cost |
|---|---|---|---|
| OpenAI | `openai` | set `OPENAI_API_KEY` | paid |
| Google Gemini | `gemini` | set `GEMINI_API_KEY` | has a free tier (with rate limits) |
| Ollama (local open models) | `ollama` | install Ollama, run `ollama pull llama3.1` | free, runs on your machine |

Model names change over time. Defaults are set in `src/config.py`; check the provider's documentation and override with
`--model <name>` or the `LLM_MODEL` environment variable. Trying two providers and comparing their accuracy on the same
50 hand-labelled reviews makes a nice extension.

## Results

*Fill this in from your own run. Do not publish numbers you have not produced.*

| Metric (on 50 hand-labelled reviews) | Keyword baseline | LLM (provider/model: _) |
|---|---|---|
| Theme accuracy | _ | _ |
| Theme macro-F1 | _ | _ |
| Sentiment accuracy | _ | _ |

Top issues found (from `outputs/summary.md`): _

Paste `outputs/top_issues.png` and `outputs/confusion_matrix.png` here, e.g. `![Top issues](outputs/top_issues.png)`.

## Design decisions

- **Structured JSON output with validation:** every response is parsed and checked against the allowed themes. Invalid responses are retried; failures are marked `error` and excluded, never silently guessed.
- **Cache:** results are stored in `outputs/llm_cache.jsonl`, so re-running analysis costs nothing.
- **Baseline first:** if keyword rules score about the same as the LLM, the LLM is not adding value for that theme. Reporting that is a finding, not a failure.
- **Balanced sampling by category:** prevents the biggest product category from dominating the conclusions.
- **Gold labels made before seeing model output:** avoids anchoring on the model's answer.

## Limitations (be upfront about these)

- 50 labelled reviews gives a noisy accuracy estimate (about +/-12 points at 95% confidence); `evaluate.py` prints this.
- Reviews are a biased sample: unhappy customers write more, and fake or incentivised reviews exist.
- One label per review; real reviews often contain several complaints.
- Severity is the model's judgment of the text, not measured business impact.
- Data is from 2013 to 2019, so findings describe those years.
- LLM outputs can vary between model versions; re-evaluate if you change the model or the prompt (`PROMPT_VERSION` in `classify_llm.py` invalidates the cache).

## Repo structure

```
src/            config, prepare_data, baseline, classify_llm, evaluate, analyze
scripts/        download_data.py
data/           README.md (sources), synthetic demo file
docs/           labeling_guide.md, business_note_template.md
tests/          offline tests (fake LLM client)
outputs/        generated results
```

## Data source

"Amazon Customer Review Data", Zenodo record 3549704. Check its licence and cite it properly:
https://zenodo.org/record/3549704

## Author

Muskan Kumari Singh - B.Tech Civil Engineering, NIT Surathkal. MIT licence.
