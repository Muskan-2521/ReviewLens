"""Offline tests: no API key or internet needed (the LLM is replaced by a fake client)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.baseline import classify_one
from src.classify_llm import classify_dataframe, parse_response
from src.prepare_data import balanced_sample, load_and_clean


class FakeClient:
    """Mimics an OpenAI-style client.chat.completions.create for tests."""
    def __init__(self, payload):
        self.payload, self.calls = payload, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.payload))])


def test_parse_valid_json_with_fences():
    out = parse_response('```json\n{"theme": "product_defect", "sentiment": "negative", "severity": 3, "reason": "dead"}\n```')
    assert out["theme"] == "product_defect" and out["severity"] == 3


def test_parse_rejects_unknown_theme():
    with pytest.raises(ValueError):
        parse_response('{"theme": "banana", "sentiment": "negative", "severity": 1}')


def test_no_issue_forces_zero_severity():
    out = parse_response('{"theme": "no_issue", "sentiment": "positive", "severity": 2}')
    assert out["severity"] == 0


def test_baseline_basic_cases():
    assert classify_one("Phone stopped working after 3 weeks", 1)["theme"] == "product_defect"
    assert classify_one("Great phone, delivery was on time", 5)["theme"] == "no_issue"
    assert classify_one("Delivery was delayed by 10 days", 2)["theme"] == "delivery_logistics"


def test_load_clean_and_sample():
    df = load_and_clean(Path("data/demo_reviews_SYNTHETIC.csv"))
    assert {"review_id", "review", "category", "rating"} <= set(df.columns)
    s = balanced_sample(df, 10)
    assert len(s) == 10


def test_llm_classification_and_cache(tmp_path):
    df = pd.DataFrame({"review_id": ["a", "b"], "review": ["bad", "good"], "category": ["x", "y"]})
    client = FakeClient(json.dumps({"theme": "other", "sentiment": "neutral", "severity": 1, "reason": "test"}))
    cache = tmp_path / "cache.jsonl"
    first = classify_dataframe(df, client, model="m", cache_path=cache, workers=1)
    assert len(first) == 2 and client.calls == 2
    classify_dataframe(df, client, model="m", cache_path=cache, workers=1)
    assert client.calls == 2  # second run served entirely from cache


def test_failures_are_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr("src.classify_llm.time.sleep", lambda s: None)
    df = pd.DataFrame({"review_id": ["a"], "review": ["bad"], "category": ["x"]})
    bad = FakeClient("not json at all")
    cache = tmp_path / "cache.jsonl"
    out = classify_dataframe(df, bad, model="m", cache_path=cache, workers=1)
    assert out.loc[0, "theme"] == "error"
    assert not cache.exists() or cache.read_text() == ""
