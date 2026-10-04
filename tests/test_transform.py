import json
from pathlib import Path

import pytest

from jobpipe.transform import (
    dedupe_key,
    fix_mojibake,
    guess_country,
    html_to_text,
    llm_excerpt,
    parse_salary_text,
    to_silver,
    to_usd_year,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("text, expected", [
    ("$20k -$35k", (20000, 35000, "USD", "year")),
    ("$90 - $150 /hour", (90, 150, "USD", "hour")),
    ("$90 - 150k", (90000, 150000, "USD", "year")),
    ("€60,000", (60000, None, "EUR", "year")),
    ("60.000 €", (60000, None, "EUR", "year")),
    ("£4,000 per month", (4000, None, "GBP", "month")),
    ("INR 12,00,000", (1200000, None, "INR", "year")),
    ("", (None, None, None, None)),
    ("Competitive", (None, None, None, None)),
    ("$5 equity", (None, None, None, None)),
])
def test_parse_salary_text(text, expected):
    assert parse_salary_text(text) == expected


def test_to_usd_year():
    assert to_usd_year(50, "USD", "hour") == 104000
    assert to_usd_year(1000, "EUR", "month") == pytest.approx(13920)
    assert to_usd_year(None, "USD", "year") is None
    assert to_usd_year(100, "XYZ", "year") is None


def test_html_to_text_keeps_structure():
    text = html_to_text("<p>We use <b>Python</b> &amp; SQL.</p><ul><li>Airflow</li><li>dbt</li></ul>")
    assert "We use Python & SQL." in text
    assert "- Airflow" in text and "- dbt" in text
    assert "<" not in text


def test_fix_mojibake():
    assert fix_mojibake("Lead â\u0080\u0094 Logistics") == "Lead — Logistics"
    assert fix_mojibake("Düsseldorf") == "Düsseldorf"  # already-correct text is untouched


@pytest.mark.parametrize("location, code", [
    ("Bengaluru, Karnataka", "IN"), ("Remote - USA", "US"), ("Düsseldorf", "DE"),
    ("Worldwide", "REMOTE"), ("London, UK", "GB"),
])
def test_guess_country(location, code):
    assert guess_country(location) == code


def test_dedupe_key_ignores_legal_suffix_and_gender_tag():
    assert dedupe_key("Auxmoney GmbH", "Senior DevOps Engineer (m/f/d)") == \
        dedupe_key("auxmoney", "Senior DevOps Engineer")


def test_llm_excerpt_prefers_requirements():
    intro = "\n\n".join(f"About us paragraph {i}. We love our culture and snacks." * 3 for i in range(20))
    reqs = "Your profile:\n- 5 years Python\n- Kafka and Kubernetes"
    excerpt = llm_excerpt(intro + "\n\n" + reqs, 600)
    assert len(excerpt) <= 600
    assert "Kafka and Kubernetes" in excerpt


@pytest.mark.parametrize("source", ["remotive", "arbeitnow", "remoteok", "adzuna_in"])
def test_every_fixture_maps_to_silver(source):
    for payload in json.loads((FIXTURES / f"{source}.json").read_text()):
        rec = to_silver(source, "id-1", payload)
        assert rec["title"]
        assert rec["content_hash"] and rec["dedupe_key"]
        assert rec["salary_currency"] is None or rec["salary_min"] is not None
        if rec["salary_max"] is not None:
            assert rec["salary_min"] <= rec["salary_max"]


def test_adzuna_mapping():
    real, predicted = json.loads((FIXTURES / "adzuna_in.json").read_text())
    rec = to_silver("adzuna_in", "1", real)
    assert rec["country"] == "IN" and rec["salary_currency"] == "INR"
    assert rec["salary_usd_year_min"] == pytest.approx(2400000 / 88, rel=1e-3)
    assert rec["is_remote"] is True  # "work from home"
    assert {"Python", "Spark", "Airflow", "AWS"} <= set(rec["keyword_skills"])
    assert to_silver("adzuna_in", "2", predicted)["salary_min"] is None  # Adzuna's estimate is ignored
