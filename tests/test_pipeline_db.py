"""Integration test: fixtures -> bronze -> silver -> checks -> gold on a real PostgreSQL.

Runs when TEST_DATABASE_URL is set (CI starts a Postgres service; locally: createdb jobs_test).
The database is wiped at the start, so never point this at real data.
"""

import json
import os
from pathlib import Path

import pytest

from jobpipe import db, quality

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
FIXTURES = Path(__file__).parent / "fixtures"
ID_FIELD = {"remotive": "id", "arbeitnow": "slug", "remoteok": "id", "adzuna_in": "id"}


@pytest.fixture()
def conn():
    c = db.connect(URL)
    c.execute("DROP SCHEMA IF EXISTS gold, silver, bronze, ops CASCADE")
    c.commit()
    db.migrate(c)
    yield c
    c.close()


def load_fixtures(conn, run_id):
    for source, id_field in ID_FIELD.items():
        rows = [(str(p[id_field]), p) for p in json.loads((FIXTURES / f"{source}.json").read_text())]
        db.upsert_raw(conn, run_id, source, rows)


def test_end_to_end_without_llm(conn):
    from pipeline import build_silver

    run_id = db.start_run(conn)
    load_fixtures(conn, run_id)
    stats = {}
    build_silver(conn, stats)
    assert stats["silver"]["parse_errors"] == 0
    assert stats["silver"]["rows"] == conn.execute("SELECT count(*) AS n FROM bronze.raw_jobs").fetchone()["n"]

    results, blocking = quality.run_checks(conn, run_id)
    # fixtures are old snapshots, so only freshness may fail here
    assert set(blocking) <= {"source_fresh"}

    db.refresh_gold(conn)
    gold = conn.execute("SELECT * FROM gold.jobs_enriched").fetchall()
    assert gold, "tech jobs should reach gold"
    assert all(not r["llm_enriched"] for r in gold)  # no LLM in tests: keyword skills are used
    assert conn.execute("SELECT count(*) AS n FROM gold.skill_demand WHERE role_family = 'all'").fetchone()["n"] > 0


def test_reingest_is_idempotent(conn):
    run_id = db.start_run(conn)
    load_fixtures(conn, run_id)
    before = conn.execute("SELECT count(*) AS n FROM bronze.raw_jobs").fetchone()["n"]
    load_fixtures(conn, run_id)
    after = conn.execute("SELECT count(*) AS n FROM bronze.raw_jobs").fetchone()["n"]
    assert before == after


def test_changed_payload_is_detected(conn):
    run_id = db.start_run(conn)
    payload = json.loads((FIXTURES / "remotive.json").read_text())[0]
    assert db.upsert_raw(conn, run_id, "remotive", [("1", payload)]) == (1, 0, 0)
    assert db.upsert_raw(conn, run_id, "remotive", [("1", payload)]) == (0, 0, 1)
    assert db.upsert_raw(conn, run_id, "remotive", [("1", {**payload, "title": "Edited"})]) == (0, 1, 0)


def test_duplicates_across_sources(conn):
    from pipeline import build_silver

    run_id = db.start_run(conn)
    job = json.loads((FIXTURES / "remotive.json").read_text())[0]
    same_job_elsewhere = {"slug": "dup", "company_name": job["company_name"] + " Inc", "title": job["title"],
                          "description": "x", "remote": True, "url": "u", "tags": [], "job_types": [],
                          "location": "Berlin", "created_at": 1790000000}
    db.upsert_raw(conn, run_id, "remotive", [(str(job["id"]), job)])
    db.upsert_raw(conn, run_id, "arbeitnow", [("dup", same_job_elsewhere)])
    build_silver(conn, {})
    rows = conn.execute("SELECT source, is_duplicate FROM silver.jobs ORDER BY source").fetchall()
    assert [(r["source"], r["is_duplicate"]) for r in rows] == [("arbeitnow", True), ("remotive", False)]
