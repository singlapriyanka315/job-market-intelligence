"""PostgreSQL access: connection, migrations, and the bronze/silver writes."""

import hashlib
import json
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from config import DATABASE_URL

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def connect(url=DATABASE_URL):
    return psycopg.connect(url, row_factory=dict_row)


def migrate(conn):
    """Apply the schema (idempotent) - gold views are rebuilt separately by refresh_gold()."""
    conn.execute((SQL_DIR / "001_schema.sql").read_text())
    conn.commit()


def refresh_gold(conn):
    conn.execute((SQL_DIR / "002_gold.sql").read_text())
    conn.commit()


# ---------------------------------------------------------------- runs
def start_run(conn):
    run_id = conn.execute("INSERT INTO ops.pipeline_runs DEFAULT VALUES RETURNING run_id").fetchone()["run_id"]
    conn.commit()
    return run_id


def finish_run(conn, run_id, status, stats, error=None):
    conn.rollback()  # in case the failure left a transaction open
    conn.execute(
        "UPDATE ops.pipeline_runs SET finished_at = now(), status = %s, stats = %s, error = %s WHERE run_id = %s",
        (status, Jsonb(stats), error, run_id),
    )
    conn.commit()


# ---------------------------------------------------------------- bronze
def payload_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:32]


def upsert_raw(conn, run_id, source, rows):
    """Returns (new, changed, unchanged) counts."""
    new = changed = unchanged = 0
    rows = list(dict(rows).items())  # paginated APIs can return a job twice when the list shifts
    for source_job_id, payload in rows:
        h = payload_hash(payload)
        r = conn.execute(
            """
            INSERT INTO bronze.raw_jobs (source, source_job_id, payload, payload_hash, first_run_id)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (source, source_job_id) DO UPDATE SET
                last_seen_at = now(),
                payload      = CASE WHEN bronze.raw_jobs.payload_hash <> EXCLUDED.payload_hash
                                    THEN EXCLUDED.payload ELSE bronze.raw_jobs.payload END,
                payload_hash = EXCLUDED.payload_hash
            RETURNING (xmax = 0) AS inserted,
                      (SELECT payload_hash FROM bronze.raw_jobs WHERE source = %s AND source_job_id = %s) AS old_hash
            """,
            (source, source_job_id, Jsonb(payload), h, run_id, source, source_job_id),
        ).fetchone()
        if r["inserted"]:
            new += 1
        elif r["old_hash"] != h:
            changed += 1
        else:
            unchanged += 1
    conn.commit()
    return new, changed, unchanged


def raw_rows(conn):
    return conn.execute("SELECT raw_id, source, source_job_id, payload FROM bronze.raw_jobs ORDER BY raw_id")


# ---------------------------------------------------------------- silver
SILVER_COLS = ["source", "source_job_id", "title", "company", "location", "country", "is_remote",
               "employment_type", "salary_min", "salary_max", "salary_currency", "salary_period",
               "salary_usd_year_min", "salary_usd_year_max", "posted_at", "url", "description",
               "content_hash", "dedupe_key", "is_tech", "keyword_skills"]


def upsert_silver(conn, raw_id, rec):
    cols = ", ".join(SILVER_COLS)
    params = ", ".join(f"%({c})s" for c in SILVER_COLS)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in SILVER_COLS if c not in ("source", "source_job_id"))
    conn.execute(
        f"""
        INSERT INTO silver.jobs (raw_id, {cols}) VALUES (%(raw_id)s, {params})
        ON CONFLICT (source, source_job_id) DO UPDATE SET {updates}, updated_at = now()
        """,
        {**rec, "raw_id": raw_id},
    )


SOURCE_PRIORITY = "CASE source WHEN 'adzuna_in' THEN 1 WHEN 'remotive' THEN 2 WHEN 'remoteok' THEN 3 ELSE 4 END"


def mark_duplicates(conn):
    """Same company + title on several sources (or reposted): keep one, flag the rest."""
    conn.execute(
        f"""
        WITH ranked AS (
            SELECT job_id, row_number() OVER (
                PARTITION BY dedupe_key ORDER BY {SOURCE_PRIORITY}, posted_at DESC NULLS LAST, job_id
            ) AS rn
            FROM silver.jobs
        )
        UPDATE silver.jobs j SET is_duplicate = (r.rn > 1)
        FROM ranked r WHERE r.job_id = j.job_id AND j.is_duplicate IS DISTINCT FROM (r.rn > 1)
        """
    )
    n = conn.execute("SELECT count(*) AS n FROM silver.jobs WHERE is_duplicate").fetchone()["n"]
    conn.commit()
    return n


# ---------------------------------------------------------------- extraction
def jobs_needing_extraction(conn, prompt_version, limit):
    """Tech jobs never extracted, or whose content / prompt changed since. Newest first."""
    return conn.execute(
        """
        SELECT j.job_id, j.title, j.company, j.location, j.description, j.content_hash
        FROM silver.jobs j
        LEFT JOIN silver.job_extractions e ON e.job_id = j.job_id
        WHERE j.is_tech AND NOT j.is_duplicate
          AND (e.job_id IS NULL OR e.content_hash <> j.content_hash OR e.prompt_version <> %s)
        ORDER BY j.posted_at DESC NULLS LAST
        LIMIT %s
        """,
        (prompt_version, limit),
    ).fetchall()


def save_extraction(conn, job_id, content_hash_, prompt_version, model, x):
    conn.execute(
        """
        INSERT INTO silver.job_extractions (job_id, content_hash, prompt_version, model, role_family,
                                            seniority, years_experience_min, remote_policy, skills)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (job_id) DO UPDATE SET
            content_hash = EXCLUDED.content_hash, prompt_version = EXCLUDED.prompt_version,
            model = EXCLUDED.model, role_family = EXCLUDED.role_family, seniority = EXCLUDED.seniority,
            years_experience_min = EXCLUDED.years_experience_min, remote_policy = EXCLUDED.remote_policy,
            skills = EXCLUDED.skills, extracted_at = now()
        """,
        (job_id, content_hash_, prompt_version, model, x["role_family"], x["seniority"],
         x["years_experience_min"], x["remote_policy"], x["skills"]),
    )
