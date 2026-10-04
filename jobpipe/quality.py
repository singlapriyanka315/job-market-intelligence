"""Data-quality checks, run after silver is built and before gold is published.

'error' checks block the gold refresh (bad data never reaches the dashboard);
'warn' checks are recorded in ops.dq_results and shown in the dashboard.
Each check is SQL that returns the number of failing rows.
"""

from dataclasses import dataclass


@dataclass
class Check:
    name: str
    severity: str
    sql: str
    description: str
    threshold: int = 0  # max failing rows allowed (for ratio checks the SQL returns 0/1)


CHECKS = [
    Check("title_not_empty", "error",
          "SELECT count(*) FROM silver.jobs WHERE trim(title) = ''",
          "Every job has a title"),
    Check("salary_min_le_max", "error",
          "SELECT count(*) FROM silver.jobs WHERE salary_max IS NOT NULL AND salary_min > salary_max",
          "Salary ranges are not inverted"),
    Check("salary_usd_plausible", "warn",
          "SELECT count(*) FROM silver.jobs WHERE salary_usd_year_min IS NOT NULL "
          "AND (salary_usd_year_min < 2000 OR salary_usd_year_min > 1000000)",
          "Annual salary (USD) between 2k and 1M"),
    Check("posted_at_not_future", "error",
          "SELECT count(*) FROM silver.jobs WHERE posted_at > now() + interval '1 day'",
          "No postings dated in the future"),
    Check("posted_within_year", "warn",
          "SELECT count(*) FROM silver.jobs WHERE posted_at < now() - interval '365 days'",
          "Postings are less than a year old"),
    Check("tech_description_present", "warn",
          "SELECT count(*) FROM silver.jobs WHERE is_tech AND length(coalesce(description, '')) < 100",
          "Tech jobs have a real description (100+ chars)"),
    Check("source_fresh", "error",
          "SELECT count(*) FROM (SELECT source FROM silver.jobs GROUP BY source "
          "HAVING max(posted_at) < now() - interval '14 days') s",
          "Every source has a posting from the last 14 days"),
    Check("duplicate_rate_under_30pct", "warn",
          "SELECT (avg(is_duplicate::int) > 0.30)::int FROM silver.jobs",
          "Fewer than 30% of rows are cross-source duplicates"),
    Check("extracted_skills_nonempty", "warn",
          "SELECT count(*) FROM silver.job_extractions e JOIN silver.jobs j USING (job_id) "
          "WHERE e.content_hash = j.content_hash "
          "AND cardinality(e.skills) = 0 AND e.role_family NOT IN ('other', 'it_support', 'engineering_manager')",
          "LLM found at least one skill for technical roles"),
]


def run_checks(conn, run_id):
    """Returns (results, blocking_failures)."""
    results, blocking = [], []
    for c in CHECKS:
        failing = list(conn.execute(c.sql).fetchone().values())[0] or 0
        passed = failing <= c.threshold
        conn.execute(
            """
            INSERT INTO ops.dq_results (run_id, check_name, severity, passed, failing_rows, details)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, check_name) DO UPDATE SET passed = EXCLUDED.passed,
                failing_rows = EXCLUDED.failing_rows, checked_at = now()
            """,
            (run_id, c.name, c.severity, passed, failing, c.description),
        )
        results.append((c, passed, failing))
        if not passed and c.severity == "error":
            blocking.append(c.name)
    conn.commit()
    return results, blocking
