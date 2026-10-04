-- Gold layer: analytics-ready views. Recreated on every run (cheap at this data size).

DROP VIEW IF EXISTS gold.jobs_enriched CASCADE;
DROP MATERIALIZED VIEW IF EXISTS gold.skill_demand CASCADE;
DROP MATERIALIZED VIEW IF EXISTS gold.skill_demand_weekly CASCADE;
DROP MATERIALIZED VIEW IF EXISTS gold.salary_by_role CASCADE;
DROP MATERIALIZED VIEW IF EXISTS gold.source_summary CASCADE;

-- One row per unique tech job, with LLM fields when available (LLM skills, else keyword skills).
-- Only extractions made from the job's *current* text count: if a posting changed since it was
-- extracted (or the cleaning changed), the stale result is ignored until it is re-extracted.
CREATE VIEW gold.jobs_enriched AS
SELECT j.job_id, j.source, j.title, j.company, j.location, j.country, j.is_remote,
       j.employment_type, j.salary_usd_year_min, j.salary_usd_year_max, j.posted_at, j.url,
       coalesce(e.role_family, 'unknown')  AS role_family,
       coalesce(e.seniority, 'unknown')    AS seniority,
       e.years_experience_min,
       coalesce(e.remote_policy, CASE WHEN j.is_remote THEN 'remote' ELSE 'unknown' END) AS remote_policy,
       CASE WHEN e.job_id IS NOT NULL THEN e.skills ELSE j.keyword_skills END AS skills,
       (e.job_id IS NOT NULL) AS llm_enriched,
       date_trunc('week', coalesce(j.posted_at, j.updated_at))::date AS week
FROM silver.jobs j
LEFT JOIN silver.job_extractions e ON e.job_id = j.job_id AND e.content_hash = j.content_hash
WHERE j.is_tech AND NOT j.is_duplicate;

-- How often each skill is asked for, overall and per role family
CREATE MATERIALIZED VIEW gold.skill_demand AS
WITH totals AS (
    SELECT role_family, count(*) AS jobs FROM gold.jobs_enriched GROUP BY ROLLUP (role_family)
)
SELECT coalesce(s.role_family, 'all') AS role_family, s.skill, s.jobs,
       round(100.0 * s.jobs / t.jobs, 1) AS pct_of_jobs
FROM (
    SELECT role_family, skill, count(DISTINCT job_id) AS jobs
    FROM gold.jobs_enriched, unnest(skills) AS skill
    GROUP BY ROLLUP (role_family), skill
) s
JOIN totals t ON t.role_family IS NOT DISTINCT FROM s.role_family
WHERE s.skill IS NOT NULL;

CREATE MATERIALIZED VIEW gold.skill_demand_weekly AS
WITH totals AS (SELECT week, count(*) AS jobs FROM gold.jobs_enriched GROUP BY week)
SELECT e.week, skill, count(DISTINCT e.job_id) AS jobs,
       round(100.0 * count(DISTINCT e.job_id) / t.jobs, 1) AS pct_of_jobs
FROM gold.jobs_enriched e
CROSS JOIN LATERAL unnest(e.skills) AS skill
JOIN totals t ON t.week = e.week
GROUP BY e.week, skill, t.jobs;

-- Salary (normalised to USD/year) by role and seniority; only rows with a salary
CREATE MATERIALIZED VIEW gold.salary_by_role AS
SELECT role_family, seniority, count(*) AS jobs_with_salary,
       round(percentile_cont(0.5) WITHIN GROUP (ORDER BY (salary_usd_year_min + coalesce(salary_usd_year_max, salary_usd_year_min)) / 2)::numeric, 0) AS median_usd_year,
       round(min(salary_usd_year_min), 0) AS min_usd_year,
       round(max(coalesce(salary_usd_year_max, salary_usd_year_min)), 0) AS max_usd_year
FROM gold.jobs_enriched
WHERE salary_usd_year_min IS NOT NULL
GROUP BY role_family, seniority;

-- Coverage per source: how much of each source made it through each stage
CREATE MATERIALIZED VIEW gold.source_summary AS
SELECT j.source,
       count(*)                                         AS jobs,
       count(*) FILTER (WHERE j.is_tech)                AS tech_jobs,
       count(*) FILTER (WHERE j.is_duplicate)           AS duplicates,
       count(e.job_id)                                  AS llm_enriched,
       count(*) FILTER (WHERE j.salary_usd_year_min IS NOT NULL) AS with_salary,
       max(j.posted_at)                                 AS latest_posting
FROM silver.jobs j
LEFT JOIN silver.job_extractions e ON e.job_id = j.job_id AND e.content_hash = j.content_hash
GROUP BY j.source;
