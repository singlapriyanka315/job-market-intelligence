-- Job Market Intelligence - PostgreSQL schema
-- Medallion layout:  bronze (raw API payloads) -> silver (clean, typed, enriched) -> gold (analytics)
-- plus ops (pipeline runs + data-quality results). Safe to run repeatedly.

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS ops;

-- ---------------------------------------------------------------- ops
CREATE TABLE IF NOT EXISTS ops.pipeline_runs (
    run_id       SERIAL PRIMARY KEY,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    status       TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'success', 'failed')),
    stats        JSONB NOT NULL DEFAULT '{}'::jsonb,   -- rows per stage
    error        TEXT
);

CREATE TABLE IF NOT EXISTS ops.dq_results (
    run_id        INT  NOT NULL REFERENCES ops.pipeline_runs(run_id),
    check_name    TEXT NOT NULL,
    severity      TEXT NOT NULL CHECK (severity IN ('error', 'warn')),
    passed        BOOLEAN NOT NULL,
    failing_rows  INT NOT NULL,
    details       TEXT,
    checked_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, check_name)
);

-- ---------------------------------------------------------------- bronze
-- Exactly what each API returned. Never edited by hand - silver can always be rebuilt from here.
CREATE TABLE IF NOT EXISTS bronze.raw_jobs (
    raw_id         BIGSERIAL PRIMARY KEY,
    source         TEXT  NOT NULL,
    source_job_id  TEXT  NOT NULL,
    payload        JSONB NOT NULL,
    payload_hash   TEXT  NOT NULL,                 -- detects when a posting is edited
    first_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    first_run_id   INT REFERENCES ops.pipeline_runs(run_id),
    UNIQUE (source, source_job_id)
);

-- ---------------------------------------------------------------- silver
CREATE TABLE IF NOT EXISTS silver.jobs (
    job_id            BIGSERIAL PRIMARY KEY,
    raw_id            BIGINT NOT NULL UNIQUE REFERENCES bronze.raw_jobs(raw_id),
    source            TEXT NOT NULL,
    source_job_id     TEXT NOT NULL,
    title             TEXT NOT NULL,
    company           TEXT,
    location          TEXT,
    country           TEXT,                        -- ISO-ish code when known (IN, US, DE, ...) or 'REMOTE'
    is_remote         BOOLEAN,
    employment_type   TEXT,                        -- full_time / part_time / contract / internship
    salary_min        NUMERIC(14, 2),
    salary_max        NUMERIC(14, 2),
    salary_currency   TEXT,
    salary_period     TEXT CHECK (salary_period IN ('year', 'month', 'hour')),
    salary_usd_year_min NUMERIC(14, 2),            -- normalised for comparison (fixed FX in config)
    salary_usd_year_max NUMERIC(14, 2),
    posted_at         TIMESTAMPTZ,
    url               TEXT,
    description       TEXT,                        -- plain text (HTML stripped)
    content_hash      TEXT NOT NULL,               -- LLM cache key: same content = no new LLM call
    dedupe_key        TEXT NOT NULL,               -- normalised company|title, for cross-source duplicates
    is_duplicate      BOOLEAN NOT NULL DEFAULT FALSE,
    is_tech           BOOLEAN NOT NULL,            -- cheap title filter: only these go to the LLM
    keyword_skills    TEXT[] NOT NULL DEFAULT '{}',-- baseline extractor (dictionary match)
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, source_job_id)
);
CREATE INDEX IF NOT EXISTS jobs_dedupe_idx ON silver.jobs (dedupe_key);
CREATE INDEX IF NOT EXISTS jobs_posted_idx ON silver.jobs (posted_at);

CREATE TABLE IF NOT EXISTS silver.job_extractions (
    job_id               BIGINT PRIMARY KEY REFERENCES silver.jobs(job_id) ON DELETE CASCADE,
    content_hash         TEXT NOT NULL,
    prompt_version       TEXT NOT NULL,
    model                TEXT NOT NULL,
    role_family          TEXT,
    seniority            TEXT,
    years_experience_min INT,
    remote_policy        TEXT,
    skills               TEXT[] NOT NULL DEFAULT '{}',
    extracted_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
