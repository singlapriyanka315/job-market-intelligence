# Job Market Intelligence

An end-to-end data pipeline that answers three questions for a job seeker:
**which skills do tech employers actually ask for, what do they pay, and which jobs fit my resume?**

It pulls postings from public job APIs, lands them raw in PostgreSQL, cleans and de-duplicates them, uses an LLM to
extract structured fields (skills, role, seniority) from free-text descriptions, checks data quality before
publishing, and serves the results in a Streamlit dashboard with semantic job search and resume matching.

![CI](https://github.com/singlapriyanka315/job-market-intelligence/actions/workflows/ci.yml/badge.svg)

## Architecture

```
                 ┌──────────── PostgreSQL ─────────────────────────────────────────────┐
 Adzuna (India)  │  bronze.raw_jobs      silver.jobs            silver.job_extractions │
 Remotive    ───►│  raw JSON payloads ─► typed, cleaned,     ─► LLM: skills, role,     │
 Arbeitnow       │  + payload hash       de-duplicated,         seniority, remote      │
 RemoteOK        │  (change detection)   keyword skills         (cached by content)    │
                 │                              │                       │              │
                 │                              └────── DQ checks ──────┘              │
                 │                                   (errors block publish)            │
                 │                                          ▼                          │
                 │  ops.pipeline_runs        gold.jobs_enriched, skill_demand,         │
                 │  ops.dq_results           skill_demand_weekly, salary_by_role ...   │
                 └──────────────────────────────────────────┬──────────────────────────┘
                                                            ▼
                                  ChromaDB (job embeddings) ──► Streamlit dashboard
                                                                · skills in demand  · salaries
                                                                · semantic search   · resume match
                                                                · pipeline health
```

| Stage | What happens | Why it's built this way |
|---|---|---|
| **Ingest → bronze** | Raw API responses stored as JSONB with a payload hash | Silver can be rebuilt any time without re-calling APIs; edits to postings are detected |
| **Transform → silver** | HTML → text, salary parsing (`$90k–$105k`, `€60.000`, `/hour`) normalised to USD/year, country detection, mojibake repair, cross-source de-duplication | Pure functions, unit-tested against real API samples |
| **LLM enrichment** | Groq (`gpt-oss-120b`) extracts role, seniority, years of experience, remote policy and skills | See *LLM design* below |
| **Data quality** | 9 SQL checks; `error` checks block the gold refresh, `warn` checks are recorded | Bad data never reaches the dashboard; every result is kept in `ops.dq_results` |
| **Publish → gold** | Views and materialized views for skill demand, weekly trends, salaries, source coverage | The dashboard only reads gold |
| **Vectors** | Unique tech jobs embedded in ChromaDB (local MiniLM); only new/changed jobs re-embedded | Semantic search and resume matching |

## LLM design

- **Constrained output.** Skills must come from a shared ~110-skill vocabulary; role, seniority and remote policy are
  validated enums; out-of-range values are dropped. A bad LLM response can't create junk categories in gold.
- **Requirement-focused excerpts.** Long postings (especially German ones) open with company marketing and list
  requirements last. Instead of sending the first 2,000 characters, paragraphs are scored by requirement headings and
  skill mentions and the best ones are sent. On a test job this took the LLM from **0 to 9 skills found**.
- **Rate-limit aware.** The free tier allows 8,000 tokens/minute. A sliding-window token bucket, fed by the token
  counts Groq returns, paces requests; 4 jobs per call amortise the prompt (~700 tokens/job).
- **Cached.** Extraction is keyed on `content_hash` + `PROMPT_VERSION`, so unchanged jobs are never sent twice and
  changing the prompt re-extracts everything. Gold only uses extractions whose hash matches the job's *current*
  text; stale ones fall back to keyword skills until they are re-extracted.
- **Quota-aware.** The free tier also has a daily token cap. When Groq asks for a wait longer than 2 minutes, the run
  stops extracting, publishes everything else, and the remaining jobs are picked up by the next run.
- **Measured.** See *Evaluation*.

## Evaluation

`eval/` compares the LLM with a keyword-matching baseline on hand-labelled postings
(precision / recall / F1 for skills, accuracy for role and seniority):

```bash
python -m eval.sample --n 50   # reproducible random sample -> eval/labels.jsonl
python -m eval.label           # label them in the terminal (no model output shown, to avoid bias)
python -m eval.evaluate        # -> eval/results.md
```

Results: _run the three commands above; `eval/results.md` is then linked here._

## Data quality checks

| Check | Severity |
|---|---|
| Every job has a title | error |
| Salary ranges are not inverted | error |
| No postings dated in the future | error |
| Every source has a posting from the last 14 days | error |
| Annual salary (USD) between 2k and 1M | warn |
| Postings are less than a year old | warn |
| Tech jobs have a real description | warn |
| Fewer than 30% cross-source duplicates | warn |
| LLM found at least one skill for technical roles | warn |

Real issues these caught during development:
- RemoteOK sends some text double-encoded (`â\x80\x94` instead of `—`) and once put an hourly rate (`30–36`) in its
  annual salary field.
- The `extracted_skills_nonempty` warning flagged senior engineering jobs where the LLM found no skills. The cause:
  RemoteOK sends entity-escaped HTML and Arbeitnow wraps escaped HTML inside real tags, so the LLM was reading markup
  instead of requirements. `html_to_text` now parses repeatedly until no tags remain (139 Arbeitnow descriptions fixed).

## Run it

Requirements: Python 3.11+, PostgreSQL.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
createdb jobs
cp .env.example .env              # add GROQ_API_KEY (free: console.groq.com); Adzuna keys optional
.venv/bin/python pipeline.py run  # ingest -> silver -> LLM -> checks -> gold -> vectors
.venv/bin/streamlit run dashboard/app.py
```

Other commands:

```bash
.venv/bin/python pipeline.py run --skip-ingest --llm-limit 40   # rebuild from bronze, extract 40 more jobs
.venv/bin/python pipeline.py search "remote data engineer airflow aws"
.venv/bin/python pipeline.py match examples/sample_resume.txt   # or your own resume.pdf
```

Without a Groq key everything still runs; the gold layer falls back to keyword-extracted skills.
Indian postings come from Adzuna, which needs a free key from developer.adzuna.com.

## Tests and CI

```bash
createdb jobs_test
TEST_DATABASE_URL=postgresql://localhost/jobs_test .venv/bin/pytest
```

- Unit tests: salary parsing, HTML/encoding cleanup, country detection, de-dup keys, requirement excerpts,
  skill matching (incl. false positives like "you will *excel*", "*R*&D", "Node*.js*"), LLM output validation,
  evaluation metrics.
- Integration tests on a real PostgreSQL: fixtures → bronze → silver → checks → gold; idempotent re-ingest;
  edited-posting detection; cross-source duplicates.
- GitHub Actions runs `ruff` and the full suite against a Postgres 17 service on every push.

## Resume matching

Retrieve-then-rerank: each resume section is searched in ChromaDB, then candidates are re-ranked by
`0.5 × semantic similarity + 0.5 × skill coverage`, where coverage = skills of the job found in the resume ÷
(skills required + 2). The +2 stops a job that names a single skill from scoring 100%. The output lists matched and
missing skills per job, and the skills missing most often across your top matches.

## Project layout

```
pipeline.py            CLI: run / search / match; records every run in ops.pipeline_runs
config.py              settings, FX rates, rate limits, prompt version
sql/001_schema.sql     bronze / silver / ops schema
sql/002_gold.sql       gold views (rebuilt each run)
jobpipe/sources/       API clients (Adzuna, Remotive, Arbeitnow, RemoteOK)
jobpipe/transform.py   bronze -> silver (pure functions)
jobpipe/skills.py      skill vocabulary + keyword baseline
jobpipe/extract.py     LLM extraction, validation, token bucket
jobpipe/quality.py     data-quality checks
jobpipe/vectors.py     ChromaDB sync, search, resume matching
dashboard/app.py       Streamlit dashboard
eval/                  sampling, labelling and scoring of the extraction
tests/                 unit + integration tests, real API fixtures
```

## Limitations and next steps

- Arbeitnow is mostly German postings; the Indian view depends on the Adzuna key.
- MiniLM embeddings are English-centric and read ~256 tokens, so matching uses titles, skills and excerpts.
- FX rates are fixed in `config.py`; salary comparisons are indicative.
- Next: Docker Compose for one-command setup, scheduling with Prefect or Airflow, and weekly skill-trend alerts.

## Data sources

[Adzuna](https://www.adzuna.in), [Remotive](https://remotive.com), [Arbeitnow](https://www.arbeitnow.com) and
[Remote OK](https://remoteok.com) public APIs, used under their terms (attribution in the dashboard footer).
