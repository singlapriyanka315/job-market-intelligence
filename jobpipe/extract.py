"""Silver enrichment: LLM extracts structured fields from job descriptions.

Design points worth knowing:
- Output is constrained: skills must come from the shared vocabulary, enums are validated,
  anything else is dropped - so bad LLM output can't pollute the gold tables.
- Jobs are batched (JOBS_PER_LLM_CALL per request) to amortise the prompt.
- A token-bucket limiter keeps us under the free tier's tokens-per-minute cap,
  using the token counts Groq reports back.
- Cached by content_hash + PROMPT_VERSION: unchanged jobs are never sent twice.
"""

import json
import time
from collections import deque

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from config import (
    DESCRIPTION_CHARS_FOR_LLM,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_TOKENS_PER_MINUTE,
    JOBS_PER_LLM_CALL,
    PROMPT_VERSION,
)
from jobpipe import db
from jobpipe.skills import SKILLS, canonicalise
from jobpipe.transform import llm_excerpt

ROLE_FAMILIES = ["data_engineer", "data_scientist", "data_analyst", "ml_engineer", "backend", "frontend",
                 "fullstack", "mobile", "devops_sre", "qa", "security", "it_support", "engineering_manager", "other"]
SENIORITY = ["intern", "junior", "mid", "senior", "lead", "unknown"]
REMOTE = ["remote", "hybrid", "onsite", "unknown"]

PROMPT = """You extract structured data from job postings. For EACH job return:
- role_family: one of {roles}
- seniority: one of {seniority} (from title and required experience; "unknown" if not stated)
- years_experience_min: minimum years of experience required as an integer, or null if not stated
- remote_policy: one of {remote}
- skills: technical skills the job REQUIRES or PREFERS, using ONLY names from this list (exact spelling):
  {skills}
  Do not include skills only mentioned about the company's product, benefits or other teams.

Jobs:
{jobs}

Reply as JSON: {{"jobs": [{{"id": <id>, "role_family": "...", "seniority": "...", "years_experience_min": null, "remote_policy": "...", "skills": ["..."]}}]}}"""


MAX_WAIT_SECONDS = 120  # a longer retry-after means the daily quota is used up


class QuotaExhausted(Exception):
    """The provider asked us to wait longer than MAX_WAIT_SECONDS - stop LLM work for this run."""


class TokenBucket:
    """Sliding one-minute window of tokens used."""

    def __init__(self, per_minute):
        self.per_minute = per_minute
        self.used = deque()  # (timestamp, tokens)

    def wait_for(self, tokens):
        while True:
            now = time.monotonic()
            while self.used and now - self.used[0][0] > 60:
                self.used.popleft()
            in_window = sum(t for _, t in self.used)
            if in_window + tokens <= self.per_minute or not self.used:
                return
            time.sleep(max(0.5, 60 - (now - self.used[0][0])))

    def record(self, tokens):
        self.used.append((time.monotonic(), tokens))


def estimate_tokens(text):
    return len(text) // 3  # rough; English is ~4 chars/token, leave margin


def validate(row):
    """Clamp one LLM result to the allowed values. Returns a clean dict."""
    years = row.get("years_experience_min")
    try:
        years = int(years) if years is not None else None
        if years is not None and not 0 <= years <= 30:
            years = None
    except (TypeError, ValueError):
        years = None
    pick = lambda v, allowed, default: v if v in allowed else default
    return {
        "role_family": pick(row.get("role_family"), ROLE_FAMILIES, "other"),
        "seniority": pick(row.get("seniority"), SENIORITY, "unknown"),
        "years_experience_min": years,
        "remote_policy": pick(row.get("remote_policy"), REMOTE, "unknown"),
        "skills": canonicalise(row.get("skills") or []),
    }


class Extractor:
    def __init__(self, model=GROQ_MODEL):
        self.model = model
        self.client = Groq(api_key=GROQ_API_KEY, max_retries=0) if GROQ_API_KEY else None
        self.bucket = TokenBucket(GROQ_TOKENS_PER_MINUTE)
        self.tokens_used = 0
        self.calls = 0

    @property
    def enabled(self):
        return self.client is not None

    def _prompt(self, jobs):
        blocks = []
        for j in jobs:
            desc = llm_excerpt(j["description"], DESCRIPTION_CHARS_FOR_LLM)
            blocks.append(f"<job id={j['job_id']}>\nTitle: {j['title']}\nCompany: {j['company']}\n"
                          f"Location: {j['location']}\n{desc}\n</job>")
        return PROMPT.format(roles=", ".join(ROLE_FAMILIES), seniority=", ".join(SENIORITY),
                             remote=", ".join(REMOTE), skills=", ".join(SKILLS), jobs="\n\n".join(blocks))

    def extract_batch(self, jobs, retries=4):
        """Returns {job_id: clean_fields} for the jobs the model answered."""
        prompt = self._prompt(jobs)
        expected = estimate_tokens(prompt) + 120 * len(jobs)
        extra = {"reasoning_effort": "low", "include_reasoning": False} if "gpt-oss" in self.model else {}
        for attempt in range(retries):
            self.bucket.wait_for(expected)
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0,
                    response_format={"type": "json_object"}, **extra)
                used = r.usage.total_tokens if r.usage else expected
                self.bucket.record(used)
                self.tokens_used += used
                self.calls += 1
                rows = json.loads(r.choices[0].message.content or "{}").get("jobs", [])
                wanted = {j["job_id"] for j in jobs}
                return {int(row["id"]): validate(row) for row in rows
                        if str(row.get("id", "")).isdigit() and int(row["id"]) in wanted}
            except (json.JSONDecodeError, KeyError, TypeError):
                self.bucket.record(expected)
                print("   invalid JSON from LLM, retrying")
            except RateLimitError as e:
                self.bucket.record(GROQ_TOKENS_PER_MINUTE)  # treat the window as full
                wait = float(e.response.headers.get("retry-after", 20)) if e.response is not None else 20
                if wait > MAX_WAIT_SECONDS:
                    raise QuotaExhausted(f"provider asked to wait {wait:.0f}s") from e
                print(f"   rate limited, waiting {wait:.0f}s")
                time.sleep(wait)
            except APIConnectionError:
                time.sleep(5 * (attempt + 1))
            except APIStatusError as e:
                print(f"   LLM error {e.status_code}: {e.message}")
                return {}
        return {}


def run(conn, limit):
    """
    Extract up to `limit` jobs. Returns (extracted, missing, stopped_reason).
    If the daily quota runs out, stops cleanly: what was extracted is kept, the rest of the
    pipeline still runs, and the remaining jobs are picked up by the next run.
    """
    ex = Extractor()
    if not ex.enabled:
        print("   LLM off (no GROQ_API_KEY) - gold layer will use keyword skills")
        return 0, 0, "no API key"
    todo = db.jobs_needing_extraction(conn, PROMPT_VERSION, limit)
    done = missing = 0
    for i in range(0, len(todo), JOBS_PER_LLM_CALL):
        batch = todo[i:i + JOBS_PER_LLM_CALL]
        try:
            results = ex.extract_batch(batch)
        except QuotaExhausted as e:
            print(f"   ! LLM quota exhausted ({e}) - stopping extraction; {len(todo) - done} jobs left for next run")
            return done, len(todo) - done, "quota exhausted"
        for j in batch:
            if j["job_id"] in results:
                db.save_extraction(conn, j["job_id"], j["content_hash"], PROMPT_VERSION, ex.model, results[j["job_id"]])
                done += 1
            else:
                missing += 1
        conn.commit()
        print(f"   extracted {done}/{len(todo)}  ({ex.tokens_used:,} tokens, {ex.calls} calls)")
    return done, missing, None
