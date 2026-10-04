"""Job sources. Each fetcher returns a list of (source_job_id, payload_dict) exactly as the API sent them.

Terms: Remotive, Arbeitnow and RemoteOK ask for attribution - the README and dashboard link back to them.
"""

import requests

from config import ADZUNA_APP_ID, ADZUNA_APP_KEY, ADZUNA_QUERIES, USER_AGENT

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def _get(url, **kw):
    r = session.get(url, timeout=45, **kw)
    r.raise_for_status()
    return r.json()


def remotive():
    jobs = {}
    for category in ["software-dev", "data", "devops", "ai-ml", "qa"]:  # Remotive asks for few requests
        for j in _get("https://remotive.com/api/remote-jobs", params={"category": category}).get("jobs", []):
            jobs[str(j["id"])] = j
    return list(jobs.items())


def arbeitnow(pages=3):
    out = []
    for page in range(1, pages + 1):
        data = _get("https://www.arbeitnow.com/api/job-board-api", params={"page": page})
        out += [(j["slug"], j) for j in data.get("data", [])]
        if not data.get("links", {}).get("next"):
            break
    return out


def remoteok():
    data = _get("https://remoteok.com/api")
    return [(str(j["id"]), j) for j in data if isinstance(j, dict) and "id" in j]  # item 0 is the legal notice


def adzuna_in(pages_per_query=2):
    """Indian jobs with salaries. Needs a free key from developer.adzuna.com (skipped when not set)."""
    if not (ADZUNA_APP_ID and ADZUNA_APP_KEY):
        return None
    out = {}
    for q in ADZUNA_QUERIES:
        for page in range(1, pages_per_query + 1):
            data = _get(f"https://api.adzuna.com/v1/api/jobs/in/search/{page}", params={
                "app_id": ADZUNA_APP_ID, "app_key": ADZUNA_APP_KEY, "what": q,
                "results_per_page": 50, "max_days_old": 30, "content-type": "application/json",
            })
            for j in data.get("results", []):
                out[str(j["id"])] = j
    return list(out.items())


FETCHERS = {"adzuna_in": adzuna_in, "remotive": remotive, "arbeitnow": arbeitnow, "remoteok": remoteok}
