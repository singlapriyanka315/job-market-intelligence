"""
Pick a random, reproducible sample of jobs for hand-labelling.

    python -m eval.sample --n 50

Writes eval/labels.jsonl. Each record stores the job text itself, so the evaluation
does not depend on the job still being in the database and can be re-run against
any model or prompt version.
"""

import argparse
import json
import random
from pathlib import Path

from config import DESCRIPTION_CHARS_FOR_LLM
from jobpipe import db
from jobpipe.transform import llm_excerpt

OUT = Path(__file__).parent / "labels.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if OUT.exists():
        raise SystemExit(f"{OUT} already exists - delete it first if you really want a new sample")

    conn = db.connect()
    rows = conn.execute(
        "SELECT source, source_job_id, title, company, location, description FROM silver.jobs "
        "WHERE is_tech AND NOT is_duplicate AND length(description) > 200 ORDER BY source, source_job_id"
    ).fetchall()
    sample = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    with OUT.open("w") as f:
        for r in sample:
            f.write(json.dumps({
                "id": f"{r['source']}:{r['source_job_id']}",
                "title": r["title"], "company": r["company"], "location": r["location"],
                # the same excerpt the LLM sees, so both extractors and the labeller read identical text
                "text": llm_excerpt(r["description"], DESCRIPTION_CHARS_FOR_LLM),
                "labelled": False,
                "skills": [], "role_family": "", "seniority": "",
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sample)} jobs to {OUT}. Label them with:  python -m eval.label")


if __name__ == "__main__":
    main()
