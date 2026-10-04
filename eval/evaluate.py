"""
Compare the LLM extractor with the keyword baseline on hand-labelled jobs.

    python -m eval.evaluate                      # current GROQ_MODEL
    python -m eval.evaluate --model openai/gpt-oss-20b

Writes eval/results.md (and results.json) with precision / recall / F1 for skills,
accuracy for role family and seniority, and the most common errors of each extractor.
"""

import argparse
import json
from datetime import date
from pathlib import Path

from config import GROQ_MODEL, JOBS_PER_LLM_CALL, PROMPT_VERSION
from eval.metrics import accuracy, per_skill_errors, set_scores
from jobpipe.extract import Extractor
from jobpipe.skills import keyword_skills

HERE = Path(__file__).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=GROQ_MODEL)
    args = p.parse_args()

    records = [json.loads(line) for line in (HERE / "labels.jsonl").read_text().splitlines() if line.strip()]
    records = [r for r in records if r["labelled"]]
    if not records:
        raise SystemExit("No labelled jobs yet - run: python -m eval.sample  then  python -m eval.label")

    # Keyword baseline sees title + the same text
    kw = [keyword_skills(f"{r['title']}\n{r['text']}") for r in records]

    ex = Extractor(model=args.model)
    if not ex.enabled:
        raise SystemExit("GROQ_API_KEY is needed for the LLM side of the evaluation")
    llm = {}
    jobs = [{"job_id": i, "title": r["title"], "company": r["company"], "location": r["location"],
             "description": r["text"]} for i, r in enumerate(records)]
    for i in range(0, len(jobs), JOBS_PER_LLM_CALL):
        llm.update(ex.extract_batch(jobs[i:i + JOBS_PER_LLM_CALL]))
        print(f"  LLM {len(llm)}/{len(jobs)}")
    empty = {"skills": [], "role_family": None, "seniority": None}
    llm_out = [llm.get(i, empty) for i in range(len(records))]

    gold = [r["skills"] for r in records]
    result = {
        "date": str(date.today()), "jobs": len(records), "model": args.model, "prompt_version": PROMPT_VERSION,
        "llm_answered": len(llm), "llm_tokens": ex.tokens_used,
        "skills": {"keyword_baseline": set_scores(gold, kw), "llm": set_scores(gold, [o["skills"] for o in llm_out])},
        "role_family_accuracy": accuracy([r["role_family"] for r in records], [o["role_family"] for o in llm_out]),
        "seniority_accuracy": accuracy([r["seniority"] for r in records], [o["seniority"] for o in llm_out]),
        "errors": {"keyword_baseline": per_skill_errors(gold, kw),
                   "llm": per_skill_errors(gold, [o["skills"] for o in llm_out])},
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2))

    k, m = result["skills"]["keyword_baseline"], result["skills"]["llm"]
    fmt = lambda e: ", ".join(f"{s} ({n})" for s, n in e) or "-"
    md = [
        f"# Extraction evaluation — {result['date']}",
        f"{len(records)} hand-labelled jobs · model `{args.model}` · prompt `{PROMPT_VERSION}`\n",
        "## Skills (micro-averaged)\n",
        "| Extractor | Precision | Recall | F1 |", "|---|---:|---:|---:|",
        f"| Keyword baseline | {k['precision']:.2f} | {k['recall']:.2f} | {k['f1']:.2f} |",
        f"| LLM | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} |\n",
        "## Other fields (LLM)\n",
        f"- Role family accuracy: {result['role_family_accuracy']}",
        f"- Seniority accuracy: {result['seniority_accuracy']}\n",
        "## Most common errors\n",
        f"- Keyword baseline invents: {fmt(result['errors']['keyword_baseline']['false_positives'])}",
        f"- Keyword baseline misses: {fmt(result['errors']['keyword_baseline']['false_negatives'])}",
        f"- LLM invents: {fmt(result['errors']['llm']['false_positives'])}",
        f"- LLM misses: {fmt(result['errors']['llm']['false_negatives'])}",
    ]
    (HERE / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
