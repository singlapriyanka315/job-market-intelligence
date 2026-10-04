"""
Hand-label jobs in eval/labels.jsonl, one at a time, in the terminal.

    python -m eval.label

You see the job text and type the skills it requires (comma-separated, from the vocabulary;
aliases like 'pyspark' or 'k8s' are accepted), then role family and seniority.
No model output is shown, so the labels aren't biased by either extractor.
Progress is saved after every job; quit any time with Ctrl+C and resume later.
"""

import json
import textwrap
from pathlib import Path

from jobpipe.extract import ROLE_FAMILIES, SENIORITY
from jobpipe.skills import SKILLS, canonicalise

PATH = Path(__file__).parent / "labels.jsonl"


def save(records):
    PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def ask_choice(prompt, options):
    while True:
        v = input(f"{prompt} [{'/'.join(options)}]: ").strip()
        if v in options:
            return v
        print("  not one of the options")


def main():
    records = [json.loads(line) for line in PATH.read_text().splitlines() if line.strip()]
    todo = [r for r in records if not r["labelled"]]
    print(f"{len(records) - len(todo)}/{len(records)} labelled.\nVocabulary: {', '.join(SKILLS)}\n")
    for r in todo:
        print("=" * 100)
        print(f"{r['title']} — {r['company']} ({r['location']})\n")
        print(textwrap.fill(r["text"], 100, replace_whitespace=False))
        print()
        while True:
            raw = [s for s in input("Skills (comma-separated, blank = none): ").split(",") if s.strip()]
            skills = canonicalise(raw)
            unknown = [s.strip() for s in raw if not canonicalise([s])]
            if unknown:
                print(f"  not in vocabulary, ignored: {unknown}")
            if input(f"  -> {skills}  ok? [Y/n] ").strip().lower() != "n":
                break
        r["skills"] = skills
        r["role_family"] = ask_choice("Role family", ROLE_FAMILIES)
        r["seniority"] = ask_choice("Seniority", SENIORITY)
        r["labelled"] = True
        save(records)
    print("All labelled. Run:  python -m eval.evaluate")


if __name__ == "__main__":
    main()
