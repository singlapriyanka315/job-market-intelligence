"""
Job Market Intelligence pipeline.

    python pipeline.py run                    # full run: ingest -> silver -> LLM -> checks -> gold -> vectors
    python pipeline.py run --llm-limit 40     # cap LLM extraction this run
    python pipeline.py run --skip-ingest      # rebuild from bronze without calling the job APIs
    python pipeline.py match resume.pdf       # rank jobs against your resume, with skill gaps
    python pipeline.py search "airflow data engineer remote"

Every run is recorded in ops.pipeline_runs; data-quality results in ops.dq_results.
"""

import argparse
import sys
import time
import traceback

from config import MAX_LLM_JOBS_PER_RUN
from jobpipe import db, extract, quality, vectors
from jobpipe.sources import FETCHERS
from jobpipe.transform import to_silver


def stage(name):
    print(f"\n▶ {name}")
    return time.monotonic()


def ingest(conn, run_id, stats):
    for source, fetch in FETCHERS.items():
        try:
            rows = fetch()
        except Exception as e:  # one source down shouldn't stop the others
            print(f"   ! {source} failed: {e}")
            stats[f"bronze.{source}"] = "failed"
            continue
        if rows is None:
            print(f"   - {source}: skipped (no API key)")
            continue
        new, changed, same = db.upsert_raw(conn, run_id, source, rows)
        stats[f"bronze.{source}"] = {"fetched": len(rows), "new": new, "changed": changed}
        print(f"   {source}: {len(rows)} fetched · {new} new · {changed} changed · {same} unchanged")


def build_silver(conn, stats):
    ok = bad = 0
    for r in db.raw_rows(conn).fetchall():
        try:
            db.upsert_silver(conn, r["raw_id"], to_silver(r["source"], r["source_job_id"], r["payload"]))
            ok += 1
        except Exception as e:  # a malformed payload is logged, not fatal
            bad += 1
            print(f"   ! could not parse {r['source']}/{r['source_job_id']}: {e}")
            conn.rollback()
    conn.commit()
    dupes = db.mark_duplicates(conn)
    tech = conn.execute("SELECT count(*) AS n FROM silver.jobs WHERE is_tech AND NOT is_duplicate").fetchone()["n"]
    stats["silver"] = {"rows": ok, "parse_errors": bad, "duplicates": dupes, "unique_tech_jobs": tech}
    print(f"   {ok} rows · {bad} parse errors · {dupes} duplicates flagged · {tech} unique tech jobs")


def run(args):
    conn = db.connect()
    db.migrate(conn)
    run_id = db.start_run(conn)
    stats = {}
    print(f"Run #{run_id}")
    try:
        t = stage("1. Ingest → bronze")
        if args.skip_ingest:
            print("   skipped")
        else:
            ingest(conn, run_id, stats)

        stage("2. Transform → silver")
        build_silver(conn, stats)

        stage("3. LLM enrichment → silver.job_extractions")
        done, missing = extract.run(conn, args.llm_limit)
        stats["llm"] = {"extracted": done, "missing": missing}

        stage("4. Data-quality checks")
        results, blocking = quality.run_checks(conn, run_id)
        for c, passed, failing in results:
            mark = "✓" if passed else ("✗" if c.severity == "error" else "!")
            print(f"   {mark} {c.name:30} {'' if passed else f'{failing} failing'}")
        stats["dq"] = {"failed_errors": blocking,
                       "failed_warnings": [c.name for c, p, _ in results if not p and c.severity == "warn"]}
        if blocking:
            raise RuntimeError(f"blocking data-quality checks failed: {', '.join(blocking)} - gold not refreshed")

        stage("5. Publish → gold")
        db.refresh_gold(conn)
        stats["gold"] = conn.execute("SELECT count(*) AS jobs FROM gold.jobs_enriched").fetchone()
        print(f"   gold.jobs_enriched: {stats['gold']['jobs']} jobs")

        stage("6. Vectors → ChromaDB")
        added, removed = vectors.sync(conn)
        stats["vectors"] = {"embedded": added, "removed": removed}
        print(f"   {added} embedded · {removed} removed")

        db.finish_run(conn, run_id, "success", stats)
        print(f"\n✔ Run #{run_id} succeeded in {time.monotonic() - t:.0f}s")
    except Exception as e:
        db.finish_run(conn, run_id, "failed", stats, f"{e}\n{traceback.format_exc()}")
        print(f"\n✘ Run #{run_id} failed: {e}")
        sys.exit(1)


def read_resume(path):
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    return open(path, encoding="utf-8").read()


def match(args):
    conn = db.connect()
    skills, matches = vectors.match_resume(conn, read_resume(args.resume), n=args.n)
    print(f"Skills found in your resume ({len(skills)}): {', '.join(skills) or 'none'}\n")
    gaps = {}
    for i, m in enumerate(matches, 1):
        print(f"{i:2}. {m['score']:.2f}  {m['title']} — {m['company']} ({m['location']})"
              f"   [meaning {m['semantic']:.2f} · skills covered {m['coverage']:.0%}]")
        print(f"      have: {', '.join(m['matched']) or '-'}")
        print(f"      missing: {', '.join(m['missing']) or '-'}")
        for s in m["missing"]:
            gaps[s] = gaps.get(s, 0) + 1
    if gaps:
        top = sorted(gaps.items(), key=lambda x: -x[1])[:8]
        print("\nSkills to learn next (missing from most of your top matches): "
              + ", ".join(f"{s} ({n})" for s, n in top))


def search(args):
    conn = db.connect()
    hits = vectors.search(args.query, n=args.n)
    rows = {r["job_id"]: r for r in conn.execute(
        "SELECT job_id, title, company, location, skills FROM gold.jobs_enriched WHERE job_id = ANY(%s)",
        ([h for h, _ in hits],)).fetchall()}
    for jid, sim in hits:
        if jid in rows:
            r = rows[jid]
            print(f"{sim:.2f}  {r['title']} — {r['company']} ({r['location']})  [{', '.join(r['skills'][:6])}]")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--llm-limit", type=int, default=MAX_LLM_JOBS_PER_RUN)
    r.add_argument("--skip-ingest", action="store_true")
    m = sub.add_parser("match")
    m.add_argument("resume")
    m.add_argument("-n", type=int, default=15)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-n", type=int, default=10)
    args = p.parse_args()
    {"run": run, "match": match, "search": search}[args.cmd](args)


if __name__ == "__main__":
    main()
