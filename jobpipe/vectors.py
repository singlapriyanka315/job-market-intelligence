"""Semantic search over jobs (ChromaDB, local MiniLM embeddings) + resume matching."""

import hashlib
import re

import chromadb

from config import CHROMA_PATH
from jobpipe.skills import keyword_skills


def _collection(path=CHROMA_PATH):
    return chromadb.PersistentClient(path=path).get_or_create_collection(
        "jobs", metadata={"hnsw:space": "cosine"})


def job_document(row):
    # MiniLM reads ~256 tokens, so lead with what matters most for matching
    skills = ", ".join(row["skills"] or [])
    return f"{row['title']}. Skills: {skills}. {(row['description'] or '')[:700]}"


def _doc_hash(doc):
    return hashlib.sha256(doc.encode()).hexdigest()[:32]


def sync(conn):
    """
    Embed new/changed unique tech jobs; remove ones no longer in gold. Returns (added, removed).
    Change detection uses a hash of the embedded document itself, so a job is re-embedded when
    its text changes *or* when the LLM fills in its skills.
    """
    col = _collection()
    rows = conn.execute(
        """
        SELECT e.job_id, e.title, e.skills, e.role_family, e.seniority, e.country, e.is_remote,
               j.description, j.content_hash
        FROM gold.jobs_enriched e JOIN silver.jobs j USING (job_id)
        """
    ).fetchall()
    existing = {}
    if col.count():
        got = col.get(include=["metadatas"])
        existing = {i: m.get("doc_hash") for i, m in zip(got["ids"], got["metadatas"], strict=True)}

    docs = {r["job_id"]: job_document(r) for r in rows}
    todo = [r for r in rows if existing.get(str(r["job_id"])) != _doc_hash(docs[r["job_id"]])]
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        col.upsert(
            ids=[str(r["job_id"]) for r in chunk],
            documents=[docs[r["job_id"]] for r in chunk],
            metadatas=[{"doc_hash": _doc_hash(docs[r["job_id"]]), "role_family": r["role_family"],
                        "seniority": r["seniority"], "country": r["country"] or "",
                        "is_remote": bool(r["is_remote"])} for r in chunk],
        )
    stale = set(existing) - {str(r["job_id"]) for r in rows}
    if stale:
        col.delete(ids=list(stale))
    return len(todo), len(stale)


def search(query, n=10, where=None):
    """Free-text semantic search. Returns [(job_id, similarity)]."""
    col = _collection()
    if not col.count():
        return []
    res = col.query(query_texts=[query], n_results=min(n, col.count()), where=where or None)
    return [(int(i), round(1 - d, 3)) for i, d in zip(res["ids"][0], res["distances"][0], strict=True)]


def _chunks(text, size=700):
    """Split a resume into ~size-char chunks on paragraph/line boundaries (MiniLM truncates long input)."""
    parts, cur = [], ""
    for line in re.split(r"\n+", text):
        if len(cur) + len(line) > size and cur:
            parts.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur)
    return parts or [text]


SEMANTIC_WEIGHT = 0.5  # rest goes to skill coverage
COVERAGE_PRIOR = 2     # smoothing: a job naming 1 skill you have scores 1/3, not 100%


def match_resume(conn, resume_text, n=15, candidates=60):
    """
    Retrieve-then-rerank:
      1. retrieve: each resume chunk is searched in ChromaDB; a job's semantic score is the
         mean of its two best chunk similarities (one strong section counts, but not alone)
      2. rerank:   final = 0.5 * semantic + 0.5 * skill coverage, where coverage is the share of
         the job's required skills that the resume mentions, smoothed as matched / (required + 2)
         so that jobs listing very few skills can't reach 100% on thin evidence. A job listing
         25 skills you mostly lack drops; a job whose stack you already have rises.
         Jobs the LLM classified as non-technical ('other') are skipped.
    Returns (resume_skills, matches); each match has matched / missing skills and both scores.
    """
    col = _collection()
    if not col.count():
        return [], []
    per_job = {}
    for chunk in _chunks(resume_text):
        res = col.query(query_texts=[chunk], n_results=min(candidates, col.count()))
        for i, d in zip(res["ids"][0], res["distances"][0], strict=True):
            per_job.setdefault(int(i), []).append(1 - d)
    semantic = {jid: sum(sorted(s, reverse=True)[:2]) / min(2, len(s)) for jid, s in per_job.items()}
    shortlist = sorted(semantic, key=lambda j: -semantic[j])[:candidates]

    resume_skills = set(keyword_skills(resume_text))
    rows = conn.execute(
        "SELECT job_id, title, company, location, role_family, seniority, skills, url, "
        "salary_usd_year_min, salary_usd_year_max FROM gold.jobs_enriched WHERE job_id = ANY(%s)",
        (shortlist,)).fetchall()
    matches = []
    for r in rows:
        r = dict(r)
        job_skills = set(r["skills"] or [])
        if r["role_family"] == "other":
            continue
        coverage = len(job_skills & resume_skills) / (len(job_skills) + COVERAGE_PRIOR)
        r.update(semantic=round(semantic[r["job_id"]], 3), coverage=round(coverage, 3),
                 score=round(SEMANTIC_WEIGHT * semantic[r["job_id"]] + (1 - SEMANTIC_WEIGHT) * coverage, 3),
                 matched=sorted(job_skills & resume_skills), missing=sorted(job_skills - resume_skills))
        matches.append(r)
    matches.sort(key=lambda m: -m["score"])
    return sorted(resume_skills), matches[:n]
