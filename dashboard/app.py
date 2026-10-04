"""
Job Market Intelligence dashboard.

    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from jobpipe import db, vectors  # noqa: E402

st.set_page_config(page_title="Job Market Intelligence", layout="wide")


@st.cache_resource
def conn():
    c = db.connect()
    c.autocommit = True
    return c


def q(sql, params=None):
    return pd.DataFrame(conn().execute(sql, params or ()).fetchall())


ROLE_LABELS = {
    "all": "All tech roles", "data_engineer": "Data engineer", "data_scientist": "Data scientist",
    "data_analyst": "Data analyst", "ml_engineer": "ML engineer", "backend": "Backend", "frontend": "Frontend",
    "fullstack": "Full-stack", "mobile": "Mobile", "devops_sre": "DevOps / SRE", "qa": "QA",
    "security": "Security", "it_support": "IT support", "engineering_manager": "Engineering manager",
    "other": "Other", "unknown": "Not yet classified",
}

# ---------------------------------------------------------------- header + KPIs
st.title("Job Market Intelligence")
st.caption("Which skills tech employers ask for, what they pay, and how your resume matches — "
           "built from public job APIs, cleaned in PostgreSQL, enriched with an LLM.")

k = q("""SELECT count(*) AS jobs, count(*) FILTER (WHERE llm_enriched) AS enriched,
                count(*) FILTER (WHERE salary_usd_year_min IS NOT NULL) AS with_salary,
                count(DISTINCT source) AS sources FROM gold.jobs_enriched""").iloc[0]
last = q("SELECT run_id, status, finished_at FROM ops.pipeline_runs ORDER BY run_id DESC LIMIT 1")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Unique tech jobs", f"{k.jobs:,}")
c2.metric(f"LLM-enriched ({100 * k.enriched / max(k.jobs, 1):.0f}%)", f"{k.enriched:,}")
c3.metric("With salary", f"{k.with_salary:,}")
if not last.empty:
    r = last.iloc[0]
    when = r.finished_at.strftime("%d %b %H:%M") if pd.notna(r.finished_at) else "in progress"
    c4.metric(f"Last run · {when}", f"#{r.run_id} {r.status}")

tab_skills, tab_salary, tab_search, tab_resume, tab_health = st.tabs(
    ["Skills in demand", "Salaries", "Semantic search", "Resume match", "Pipeline health"])

# ---------------------------------------------------------------- skills
with tab_skills:
    roles = q("SELECT role_family, count(*) AS n FROM gold.jobs_enriched GROUP BY 1 ORDER BY 2 DESC")
    options = ["all"] + [r for r in roles.role_family if r not in ("unknown", "other")]
    left, right = st.columns([1, 3])
    role = left.selectbox("Role", options, format_func=lambda r: ROLE_LABELS.get(r, r))
    top_n = left.slider("Skills to show", 5, 30, 15)
    enriched_only = left.toggle("LLM-enriched jobs only", value=True,
                                help="Jobs not yet sent to the LLM fall back to keyword-matched skills.")
    where = "AND llm_enriched" if enriched_only else ""
    role_filter = "" if role == "all" else "AND role_family = %(role)s"
    total = q(f"SELECT count(*) AS n FROM gold.jobs_enriched WHERE true {where} {role_filter}", {"role": role}).n[0]
    skills = q(f"""
        SELECT skill, count(*) AS jobs FROM gold.jobs_enriched, unnest(skills) AS skill
        WHERE true {where} {role_filter} GROUP BY skill ORDER BY jobs DESC LIMIT %(n)s
    """, {"role": role, "n": top_n})
    if skills.empty:
        right.info("No jobs for this filter yet.")
    else:
        skills["pct"] = (100 * skills.jobs / total).round(1)
        base = alt.Chart(skills).encode(
            x=alt.X("pct:Q", title="% of job postings"),
            y=alt.Y("skill:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=180)),
            tooltip=["skill", "jobs", alt.Tooltip("pct", title="% of jobs")])
        right.altair_chart((base.mark_bar(cornerRadiusEnd=3)
                            + base.mark_text(align="left", dx=4, fontSize=11, color="#8b95a1").encode(text=alt.Text("pct:Q", format=".0f")))
                           .properties(height=30 * len(skills)), use_container_width=True)
        right.caption(f"Based on {total:,} postings. A skill counts once per posting.")

    st.subheader("Role mix")
    mix = roles[~roles.role_family.isin(["unknown"])].copy()
    mix["role"] = mix.role_family.map(lambda r: ROLE_LABELS.get(r, r))
    st.altair_chart(alt.Chart(mix).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("n:Q", title="Postings"), y=alt.Y("role:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
        tooltip=["role", "n"]).properties(height=30 * len(mix)), use_container_width=True)

# ---------------------------------------------------------------- salary
with tab_salary:
    sal = q("""SELECT role_family, count(*) AS jobs,
                      round(percentile_cont(0.5) WITHIN GROUP (ORDER BY
                          (salary_usd_year_min + coalesce(salary_usd_year_max, salary_usd_year_min)) / 2)) AS median
               FROM gold.jobs_enriched WHERE salary_usd_year_min IS NOT NULL AND role_family NOT IN ('unknown')
               GROUP BY 1 ORDER BY median DESC""")
    st.caption("Annual salary normalised to USD (fixed FX rates in config.py). Only postings that state a salary; "
               "Adzuna's *predicted* salaries are excluded.")
    if sal.empty:
        st.info("No salary data yet.")
    else:
        sal["role"] = sal.role_family.map(lambda r: ROLE_LABELS.get(r, r))
        sal["label"] = sal.role + " (n=" + sal.jobs.astype(str) + ")"
        st.altair_chart(alt.Chart(sal).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("median:Q", title="Median annual salary (USD)", axis=alt.Axis(format="$,.0f")),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)),
            tooltip=["role", "jobs", alt.Tooltip("median", format="$,.0f")],
        ).properties(height=34 * len(sal)), use_container_width=True)
        if sal.jobs.min() < 5:
            st.warning("Some roles have fewer than 5 salaries — treat those medians as anecdotes, not statistics.")
        st.dataframe(q("""SELECT title, company, location, source, salary_usd_year_min AS min_usd,
                                 salary_usd_year_max AS max_usd FROM gold.jobs_enriched
                          WHERE salary_usd_year_min IS NOT NULL ORDER BY salary_usd_year_min DESC"""),
                     use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- search
with tab_search:
    query = st.text_input("Describe the job you want", "remote data engineer with Python, Airflow and AWS")
    if query:
        hits = vectors.search(query, n=15)
        if hits:
            rows = q("""SELECT job_id, title, company, location, role_family, skills, url FROM gold.jobs_enriched
                        WHERE job_id = ANY(%s)""", ([h for h, _ in hits],)).set_index("job_id")
            for jid, sim in hits:
                if jid in rows.index:
                    r = rows.loc[jid]
                    st.markdown(f"**[{r.title}]({r.url})** — {r.company} · {r.location}  \n"
                                f"<span style='opacity:.7'>similarity {sim:.2f} · "
                                f"{', '.join(list(r.skills)[:8])}</span>", unsafe_allow_html=True)
        st.caption("Meaning-based search (MiniLM embeddings in ChromaDB), not keyword matching.")

# ---------------------------------------------------------------- resume
with tab_resume:
    st.write("Upload your resume to see the best-matching jobs and which skills they ask for that it doesn't mention. "
             "The file is processed in memory and not stored.")
    up = st.file_uploader("Resume (PDF or TXT)", type=["pdf", "txt"])
    if up:
        if up.name.lower().endswith(".pdf"):
            from pypdf import PdfReader
            text = "\n".join(p.extract_text() or "" for p in PdfReader(up).pages)
        else:
            text = up.read().decode("utf-8", errors="ignore")
        have, matches = vectors.match_resume(conn(), text, n=15)
        st.markdown(f"**Skills found in your resume:** {', '.join(have) or 'none detected'}")
        gaps = {}
        for m in matches:
            for s in m["missing"]:
                gaps[s] = gaps.get(s, 0) + 1
        if gaps:
            g = pd.DataFrame(sorted(gaps.items(), key=lambda x: -x[1])[:10], columns=["skill", "matches"])
            st.markdown("**Skills to learn next** — asked for by your best matches but missing from your resume:")
            st.altair_chart(alt.Chart(g).mark_bar(cornerRadiusEnd=3, color="#c2410c").encode(
                x=alt.X("matches:Q", title="of your top matches"),
                y=alt.Y("skill:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False))
            ).properties(height=30 * len(g)), use_container_width=True)
        for m in matches:
            st.markdown(f"**[{m['title']}]({m['url']})** — {m['company']} · match {m['score']:.2f}  \n"
                        f"✅ {', '.join(m['matched']) or '—'}  \n❌ {', '.join(m['missing']) or '—'}")

# ---------------------------------------------------------------- health
with tab_health:
    st.subheader("Sources")
    st.dataframe(q("SELECT * FROM gold.source_summary ORDER BY jobs DESC"), use_container_width=True, hide_index=True)
    st.subheader("Data-quality checks (latest run)")
    dq = q("""SELECT check_name, severity, passed, failing_rows, details FROM ops.dq_results
              WHERE run_id = (SELECT max(run_id) FROM ops.dq_results) ORDER BY passed, severity, check_name""")
    st.dataframe(dq, use_container_width=True, hide_index=True)
    st.subheader("Recent runs")
    st.dataframe(q("""SELECT run_id, status, started_at, finished_at - started_at AS duration, stats
                      FROM ops.pipeline_runs ORDER BY run_id DESC LIMIT 10"""),
                 use_container_width=True, hide_index=True)

st.divider()
st.caption("Data: [Remotive](https://remotive.com), [Arbeitnow](https://www.arbeitnow.com), "
           "[Remote OK](https://remoteok.com), [Adzuna](https://www.adzuna.in) public APIs. "
           "Skills, roles and seniority are extracted by an LLM; see eval/ for measured accuracy.")
