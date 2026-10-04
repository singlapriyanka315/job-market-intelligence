from jobpipe.extract import TokenBucket, validate
from jobpipe.skills import canonicalise, is_tech_title, keyword_skills


def test_keyword_skills_aliases_and_symbols():
    found = keyword_skills("PySpark, Airflow, k8s, Node.js, C++, C#, CI/CD, Postgres and LLMs")
    assert {"Spark", "Airflow", "Kubernetes", "Node.js", "C++", "C#", "CI/CD", "PostgreSQL", "LLMs"} <= set(found)


def test_keyword_skills_avoids_common_false_positives():
    assert keyword_skills("You will excel in R&D and react quickly to html changes") == []
    assert "JavaScript" not in keyword_skills("Node.js")  # ".js" is not JavaScript
    assert "Machine Learning" not in keyword_skills("HTML")


def test_canonicalise_maps_aliases_and_drops_unknown():
    assert canonicalise(["pyspark", "K8S", "postgres", "Blockchain Wizardry", None]) == \
        ["Kubernetes", "PostgreSQL", "Spark"]


def test_is_tech_title():
    assert is_tech_title("Senior Data Engineer")
    assert is_tech_title("Werkstudent IT Support (m/w/d)")
    assert not is_tech_title("Senior Sales Manager")
    assert not is_tech_title("Sales - it is great")


def test_validate_clamps_llm_output():
    out = validate({"role_family": "wizard", "seniority": "senior", "years_experience_min": "45",
                    "remote_policy": "remote", "skills": ["python", "made-up"]})
    assert out == {"role_family": "other", "seniority": "senior", "years_experience_min": None,
                   "remote_policy": "remote", "skills": ["Python"]}


def test_token_bucket_does_not_wait_when_under_limit():
    b = TokenBucket(1000)
    b.record(300)
    b.wait_for(500)  # 800 <= 1000: returns immediately
    assert sum(t for _, t in b.used) == 300
