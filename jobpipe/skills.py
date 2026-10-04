"""Canonical skill vocabulary, shared by the LLM extractor and the keyword baseline.

Both extractors may only return names from SKILLS, so their output can be compared
fairly in the evaluation and aggregated consistently in the gold layer.
"""

import re

# canonical name -> extra aliases (the canonical name itself is always matched)
SKILLS = {
    # languages
    "Python": [], "SQL": [], "Java": [], "JavaScript": ["js"], "TypeScript": [], "Go": ["golang"],
    "Rust": [], "C++": ["cpp"], "C#": ["c sharp", ".net", "dotnet"], "Scala": [], "R": [],
    "Kotlin": [], "Swift": [], "PHP": [], "Ruby": ["rails", "ruby on rails"], "Bash": ["shell scripting"],
    # data engineering
    "Spark": ["pyspark", "apache spark"], "Kafka": ["apache kafka"], "Airflow": ["apache airflow"],
    "dbt": [], "Snowflake": [], "BigQuery": [], "Redshift": [], "Databricks": [], "Hadoop": [],
    "Flink": ["apache flink"], "ETL": ["elt", "data pipelines"], "Data Warehousing": ["data warehouse"],
    # databases
    "PostgreSQL": ["postgres"], "MySQL": [], "MongoDB": [], "Redis": [], "Elasticsearch": ["opensearch"],
    "Cassandra": [], "DynamoDB": [],
    # analytics / BI
    "Excel": [], "Tableau": [], "Power BI": ["powerbi"], "Looker": [], "pandas": [], "NumPy": [],
    "Statistics": ["statistical"], "A/B Testing": ["ab testing", "experimentation"],
    # ML / AI
    "Machine Learning": ["ML"], "Deep Learning": [], "PyTorch": [], "TensorFlow": [], "scikit-learn": ["sklearn"],
    "NLP": ["natural language processing"], "Computer Vision": [], "LLMs": ["llm", "large language models", "genai", "generative ai"],
    "RAG": ["retrieval augmented generation", "retrieval-augmented generation"], "MLOps": [],
    # cloud / infra
    "AWS": ["amazon web services"], "GCP": ["google cloud"], "Azure": ["microsoft azure"],
    "Docker": [], "Kubernetes": ["k8s"], "Terraform": [], "CI/CD": ["ci cd", "continuous integration"],
    "Linux": [], "Git": ["github", "gitlab"], "Microservices": [], "REST APIs": ["rest api", "restful"],
    "GraphQL": [],
    # web / mobile
    "React": ["react.js", "reactjs"], "Node.js": ["nodejs"], "Angular": [], "Vue": ["vue.js"],
    "Next.js": ["nextjs"], "Django": [], "Flask": [], "FastAPI": [], "Spring": ["spring boot"],
    "iOS": [], "Android": [], "React Native": [], "Flutter": [],
    # practices
    "Agile": ["scrum"], "System Design": [], "Security": ["cybersecurity", "infosec"], "Testing": ["unit testing", "QA", "test automation"],
}

# Names that are also ordinary words are matched case-sensitively
# ("you will excel", "react quickly", "spring internship", "go to market", "R&D")
CASE_SENSITIVE = {"R", "Go", "ML", "QA", "ETL", "SQL", "Spring", "Excel", "Swift", "React", "Rust", "Flask"}

_PATTERNS = []
for canonical, aliases in SKILLS.items():
    for name in [canonical, *aliases]:
        flags = 0 if name in CASE_SENSITIVE else re.IGNORECASE
        # word boundaries that also work for names like C++, C#, Node.js, CI/CD ("&" so R&D isn't R)
        _PATTERNS.append((canonical, re.compile(rf"(?<![\w+#&.]){re.escape(name)}(?![\w+#&])", flags)))

TECH_TITLE = re.compile(
    r"engineer|developer|programmer|data|analyst|scientist|machine learning|\bml\b|\bai\b|devops|"
    r"\bsre\b|cloud|architect|software|backend|back-end|frontend|front-end|full[- ]?stack|"
    r"\bqa\b|test|security|platform|infrastructure|mobile|ios|android",
    re.IGNORECASE,
)


def keyword_skills(text):
    """Baseline extractor: dictionary match against the description. Returns sorted canonical names."""
    found = {canonical for canonical, pattern in _PATTERNS if pattern.search(text or "")}
    return sorted(found)


def is_tech_title(title):
    title = title or ""
    return bool(TECH_TITLE.search(title) or re.search(r"\bIT\b", title))  # "IT Support", case-sensitive


def canonicalise(names):
    """Map LLM output onto the vocabulary (case-insensitive, aliases allowed); drop anything unknown."""
    lookup = {}
    for canonical, aliases in SKILLS.items():
        for name in [canonical, *aliases]:
            lookup[name.lower()] = canonical
    return sorted({lookup[n.strip().lower()] for n in names if isinstance(n, str) and n.strip().lower() in lookup})
