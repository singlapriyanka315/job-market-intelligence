import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/jobs")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./vector_db")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# Free tier: 8,000 tokens/minute and 1,000 requests/day - stay a little under
GROQ_TOKENS_PER_MINUTE = int(os.environ.get("GROQ_TOKENS_PER_MINUTE", "7000"))
JOBS_PER_LLM_CALL = 4          # batching amortises the prompt (skill list) over several jobs
MAX_LLM_JOBS_PER_RUN = int(os.environ.get("MAX_LLM_JOBS_PER_RUN", "120"))
DESCRIPTION_CHARS_FOR_LLM = 2000

# Bump when the extraction prompt changes: jobs are re-extracted with the new version
PROMPT_VERSION = "v1"

ADZUNA_APP_ID = os.environ.get("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY")
ADZUNA_QUERIES = ["data engineer", "data analyst", "data scientist", "machine learning engineer",
                  "python developer", "backend developer", "devops engineer"]

USER_AGENT = "job-market-intelligence/1.0 (learning project; https://github.com/singlapriyanka315)"

# Fixed FX rates for comparing salaries (approximate; update occasionally)
USD_PER = {"USD": 1.0, "INR": 1 / 88.0, "EUR": 1.16, "GBP": 1.34, "CAD": 0.72, "AUD": 0.66}
HOURS_PER_YEAR = 2080
MONTHS_PER_YEAR = 12
