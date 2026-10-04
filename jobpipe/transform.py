"""Bronze -> silver: turn each source's payload into one clean, typed record.

Everything here is a pure function (no database), so it is fully unit-tested.
"""

import hashlib
import html
import re
from datetime import UTC, datetime
from html.parser import HTMLParser

from config import DESCRIPTION_CHARS_FOR_LLM, HOURS_PER_YEAR, MONTHS_PER_YEAR, USD_PER
from jobpipe.skills import is_tech_title, keyword_skills

# ---------------------------------------------------------------- text
# a UTF-8 lead byte (C2-F4) followed by a continuation byte (80-BF), both read as Latin-1
_MOJIBAKE = re.compile("[\u00c2-\u00f4][\u0080-\u00bf]")


def fix_mojibake(text):
    """Undo UTF-8 that was decoded as Latin-1 ('â\x80\x94' -> '—'); some APIs send text like this."""
    if not text or not _MOJIBAKE.search(text):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


class _TextExtractor(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "tr"}

    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(raw):
    if not raw:
        return ""
    if "<" not in raw:
        return re.sub(r"[ \t]+", " ", html.unescape(raw)).strip()
    p = _TextExtractor()
    p.feed(raw)
    text = html.unescape("".join(p.parts))
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


REQUIREMENT_HEADERS = re.compile(
    r"requirement|qualification|you have|you bring|what you.?ll need|your profile|skills|experience with|"
    r"must have|nice to have|tech stack|we.?re looking for|anforderung|dein profil|ihr profil|das bringst du",
    re.IGNORECASE,
)


def llm_excerpt(description, limit):
    """
    Pick the parts of a description worth sending to the LLM, within `limit` chars.
    Long postings often open with company marketing and list requirements at the end, so a
    plain [:limit] cut loses the skills. Paragraphs are scored by requirement-style headings
    and skill mentions, the best are kept, and they are put back in their original order.
    """
    if len(description or "") <= limit:
        return description or ""
    paras = [x for x in re.split(r"\n\s*\n|\n(?=- )", description) if x.strip()]
    scored = []
    for i, para in enumerate(paras):
        score = 3 * bool(REQUIREMENT_HEADERS.search(para)) + len(keyword_skills(para))
        scored.append((score, -i, i, para))
    keep, used = set(), 0
    for _score, _, i, para in sorted(scored, reverse=True):
        if used + len(para) > limit:
            continue
        keep.add(i)
        used += len(para) + 2
    if not keep:
        return description[:limit]
    return "\n\n".join(paras[i] for i in sorted(keep))


# ---------------------------------------------------------------- salary
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}
_NUM = r"(\d+(?:[.,]\d+)*)\s*([kK])?"


def _to_number(num, k):
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", num):  # European thousands: 60.000
        num = num.replace(".", "")
    n = float(num.replace(",", ""))
    return n * 1000 if k else n


def parse_salary_text(text):
    """
    '$90k - $105k' -> (90000, 105000, 'USD', 'year')
    '$90 - $150 /hour' -> (90, 150, 'USD', 'hour')
    '€60,000' -> (60000, None, 'EUR', 'year')
    Returns (None, None, None, None) when there is no usable salary.
    """
    if not text or not re.search(r"\d", text):
        return None, None, None, None
    currency = next((c for sym, c in CURRENCY_SYMBOLS.items() if sym in text), None)
    if currency is None:
        m = re.search(r"\b(USD|EUR|GBP|INR|CAD|AUD)\b", text, re.IGNORECASE)
        currency = m.group(1).upper() if m else "USD"
    nums = re.findall(_NUM, text)
    if not nums:
        return None, None, None, None
    values = [_to_number(n, k) for n, k in nums[:2]]
    low = text.lower()
    if re.search(r"/\s*h(ou)?r|per hour|hourly|/h\b", low):
        period = "hour"
    elif re.search(r"/\s*mo(nth)?|per month|monthly", low):
        period = "month"
    else:
        period = "year"
    # '90 - 150k' -> both are thousands
    if len(values) == 2 and nums[1][1] and not nums[0][1] and values[0] < 1000:
        values[0] *= 1000
    lo = min(values)
    hi = max(values) if len(values) == 2 else None
    if period == "year" and lo < 1000:  # e.g. "$5 equity" - not a salary
        return None, None, None, None
    return lo, hi, currency, period


def to_usd_year(amount, currency, period):
    if amount is None or currency not in USD_PER or period is None:
        return None
    factor = {"year": 1, "month": MONTHS_PER_YEAR, "hour": HOURS_PER_YEAR}[period]
    return round(float(amount) * factor * USD_PER[currency], 2)


# ---------------------------------------------------------------- location
COUNTRY_HINTS = [
    ("IN", r"\bindia\b|bengaluru|bangalore|mumbai|delhi|gurgaon|gurugram|noida|hyderabad|pune|chennai|kolkata|ahmedabad"),
    ("US", r"\busa?\b|united states|america|new york|san francisco|california|texas|seattle|boston|chicago"),
    ("GB", r"\buk\b|united kingdom|england|london|manchester|scotland"),
    ("DE", r"germany|deutschland|berlin|munich|münchen|hamburg|frankfurt|cologne|köln|stuttgart|düsseldorf|leipzig|dresden|nuremberg|nürnberg|hannover|bremen|essen|dortmund|bonn|mannheim|karlsruhe"),
    ("CA", r"canada|toronto|vancouver|montreal"),
    ("NL", r"netherlands|amsterdam|rotterdam"),
    ("EU", r"\beurope\b|\bemea\b|\beu\b"),
    ("REMOTE", r"worldwide|anywhere|global|remote"),
]


def guess_country(location, default=None):
    loc = (location or "").lower()
    for code, pattern in COUNTRY_HINTS:
        if re.search(pattern, loc):
            return code
    return default


# ---------------------------------------------------------------- keys
_COMPANY_SUFFIX = re.compile(r"\b(gmbh|inc|ltd|llc|pvt|private|limited|corp|co|ag|se|plc)\b\.?", re.IGNORECASE)
_GENDER_TAG = re.compile(r"\((m|w|f|d|x|all)(\s*/\s*(m|w|f|d|x))*\)|\b(m|f|w)/(m|f|w)/(d|x)\b", re.IGNORECASE)


def norm_company(name):
    s = _COMPANY_SUFFIX.sub("", (name or "").lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def norm_title(title):
    s = _GENDER_TAG.sub("", (title or "").lower())
    return re.sub(r"[^a-z0-9+#]+", " ", s).strip()


def dedupe_key(company, title):
    return f"{norm_company(company)}|{norm_title(title)}"


def content_hash(title, company, description):
    blob = f"{title}\n{company}\n{(description or '')[:DESCRIPTION_CHARS_FOR_LLM]}"
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _ts(value):
    if value in (None, "", 0):
        return None
    if isinstance(value, (int, float)) or str(value).isdigit():
        return datetime.fromtimestamp(int(value), UTC)
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)  # Remotive sends UTC without offset


EMPLOYMENT = {
    "full_time": "full_time", "full time": "full_time", "full-time": "full_time", "permanent": "full_time",
    "part_time": "part_time", "part time": "part_time", "part-time": "part_time",
    "contract": "contract", "freelance": "contract", "contractor": "contract",
    "internship": "internship", "intern": "internship", "praktikum": "internship", "working student": "part_time",
}


def _employment(*values):
    for v in values:
        for item in v if isinstance(v, list) else [v]:
            key = (item or "").strip().lower()
            if key in EMPLOYMENT:
                return EMPLOYMENT[key]
    return None


# ---------------------------------------------------------------- per-source mapping
def from_remotive(p):
    lo, hi, cur, per = parse_salary_text(p.get("salary"))
    return dict(title=p["title"], company=p.get("company_name"), location=p.get("candidate_required_location"),
                country=guess_country(p.get("candidate_required_location"), "REMOTE"), is_remote=True,
                employment_type=_employment(p.get("job_type")), salary_min=lo, salary_max=hi,
                salary_currency=cur, salary_period=per, posted_at=_ts(p.get("publication_date")),
                url=p.get("url"), description=html_to_text(p.get("description")))


def from_arbeitnow(p):
    return dict(title=p["title"], company=p.get("company_name"), location=p.get("location"),
                country=guess_country(p.get("location"), "DE"), is_remote=bool(p.get("remote")),
                employment_type=_employment(p.get("job_types")), salary_min=None, salary_max=None,
                salary_currency=None, salary_period=None, posted_at=_ts(p.get("created_at")),
                url=p.get("url"), description=html_to_text(p.get("description")))


def from_remoteok(p):
    lo = float(p["salary_min"]) if str(p.get("salary_min") or "0") not in ("0", "") else None
    hi = float(p["salary_max"]) if str(p.get("salary_max") or "0") not in ("0", "") else None
    if lo is not None and lo < 1000:  # e.g. 30-36: an hourly rate in an annual field - unit unknown, drop it
        lo = hi = None
    return dict(title=p.get("position") or "", company=p.get("company"), location=p.get("location") or "Remote",
                country=guess_country(p.get("location"), "REMOTE"), is_remote=True,
                employment_type=_employment(p.get("tags")), salary_min=lo, salary_max=hi,
                salary_currency="USD" if lo else None, salary_period="year" if lo else None,
                posted_at=_ts(p.get("date")), url=p.get("url"), description=html_to_text(p.get("description")))


def from_adzuna(p):
    predicted = str(p.get("salary_is_predicted", "0")) == "1"  # Adzuna's own estimate, not the employer's
    lo = None if predicted else p.get("salary_min")
    hi = None if predicted else p.get("salary_max")
    if lo is not None and hi is not None and lo == hi:
        hi = None
    return dict(title=p.get("title") or "", company=(p.get("company") or {}).get("display_name"),
                location=(p.get("location") or {}).get("display_name"), country="IN",
                is_remote=bool(re.search(r"remote|work from home|wfh", (p.get("title", "") + p.get("description", "")).lower())),
                employment_type=_employment(p.get("contract_time"), p.get("contract_type")),
                salary_min=lo, salary_max=hi, salary_currency="INR" if lo else None,
                salary_period="year" if lo else None, posted_at=_ts(p.get("created")),
                url=p.get("redirect_url"), description=html_to_text(p.get("description")))


MAPPERS = {"remotive": from_remotive, "arbeitnow": from_arbeitnow, "remoteok": from_remoteok, "adzuna_in": from_adzuna}


def to_silver(source, source_job_id, payload):
    rec = MAPPERS[source](payload)
    for field in ("title", "company", "location", "description"):
        rec[field] = fix_mojibake(rec[field])
    rec["title"] = html.unescape(rec["title"]).strip()
    rec.update(
        source=source,
        source_job_id=source_job_id,
        salary_usd_year_min=to_usd_year(rec["salary_min"], rec["salary_currency"], rec["salary_period"]),
        salary_usd_year_max=to_usd_year(rec["salary_max"], rec["salary_currency"], rec["salary_period"]),
        content_hash=content_hash(rec["title"], rec["company"], rec["description"]),
        dedupe_key=dedupe_key(rec["company"], rec["title"]),
        is_tech=is_tech_title(rec["title"]),
        keyword_skills=keyword_skills(f"{rec['title']}\n{rec['description']}"),
    )
    return rec
