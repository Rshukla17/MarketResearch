import time
import requests

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

def _zfill_cik(cik: int) -> str:
    return str(int(cik)).zfill(10)

def get_sec_ticker_map(session: requests.Session) -> dict:
    r = session.get(SEC_TICKER_MAP_URL, timeout=20)
    r.raise_for_status()
    data = r.json()
    out = {}
    for _, v in data.items():
        t = str(v.get("ticker", "")).upper().strip()
        cik = int(v.get("cik_str", 0))
        title = v.get("title", "")
        if t and cik:
            out[t] = {"cik": cik, "title": title}
    return out

def get_company_submissions(session: requests.Session, cik: int) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{_zfill_cik(cik)}.json"
    r = session.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def get_company_facts(session: requests.Session, cik: int) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{_zfill_cik(cik)}.json"
    r = session.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def latest_filings_table(submissions: dict, cik: int, limit: int = 15) -> list[dict]:
    recent = (submissions.get("filings") or {}).get("recent") or {}
    forms = recent.get("form", []) or []
    acc = recent.get("accessionNumber", []) or []
    filed = recent.get("filingDate", []) or []
    primary = recent.get("primaryDocument", []) or []

    rows = []
    for i in range(min(limit, len(forms))):
        accession = str(acc[i]).replace("-", "")
        doc = primary[i] if i < len(primary) else ""
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
        rows.append({"date": filed[i] if i < len(filed) else None, "form": forms[i], "document": doc, "url": url})
    return rows

def _get_units_series(facts: dict, tag: str):
    """
    Returns a list of fact records for a tag if available, else None.
    """
    try:
        node = facts["facts"]["us-gaap"][tag]["units"]
    except Exception:
        return None

    # prefer USD, else first unit
    if "USD" in node:
        return node["USD"]
    first_key = next(iter(node.keys()))
    return node[first_key]

def latest_annual_two(facts: dict, tag: str):
    """
    Two latest annual FY values for a given tag.
    Returns dict with curr and prev or None.
    """
    series = _get_units_series(facts, tag)
    if not series:
        return None
    rows = [r for r in series if str(r.get("fp", "")).upper() == "FY"]
    rows = sorted(rows, key=lambda x: str(x.get("end", "")))
    if len(rows) < 2:
        return None
    prev, curr = rows[-2], rows[-1]
    return {
        "prev": prev.get("val"),
        "prev_end": prev.get("end"),
        "curr": curr.get("val"),
        "curr_end": curr.get("end"),
    }

def last_quarters(facts: dict, tag: str, n: int = 4) -> list[dict]:
    """
    Last n quarterly values for a tag.
    """
    series = _get_units_series(facts, tag)
    if not series:
        return []
    rows = [r for r in series if str(r.get("fp", "")).upper() in ["Q1", "Q2", "Q3", "Q4"]]
    rows = sorted(rows, key=lambda x: str(x.get("end", "")))
    rows = rows[-n:]
    out = []
    for r in rows:
        out.append({"end": r.get("end"), "fy": r.get("fy"), "fp": r.get("fp"), "val": r.get("val"), "form": r.get("form")})
    return out
