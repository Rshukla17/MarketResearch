import os
import json
import traceback
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import xml.etree.ElementTree as ET

try:
    import ollama
except Exception:
    ollama = None

# -----------------------------
# CONFIG
# -----------------------------
from pathlib import Path
import streamlit as st

LOGO_PATH = Path(__file__).parent / "logo.png"

with st.sidebar:
    st.image(str(LOGO_PATH), use_container_width=True)
    st.markdown("---")

st.set_page_config(page_title="Brahmin Terminal", layout="wide")
st.title("Brahmin Terminal")
st.caption("Astrology for Equities")

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "AlphaMindResearch/1.0 (contact: youremail@example.com)")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")

session = requests.Session()
session.headers.update({"User-Agent": SEC_USER_AGENT})

HTTP_TIMEOUT = 20


# -----------------------------
# HELPERS
# -----------------------------
def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _pct(x, y):
    if x is None or y is None or y == 0:
        return None
    return (x / y) * 100.0


def zfill_cik(cik: int) -> str:
    return str(int(cik)).zfill(10)


def _get_units_series(facts: dict, tag: str):
    try:
        node = facts["facts"]["us-gaap"][tag]["units"]
    except Exception:
        return None

    if "USD" in node:
        return node["USD"]

    try:
        return node[next(iter(node.keys()))]
    except Exception:
        return None


def latest_annual_two(facts: dict, tag: str):
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


def last_quarters(facts: dict, tag: str, n: int = 4):
    series = _get_units_series(facts, tag)
    if not series:
        return []

    rows = [r for r in series if str(r.get("fp", "")).upper() in ["Q1", "Q2", "Q3", "Q4"]]
    rows = sorted(rows, key=lambda x: str(x.get("end", "")))
    rows = rows[-n:]

    return [
        {
            "end": r.get("end"),
            "fy": r.get("fy"),
            "fp": r.get("fp"),
            "val": r.get("val"),
            "form": r.get("form"),
        }
        for r in rows
    ]


def compute_piotroski(facts: dict):
    missing = []

    def two(tag, name):
        v = latest_annual_two(facts, tag)
        if v is None:
            missing.append(name)
            return None
        return v

    ni = two("NetIncomeLoss", "NetIncomeLoss")
    assets = two("Assets", "Assets")
    cfo = two("NetCashProvidedByUsedInOperatingActivities", "OperatingCashFlow")
    ltd = two("LongTermDebtNoncurrent", "LongTermDebtNoncurrent")
    ca = two("AssetsCurrent", "AssetsCurrent")
    cl = two("LiabilitiesCurrent", "LiabilitiesCurrent")
    shares = two("CommonStockSharesOutstanding", "CommonStockSharesOutstanding")
    gross = two("GrossProfit", "GrossProfit")
    rev = two("Revenues", "Revenues")

    checks = {}

    roa_curr = None
    roa_prev = None
    if ni and assets and _safe_float(assets["curr"]) and _safe_float(assets["prev"]):
        if _safe_float(ni["curr"]) is not None:
            roa_curr = _safe_float(ni["curr"]) / _safe_float(assets["curr"])
        if _safe_float(ni["prev"]) is not None:
            roa_prev = _safe_float(ni["prev"]) / _safe_float(assets["prev"])

    checks["ROA positive"] = 1 if (roa_curr is not None and roa_curr > 0) else 0
    checks["CFO positive"] = 1 if (_safe_float(cfo["curr"]) is not None and _safe_float(cfo["curr"]) > 0) else 0
    checks["ROA improved"] = 1 if (roa_curr is not None and roa_prev is not None and (roa_curr - roa_prev) > 0) else 0
    checks["CFO exceeds NI"] = 1 if (
        _safe_float(cfo["curr"]) is not None
        and _safe_float(ni["curr"]) is not None
        and _safe_float(cfo["curr"]) > _safe_float(ni["curr"])
    ) else 0

    checks["Leverage down"] = 1 if (
        _safe_float(ltd["curr"]) is not None
        and _safe_float(ltd["prev"]) is not None
        and (_safe_float(ltd["curr"]) - _safe_float(ltd["prev"])) < 0
    ) else 0

    cr_curr = None
    cr_prev = None
    if ca and cl and _safe_float(ca["curr"]) and _safe_float(cl["curr"]):
        cr_curr = _safe_float(ca["curr"]) / _safe_float(cl["curr"])
    if ca and cl and _safe_float(ca["prev"]) and _safe_float(cl["prev"]):
        cr_prev = _safe_float(ca["prev"]) / _safe_float(cl["prev"])

    checks["Current ratio up"] = 1 if (cr_curr is not None and cr_prev is not None and (cr_curr - cr_prev) > 0) else 0

    checks["No dilution"] = 1 if (
        _safe_float(shares["curr"]) is not None
        and _safe_float(shares["prev"]) is not None
        and (_safe_float(shares["curr"]) - _safe_float(shares["prev"])) <= 0
    ) else 0

    gm_curr = None
    gm_prev = None
    if gross and rev and _safe_float(gross["curr"]) is not None and _safe_float(rev["curr"]):
        gm_curr = _safe_float(gross["curr"]) / _safe_float(rev["curr"])
    if gross and rev and _safe_float(gross["prev"]) is not None and _safe_float(rev["prev"]):
        gm_prev = _safe_float(gross["prev"]) / _safe_float(rev["prev"])

    checks["Gross margin up"] = 1 if (gm_curr is not None and gm_prev is not None and (gm_curr - gm_prev) > 0) else 0

    at_curr = None
    at_prev = None
    if rev and assets and _safe_float(rev["curr"]) is not None and _safe_float(assets["curr"]):
        at_curr = _safe_float(rev["curr"]) / _safe_float(assets["curr"])
    if rev and assets and _safe_float(rev["prev"]) is not None and _safe_float(assets["prev"]):
        at_prev = _safe_float(rev["prev"]) / _safe_float(assets["prev"])

    checks["Asset turnover up"] = 1 if (at_curr is not None and at_prev is not None and (at_curr - at_prev) > 0) else 0

    score = sum(checks.values())
    return {"score": score, "checks": checks, "missing": missing}


def google_news_rss(query: str, limit: int = 20) -> pd.DataFrame:
    url = "https://news.google.com/rss/search"
    params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}

    try:
        r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            return pd.DataFrame()

        root = ET.fromstring(r.text)
        items = root.findall(".//item")
        rows = []
        for it in items[:limit]:
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            pubdate = (it.findtext("pubDate") or "").strip()
            source = it.find("source")
            source_name = source.text.strip() if source is not None and source.text else "GoogleNews"
            rows.append({"seen": pubdate, "title": title, "source": source_name, "url": link})

        df = pd.DataFrame(rows)
        if not df.empty:
            df["seen"] = pd.to_datetime(df["seen"], errors="coerce")
            df = df.sort_values("seen", ascending=False).reset_index(drop=True)
        return df

    except Exception:
        return pd.DataFrame()


def gdelt_news(query: str, max_records: int = 25) -> pd.DataFrame:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "sort": "HybridRel",
    }

    try:
        r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            return pd.DataFrame()

        text = (r.text or "").strip()
        if not text:
            return pd.DataFrame()

        ct = (r.headers.get("Content-Type") or "").lower()
        if "json" not in ct:
            try:
                j = r.json()
            except Exception:
                return pd.DataFrame()
        else:
            j = r.json()

        arts = j.get("articles", []) or []
        rows = []
        for a in arts:
            rows.append({"seen": a.get("seendate"), "title": a.get("title"), "source": a.get("domain"), "url": a.get("url")})

        df = pd.DataFrame(rows)
        if not df.empty:
            df["seen"] = pd.to_datetime(df["seen"], errors="coerce")
            df = df.sort_values("seen", ascending=False).reset_index(drop=True)
        return df

    except Exception:
        return pd.DataFrame()


def latest_filings_table(submissions: dict, cik: int, limit: int = 15):
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


def summarize_price_moves(hist: pd.DataFrame, top_n: int = 8):
    if hist is None or hist.empty:
        return []

    df = hist.copy()
    if "date" not in df.columns:
        return []

    df = df.sort_values("date")
    if "Close" not in df.columns:
        return []

    df["ret"] = df["Close"].pct_change()
    df = df.dropna(subset=["ret"])
    df["absret"] = df["ret"].abs()
    df = df.sort_values("absret", ascending=False).head(top_n)

    out = []
    for _, r in df.iterrows():
        d = r["date"]
        if hasattr(d, "strftime"):
            ds = d.strftime("%Y-%m-%d")
        else:
            ds = str(d)
        out.append({"date": ds, "return_pct": float(r["ret"] * 100.0)})
    return out


@st.cache_data(ttl=6 * 60 * 60)
def sec_ticker_map():
    url = "https://www.sec.gov/files/company_tickers.json"
    r = session.get(url, timeout=HTTP_TIMEOUT)
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


@st.cache_data(ttl=20 * 60)
def sec_submissions(cik: int):
    url = f"https://data.sec.gov/submissions/CIK{zfill_cik(cik)}.json"
    r = session.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=20 * 60)
def sec_facts(cik: int):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{zfill_cik(cik)}.json"
    r = session.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=10 * 60)
def price_5y(ticker: str) -> pd.DataFrame:
    h = yf.Ticker(ticker).history(period="5y")
    if h is None or h.empty:
        return pd.DataFrame()
    h = h.reset_index()
    if "Date" in h.columns:
        h = h.rename(columns={"Date": "date"})
    elif "Datetime" in h.columns:
        h = h.rename(columns={"Datetime": "date"})
    return h


@st.cache_data(ttl=10 * 60)
def yf_profile(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        return info
    except Exception:
        return {}


def extract_ceo(info: dict):
    officers = info.get("companyOfficers")
    if isinstance(officers, list):
        for o in officers:
            t = str(o.get("title", "")).lower()
            if "chief executive" in t or t.strip() == "ceo":
                return o.get("name")
    return info.get("ceo") or None


def safe_text(x, fallback="N A"):
    if x is None:
        return fallback
    s = str(x).strip()
    return s if s else fallback


def format_money(x):
    try:
        if x is None:
            return "N A"
        x = float(x)
        sign = "-" if x < 0 else ""
        x = abs(x)
        for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
            if x >= div:
                return f"{sign}{x/div:.2f}{unit}"
        return f"{sign}{x:.0f}"
    except Exception:
        return "N A"


def ollama_available():
    return ollama is not None


def infer_competitors_with_llm(ticker: str, name: str, sector: str, industry: str, summary: str):
    if not ollama_available():
        return []

    prompt = f"""
You are a buy side equity researcher.
Return a JSON array of up to 8 US listed tickers that are direct competitors of {name} ({ticker}).
Use sector and industry and business summary to infer peers.
Only include tickers that are real and tradable in the US.
Output only JSON, no commentary.

Sector: {sector}
Industry: {industry}
Business summary: {summary}
"""
    try:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        txt = (resp.get("message") or {}).get("content", "").strip()
        txt = txt.replace("```json", "").replace("```", "").strip()
        data = json.loads(txt)
        if isinstance(data, list):
            cleaned = []
            for t in data:
                if isinstance(t, str):
                    tt = t.strip().upper()
                    if tt and len(tt) <= 8 and tt.isalnum():
                        cleaned.append(tt)
            uniq = []
            for t in cleaned:
                if t not in uniq and t != ticker:
                    uniq.append(t)
            return uniq[:8]
        return []
    except Exception:
        return []


@st.cache_data(ttl=10 * 60)
def basic_perf(ticker: str) -> dict:
    h = yf.Ticker(ticker).history(period="1y")
    if h is None or h.empty:
        return {"ticker": ticker, "price": None, "ret_1y_pct": None, "vol_1y_pct": None}

    h = h.dropna()
    if h.empty:
        return {"ticker": ticker, "price": None, "ret_1y_pct": None, "vol_1y_pct": None}

    close = h["Close"]
    r = close.pct_change().dropna()
    ret = (close.iloc[-1] / close.iloc[0] - 1.0) * 100.0 if len(close) > 1 else None
    vol = (r.std() * (252 ** 0.5)) * 100.0 if len(r) > 10 else None
    return {"ticker": ticker, "price": float(close.iloc[-1]), "ret_1y_pct": ret, "vol_1y_pct": vol}


def conviction_score_proxy(piotroski_score: int, ret_1y_pct, vol_1y_pct):
    score = 20.0
    score += 6.0 * float(piotroski_score)

    if ret_1y_pct is not None:
        score += max(min(ret_1y_pct / 3.0, 20.0), -10.0)

    if vol_1y_pct is not None:
        score += max(min(15.0 - (vol_1y_pct / 5.0), 15.0), -10.0)

    score = max(min(score, 100.0), 1.0)
    return int(round(score))


def build_llm_context(ticker: str, company: dict, sec_sub: dict, pio: dict, eps_q, rev_q, filings, news_df, moves):
    news_lines = []
    if news_df is not None and not news_df.empty:
        for _, r in news_df.head(10).iterrows():
            title = str(r.get("title", "")).strip()
            src = str(r.get("source", "")).strip()
            seen = r.get("seen")
            if hasattr(seen, "strftime"):
                seen_s = seen.strftime("%Y-%m-%d %H:%M")
            else:
                seen_s = str(seen) if seen is not None else ""
            if title:
                news_lines.append(f"{seen_s} | {src} | {title}")

    filings_lines = []
    for f in (filings or [])[:10]:
        filings_lines.append(f"{f.get('date')} | {f.get('form')} | {f.get('document')}")

    eps_lines = []
    for r in (eps_q or [])[:4]:
        eps_lines.append(f"{r.get('end')} {r.get('fp')} {r.get('fy')} EPS {r.get('val')}")

    rev_lines = []
    for r in (rev_q or [])[:4]:
        rev_lines.append(f"{r.get('end')} {r.get('fp')} {r.get('fy')} Revenue {r.get('val')}")

    pio_lines = []
    for k, v in (pio.get("checks") or {}).items():
        pio_lines.append(f"{k}: {v}")

    moves_lines = []
    for m in (moves or [])[:8]:
        moves_lines.append(f"{m.get('date')} return_pct {m.get('return_pct'):.2f}")

    ctx = f"""
TICKER: {ticker}
COMPANY: {safe_text(company.get('longName') or company.get('shortName'))}
SECTOR: {safe_text(company.get('sector'))}
INDUSTRY: {safe_text(company.get('industry'))}
CEO: {safe_text(company.get('ceo_name'))}
EMPLOYEES: {safe_text(company.get('fullTimeEmployees'))}
MARKET CAP: {safe_text(company.get('marketCap'))}
WEBSITE: {safe_text(company.get('website'))}

SEC NAME: {safe_text(sec_sub.get('name'))}
SIC: {safe_text(sec_sub.get('sic'))}
SIC DESC: {safe_text(sec_sub.get('sicDescription'))}

BUSINESS SUMMARY:
{safe_text(company.get('longBusinessSummary'))}

LAST 4 QUARTERS EPS:
{chr(10).join(eps_lines)}

LAST 4 QUARTERS REVENUE:
{chr(10).join(rev_lines)}

PIOTROSKI:
Score {pio.get('score')}/9
{chr(10).join(pio_lines)}

RECENT SEC FILINGS:
{chr(10).join(filings_lines)}

TOP PRICE MOVES (largest daily moves, last 5y series):
{chr(10).join(moves_lines)}

RECENT NEWS:
{chr(10).join(news_lines)}
"""
    return ctx.strip()


def ask_llm(question: str, context: str):
    if not ollama_available():
        return "Ollama python package is not installed. Run: pip install ollama"

    if not question.strip():
        return "Ask a question."

    sys = """
You are a senior equity research assistant.
Use ONLY the provided context.
If the answer is not in the context, say what is missing and give the closest inference with clear uncertainty.
Be concrete, cite numbers from the context when possible.
Avoid hype and generic lines.
"""

    user = f"""
CONTEXT:
{context}

QUESTION:
{question}

RESPONSE FORMAT:
Answer in 6 to 12 bullet points.
If asked about a price move, connect it to news or filings near that period. If no link exists, say so.
If asked about market share, be explicit if market share is not available in the context and use proxies like revenue scale.
"""

    try:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": sys.strip()},
                {"role": "user", "content": user.strip()},
            ],
        )
        txt = (resp.get("message") or {}).get("content", "").strip()
        txt = txt.replace("<think>", "").replace("</think>", "").strip()
        return txt if txt else "No response."
    except Exception as e:
        return f"LLM error: {type(e).__name__}: {e}"


# -----------------------------
# UI: sidebar
# -----------------------------
with st.sidebar:
    st.header("Mode")
    compare_mode = st.toggle("Compare multiple tickers", value=False)
    st.divider()

    if compare_mode:
        st.caption("Use US equities tickers. ETFs might not have SEC fundamentals.")
        compare_tickers = st.text_area("Tickers (comma separated)", value="AAPL,MSFT,NVDA").strip()
        compare_list = []
        if compare_tickers:
            for t in compare_tickers.split(","):
                tt = t.strip().upper()
                if tt:
                    compare_list.append(tt)
        compare_list = list(dict.fromkeys(compare_list))[:8]
    else:
        st.header("Search")
        ticker = st.text_input("US ticker", value="AAPL").strip().upper()


# -----------------------------
# COMPARE MODE
# -----------------------------
if compare_mode:
    st.subheader("Compare mode")

    if not compare_list:
        st.info("Enter at least one ticker.")
        st.stop()

    tmap = sec_ticker_map()

    rows = []
    for t in compare_list:
        info = yf_profile(t)
        ceo = extract_ceo(info)
        info["ceo_name"] = ceo

        pio_score = None
        if t in tmap:
            try:
                cik = int(tmap[t]["cik"])
                facts = sec_facts(cik)
                pio = compute_piotroski(facts)
                pio_score = pio.get("score")
            except Exception:
                pio_score = None

        perf = basic_perf(t)

        rows.append(
            {
                "Ticker": t,
                "Name": safe_text(info.get("longName") or info.get("shortName")),
                "Sector": safe_text(info.get("sector")),
                "Industry": safe_text(info.get("industry")),
                "CEO": safe_text(ceo),
                "Employees": info.get("fullTimeEmployees"),
                "MarketCap": info.get("marketCap"),
                "Price": perf.get("price"),
                "Return 1Y %": perf.get("ret_1y_pct"),
                "Vol 1Y %": perf.get("vol_1y_pct"),
                "Piotroski": pio_score,
            }
        )

    df = pd.DataFrame(rows)

    def _fmt_num(x, digits=2):
        try:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return "N A"
            return f"{float(x):.{digits}f}"
        except Exception:
            return str(x)

    def _fmt_int(x):
        try:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return "N A"
            return f"{int(x)}"
        except Exception:
            return str(x)

    show = df.copy()
    show["Employees"] = show["Employees"].apply(_fmt_int)
    show["MarketCap"] = show["MarketCap"].apply(format_money)
    show["Price"] = show["Price"].apply(lambda v: _fmt_num(v, 2))
    show["Return 1Y %"] = show["Return 1Y %"].apply(lambda v: _fmt_num(v, 2))
    show["Vol 1Y %"] = show["Vol 1Y %"].apply(lambda v: _fmt_num(v, 2))
    show["Piotroski"] = show["Piotroski"].apply(lambda v: _fmt_int(v))

    st.dataframe(show, use_container_width=True, hide_index=True)

    if ollama_available():
        st.markdown("### Compare analyst note")
        if st.button("Generate AI comparison note"):
            context_lines = []
            for _, r in df.iterrows():
                context_lines.append(
                    f"{r['Ticker']} | sector {r['Sector']} | industry {r['Industry']} | mcap {r['MarketCap']} | ret1y {r['Return 1Y %']} | vol1y {r['Vol 1Y %']} | piotroski {r['Piotroski']}"
                )
            ctx = "\n".join(context_lines)
            q = "Compare these stocks. Who looks strongest on quality and momentum and who looks weakest. Give a quick ranking and one reason per name."
            st.write(ask_llm(q, ctx))
    else:
        st.caption("Install ollama python package if you want AI comparison notes.")

    st.stop()


# -----------------------------
# SINGLE TICKER MODE
# -----------------------------
try:
    if not ticker:
        st.stop()

    tmap = sec_ticker_map()
    if ticker not in tmap:
        st.error("Ticker not found in SEC mapping. It might be an ETF or not an SEC reporting issuer.")
        st.stop()

    cik = int(tmap[ticker]["cik"])
    sub = sec_submissions(cik)
    facts = sec_facts(cik)

    info = yf_profile(ticker)
    info["ceo_name"] = extract_ceo(info)

    name = sub.get("name") or info.get("longName") or tmap[ticker].get("title") or ticker
    sic = sub.get("sic")
    sic_desc = sub.get("sicDescription")

    hist = price_5y(ticker)
    moves = summarize_price_moves(hist, 8)

    pio = compute_piotroski(facts)
    eps_q = last_quarters(facts, "EarningsPerShareDiluted", 4) or last_quarters(facts, "EarningsPerShareBasic", 4)
    rev_q = last_quarters(facts, "Revenues", 4)
    filings = latest_filings_table(sub, cik, 12)

    query = f'"{ticker}" OR "{name}"'
    news = gdelt_news(query, 25)
    if news is None or news.empty:
        news = google_news_rss(f"{ticker} {name}", 25)

    perf = basic_perf(ticker)
    conv = conviction_score_proxy(pio.get("score", 0), perf.get("ret_1y_pct"), perf.get("vol_1y_pct"))

    left, right = st.columns([2, 1])

    with left:
        st.subheader(f"{name} ({ticker})")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Piotroski", f"{pio['score']}/9")
        c2.metric("Conviction", f"{conv}/100")
        c3.metric("Market cap", format_money(info.get("marketCap")))
        c4.metric("Employees", safe_text(info.get("fullTimeEmployees")))
        c5.metric("CEO", safe_text(info.get("ceo_name")))

        st.markdown("### Price (5 years)")
        if not hist.empty:
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=hist["date"],
                        open=hist["Open"],
                        high=hist["High"],
                        low=hist["Low"],
                        close=hist["Close"],
                    )
                ]
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No price data returned.")

        st.markdown("### Company profile")
        p1, p2 = st.columns(2)
        with p1:
            st.write(f"Sector: {safe_text(info.get('sector'))}")
            st.write(f"Industry: {safe_text(info.get('industry'))}")
            st.write(f"Website: {safe_text(info.get('website'))}")
            st.write(f"SIC: {safe_text(sic)}")
            st.write(f"SIC description: {safe_text(sic_desc)}")
        with p2:
            st.write(f"Exchange: {safe_text(info.get('exchange'))}")
            st.write(f"Country: {safe_text(info.get('country'))}")
            st.write(f"City: {safe_text(info.get('city'))}")
            st.write(f"State: {safe_text(info.get('state'))}")
            st.write(f"Currency: {safe_text(info.get('currency'))}")

        if info.get("longBusinessSummary"):
            st.markdown("### Business summary")
            st.write(info.get("longBusinessSummary"))

        st.markdown("### Earnings actuals (SEC XBRL)")
        st.write("EPS last 4 quarters")
        st.dataframe(pd.DataFrame(eps_q), use_container_width=True, hide_index=True)
        st.write("Revenue last 4 quarters")
        st.dataframe(pd.DataFrame(rev_q), use_container_width=True, hide_index=True)

        st.markdown("### Piotroski breakdown")
        st.dataframe(
            pd.DataFrame([{"check": k, "pass": v} for k, v in pio["checks"].items()]),
            use_container_width=True,
            hide_index=True,
        )
        if pio["missing"]:
            st.caption("Missing inputs: " + ", ".join(sorted(set(pio["missing"]))))

    with right:
        tabs = st.tabs(["Intelligence", "Chat", "Peers"])

        with tabs[0]:
            st.markdown("SEC filings")
            st.dataframe(pd.DataFrame(filings), use_container_width=True, hide_index=True)

            st.markdown("News")
            if news is None or news.empty:
                st.info("No news returned right now.")
            else:
                for _, r in news.head(12).iterrows():
                    seen = r.get("seen")
                    if isinstance(seen, str):
                        seen_s = seen
                    else:
                        seen_s = seen.strftime("%Y-%m-%d %H:%M") if pd.notna(seen) else ""
                    title = r.get("title", "Untitled")
                    src = r.get("source", "")
                    url = r.get("url", "")
                    st.markdown(f"- {seen_s} **{src}** [{title}]({url})")

            st.markdown("Largest daily moves")
            if moves:
                st.dataframe(pd.DataFrame(moves), use_container_width=True, hide_index=True)
            else:
                st.caption("No moves computed.")

        with tabs[1]:
            st.markdown("### Analyst chat")
            if not ollama_available():
                st.error("Ollama python package not found. Run: pip install ollama")
            else:
                st.caption(f"Model: {OLLAMA_MODEL}")

                ctx = build_llm_context(
                    ticker=ticker,
                    company=info,
                    sec_sub=sub,
                    pio=pio,
                    eps_q=eps_q,
                    rev_q=rev_q,
                    filings=filings,
                    news_df=news,
                    moves=moves,
                )

                chat_key = f"chat_{ticker}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []

                for msg in st.session_state[chat_key]:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

                q = st.chat_input(f"Ask anything about {ticker}")
                if q:
                    st.session_state[chat_key].append({"role": "user", "content": q})
                    with st.chat_message("user"):
                        st.write(q)

                    with st.chat_message("assistant"):
                        ans = ask_llm(q, ctx)
                        st.write(ans)

                    st.session_state[chat_key].append({"role": "assistant", "content": ans})

                with st.expander("Show model context used", expanded=False):
                    st.code(ctx)

        with tabs[2]:
            st.markdown("### Competitors and peers")
            st.caption("Peers are inferred using sector, industry, and business summary. Market share is not reliably available from free structured feeds, so we use proxies like size and returns.")

            peers = infer_competitors_with_llm(
                ticker=ticker,
                name=safe_text(name),
                sector=safe_text(info.get("sector")),
                industry=safe_text(info.get("industry")),
                summary=safe_text(info.get("longBusinessSummary"), ""),
            )

            if not peers:
                st.info("No peers inferred. Try again or ask in chat: list competitors for this company.")
            else:
                peer_rows = []
                for p in peers:
                    pinfo = yf_profile(p)
                    pperf = basic_perf(p)
                    peer_rows.append(
                        {
                            "Ticker": p,
                            "Name": safe_text(pinfo.get("longName") or pinfo.get("shortName")),
                            "MarketCap": format_money(pinfo.get("marketCap")),
                            "Price": pperf.get("price"),
                            "Return 1Y %": pperf.get("ret_1y_pct"),
                            "Vol 1Y %": pperf.get("vol_1y_pct"),
                            "Sector": safe_text(pinfo.get("sector")),
                            "Industry": safe_text(pinfo.get("industry")),
                        }
                    )
                pdf = pd.DataFrame(peer_rows)

                def _fmt(v, digits=2):
                    try:
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            return "N A"
                        return f"{float(v):.{digits}f}"
                    except Exception:
                        return str(v)

                show = pdf.copy()
                show["Price"] = show["Price"].apply(lambda v: _fmt(v, 2))
                show["Return 1Y %"] = show["Return 1Y %"].apply(lambda v: _fmt(v, 2))
                show["Vol 1Y %"] = show["Vol 1Y %"].apply(lambda v: _fmt(v, 2))
                st.dataframe(show, use_container_width=True, hide_index=True)

                if ollama_available():
                    if st.button("Ask AI to summarize competition landscape"):
                        ctx = f"""
Company: {name} ({ticker})
Sector: {safe_text(info.get('sector'))}
Industry: {safe_text(info.get('industry'))}
Business summary: {safe_text(info.get('longBusinessSummary'))}

Peers:
{show.to_string(index=False)}
"""
                        q = "Explain the competitive landscape. Who are the closest direct competitors. What are the key battleground products or segments. If market share is unknown, say so and use proxies."
                        st.write(ask_llm(q, ctx))

except Exception:
    st.error("App crashed. Here is the error:")
    st.code(traceback.format_exc())
