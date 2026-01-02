import os
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

import sources_universe as universe_src
import sources_sec as sec_src
import sources_gdelt as gdelt_src
import sources_profile as profile_src
import features_piotroski as pio_mod
import features_scoring as score_mod

# SEC asks for a real user agent with contact
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "AlphaMindResearch/1.0 (contact: Rajan2shukla@gmail.com)")

session = requests.Session()
session.headers.update({"User-Agent": SEC_USER_AGENT})

st.set_page_config(page_title="Brahmin Terminal", layout="wide")
st.title("Brahmin Terminal")
st.caption("SEC filings + SEC XBRL facts + GDELT news + 5 year price. Free sources only.")

@st.cache_data(ttl=6 * 60 * 60)
def load_universe(include_etfs: bool, include_test_issues: bool) -> pd.DataFrame:
    df = universe_src.get_us_listed_universe(session)
    return universe_src.filter_universe(df, include_etfs, include_test_issues)

@st.cache_data(ttl=6 * 60 * 60)
def load_sec_ticker_map() -> dict:
    return sec_src.get_sec_ticker_map(session)

@st.cache_data(ttl=20 * 60)
def load_sec_data(cik: int) -> tuple[dict, dict]:
    sub = sec_src.get_company_submissions(session, cik)
    facts = sec_src.get_company_facts(session, cik)
    return sub, facts

@st.cache_data(ttl=10 * 60)
def load_prices_5y(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    hist = tk.history(period="5y", auto_adjust=False)
    if hist is None or hist.empty:
        return pd.DataFrame()
    hist = hist.reset_index()
    # normalize for plotting
    if "Date" in hist.columns:
        hist = hist.rename(columns={"Date": "date"})
    elif "Datetime" in hist.columns:
        hist = hist.rename(columns={"Datetime": "date"})
    return hist

@st.cache_data(ttl=10 * 60)
def load_gdelt_news(query: str) -> pd.DataFrame:
    return gdelt_src.gdelt_doc_search(session, query=query, max_records=25)

# Sidebar
st.sidebar.header("Universe")
include_etfs = st.sidebar.checkbox("Include ETFs", value=False)
include_test = st.sidebar.checkbox("Include test issues", value=False)

df_u = load_universe(include_etfs, include_test)
st.sidebar.caption(f"Universe size: {len(df_u):,}")

st.sidebar.header("Ticker")
ticker = st.sidebar.text_input("US ticker", value="AAPL").strip().upper()

if not ticker:
    st.stop()

# Resolve to CIK using SEC ticker map
tmap = load_sec_ticker_map()
if ticker not in tmap:
    st.error("Ticker not found in SEC mapping. It may be an ETF, warrant, or non reporting issue.")
    st.stop()

cik = int(tmap[ticker]["cik"])

# Pull SEC and price data
sub, facts = load_sec_data(cik)
hist = load_prices_5y(ticker)

# Profile best effort (CEO, employees, sector)
profile = profile_src.get_profile_best_effort(ticker)

company_name = sub.get("name") or profile.get("name") or tmap[ticker].get("title") or ticker
sic = sub.get("sic")
sic_desc = sub.get("sicDescription")

# Piotroski
class Helpers:
    latest_annual_two = staticmethod(sec_src.latest_annual_two)

pio = pio_mod.compute_piotroski(facts, Helpers)

# Conviction score
conv = score_mod.conviction_score(pio["score"], hist if "Close" in hist.columns else pd.DataFrame())

# SEC filings list
filings_rows = sec_src.latest_filings_table(sub, cik, limit=15)

# SEC quarterly EPS and Revenue
eps_q = sec_src.last_quarters(facts, "EarningsPerShareDiluted", n=4)
if not eps_q:
    eps_q = sec_src.last_quarters(facts, "EarningsPerShareBasic", n=4)
rev_q = sec_src.last_quarters(facts, "Revenues", n=4)

# News query (ticker + company name)
news_query = f'"{ticker}" OR "{company_name}"'
news_df = pd.DataFrame()
try:
    news_df = load_gdelt_news(news_query)
except Exception:
    news_df = pd.DataFrame()

# Layout
left, right = st.columns([2, 1])

with left:
    st.subheader(f"{company_name} ({ticker})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CIK", str(cik))
    c2.metric("SIC", str(sic) if sic else "N A")
    c3.metric("Industry", sic_desc if sic_desc else "N A")
    c4.metric("Piotroski", f"{pio['score']}/9")

    st.markdown("### Company profile (best effort)")
    pcols = st.columns(3)
    pcols[0].write(f"**CEO:** {profile.get('ceo') or 'N A'}")
    pcols[1].write(f"**Employees:** {profile.get('employees') or 'N A'}")
    pcols[2].write(f"**Sector:** {profile.get('sector') or 'N A'}")
    st.write(f"**Industry:** {profile.get('industry') or 'N A'}")
    if profile.get("website"):
        st.write(f"**Website:** {profile.get('website')}")
    if profile.get("summary"):
        st.write(profile.get("summary"))

    st.markdown("### Conviction score")
    st.metric("Conviction (0 to 100)", conv.get("score", 0))
    parts = conv.get("parts", {})
    st.caption(f"Quality {parts.get('quality',0)}  Momentum {parts.get('momentum',0)}  Risk {parts.get('risk',0)}")
    if conv.get("ret_12m") is not None:
        st.write(f"12 month return: {conv['ret_12m']*100:.1f}%")
    if conv.get("max_drawdown") is not None:
        st.write(f"Max drawdown in window: {conv['max_drawdown']*100:.1f}%")
    if conv.get("rsi") is not None:
        st.write(f"RSI: {conv['rsi']:.1f}")

    st.markdown("### Price chart (5 years)")
    if hist is None or hist.empty:
        st.warning("No 5 year price data returned.")
    else:
        fig = go.Figure(
            data=[go.Candlestick(
                x=hist["date"],
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                name=ticker
            )]
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Earnings actuals (SEC XBRL)")
    ecols = st.columns(2)
    with ecols[0]:
        st.write("**EPS last 4 quarters**")
        st.dataframe(pd.DataFrame(eps_q), use_container_width=True, hide_index=True)
    with ecols[1]:
        st.write("**Revenue last 4 quarters**")
        st.dataframe(pd.DataFrame(rev_q), use_container_width=True, hide_index=True)

    st.markdown("### Next earnings")
    st.write(profile.get("next_earnings_date") or "Next earnings date not available via free sources consistently.")

    st.markdown("### Piotroski breakdown")
    checks_df = pd.DataFrame([{"check": k, "pass": v} for k, v in pio["checks"].items()])
    st.dataframe(checks_df, use_container_width=True, hide_index=True)
    if pio["missing"]:
        st.caption("Missing inputs: " + ", ".join(sorted(set(pio["missing"]))))

    st.markdown("### Competitors and peer comparison")
    st.caption("Free day one version: you define peers. Tool computes side by side performance and proxy share within the peer set.")
    peers_text = st.text_input("Peer tickers separated by commas", value="")
    peers = [p.strip().upper() for p in peers_text.split(",") if p.strip()]
    peers = [p for p in peers if p and p != ticker]
    peer_set = [ticker] + peers

    if len(peer_set) >= 2:
        rows = []
        for t in peer_set:
            try:
                h = yf.Ticker(t).history(period="1y")
                last = float(h["Close"].iloc[-1]) if h is not None and not h.empty else None
                ret_1y = (float(h["Close"].iloc[-1]) / float(h["Close"].iloc[0]) - 1) if h is not None and not h.empty else None
            except Exception:
                last = None
                ret_1y = None

            rows.append({"ticker": t, "last_price": last, "return_1y": ret_1y})

        peer_df = pd.DataFrame(rows)

        # proxy market share using revenue within peer set when available
        peer_revs = []
        for t in peer_set:
            rev = None
            if t in tmap:
                try:
                    cik_p = int(tmap[t]["cik"])
                    _, facts_p = load_sec_data(cik_p)
                    rev_two = sec_src.latest_annual_two(facts_p, "Revenues")
                    if rev_two and rev_two.get("curr") is not None:
                        rev = float(rev_two["curr"])
                except Exception:
                    rev = None
            peer_revs.append({"ticker": t, "annual_revenue": rev})

        rev_df = pd.DataFrame(peer_revs)
        merged = peer_df.merge(rev_df, on="ticker", how="left")
        total_rev = merged["annual_revenue"].dropna().sum()
        if total_rev and total_rev > 0:
            merged["proxy_revenue_share_in_peer_set"] = merged["annual_revenue"] / total_rev
        else:
            merged["proxy_revenue_share_in_peer_set"] = None

        st.dataframe(merged, use_container_width=True, hide_index=True)
    else:
        st.info("Add at least 1 peer ticker to compare.")

with right:
    st.subheader("Real time intelligence")

    st.markdown("### SEC filings (hard news)")
    st.dataframe(pd.DataFrame(filings_rows), use_container_width=True, hide_index=True)

    st.markdown("### News stream (GDELT)")
    if news_df is None or news_df.empty:
        st.info("No news returned right now. Try again or adjust query.")
    else:
        for _, r in news_df.head(12).iterrows():
            seen = r.get("seen")
            seen_s = seen.strftime("%Y-%m-%d %H:%M") if pd.notna(seen) else ""
            title = r.get("title") or "Untitled"
            src = r.get("source") or ""
            url = r.get("url") or ""
            st.markdown(f"- {seen_s}  **{src}**  [{title}]({url})")

st.divider()
st.caption("Core sources: SEC data APIs on data.sec.gov, Nasdaq Trader symbol directory, GDELT DOC API.")
