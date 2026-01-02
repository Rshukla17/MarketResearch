import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from io import StringIO

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="Alpha Mind Terminal", layout="wide", page_icon="🏛️")

session = requests.Session()
session.headers.update(
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
)

# =============================================================================
# OPTIONAL OLLAMA
# =============================================================================
OLLAMA_AVAILABLE = False
try:
    import ollama  # type: ignore
    ollama.list()
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

# =============================================================================
# STREAMLIT COMPAT HELPERS (width vs use_container_width)
# =============================================================================
def st_df(df: pd.DataFrame):
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)

def st_plot(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# SMALL UTILS
# =============================================================================
def norm_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    return t.replace(".", "-")

def safe_num(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, float) and np.isnan(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def clamp01(x) -> float:
    x = safe_num(x, 0.0)
    if np.isnan(x) or np.isinf(x):
        return 0.0
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return float(x)

def get_two_periods(df: pd.DataFrame):
    if df is None or df.empty or df.shape[1] == 0:
        return None, None
    cols = list(df.columns)
    current = cols[0]
    prior = cols[1] if len(cols) > 1 else cols[0]
    return current, prior

# =============================================================================
# DATA PULL (CACHE AS RESOURCE TO AVOID PICKLE ERRORS)
# =============================================================================
@st.cache_resource(ttl=7200, show_spinner=False)
def get_enriched_stock_data(ticker: str) -> dict:
    t = norm_ticker(ticker)
    stock = yf.Ticker(t, session=session)

    try:
        info = stock.info or {}
    except Exception:
        info = {}

    def safe_df(x):
        return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

    try:
        financials = safe_df(stock.financials)
    except Exception:
        financials = pd.DataFrame()

    try:
        balance_sheet = safe_df(stock.balance_sheet)
    except Exception:
        balance_sheet = pd.DataFrame()

    try:
        cashflow = safe_df(stock.cashflow)
    except Exception:
        cashflow = pd.DataFrame()

    try:
        news = stock.news or []
    except Exception:
        news = []

    try:
        hist = stock.history(period="1y")
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)
    except Exception:
        hist = pd.DataFrame()

    return {
        "ticker": t,
        "info": info,
        "financials": financials,
        "balance_sheet": balance_sheet,
        "cashflow": cashflow,
        "news": news,
        "hist": hist,
    }

# =============================================================================
# FUNDAMENTALS
# =============================================================================
def calculate_piotroski_score(financials: pd.DataFrame, balance_sheet: pd.DataFrame, cashflow: pd.DataFrame) -> int | None:
    try:
        if financials.empty or balance_sheet.empty or cashflow.empty:
            return None

        score = 0
        cur_f, prior_f = get_two_periods(financials)
        cur_b, prior_b = get_two_periods(balance_sheet)
        cur_c, prior_c = get_two_periods(cashflow)

        if cur_f is None or cur_b is None or cur_c is None:
            return None

        # Profitability (4)
        net_income_cur = safe_num(financials.loc["Net Income", cur_f], 0) if "Net Income" in financials.index else 0
        total_assets_cur = safe_num(balance_sheet.loc["Total Assets", cur_b], 0) if "Total Assets" in balance_sheet.index else 0
        if total_assets_cur > 0 and net_income_cur > 0:
            score += 1

        cfo_cur = safe_num(cashflow.loc["Operating Cash Flow", cur_c], 0) if "Operating Cash Flow" in cashflow.index else 0
        if cfo_cur > 0:
            score += 1

        # ROA increase
        try:
            net_income_prior = safe_num(financials.loc["Net Income", prior_f], 0) if "Net Income" in financials.index else 0
            total_assets_prior = safe_num(balance_sheet.loc["Total Assets", prior_b], 0) if "Total Assets" in balance_sheet.index else 0
            if total_assets_cur > 0 and total_assets_prior > 0:
                roa_cur = net_income_cur / total_assets_cur
                roa_prior = net_income_prior / total_assets_prior
                if roa_cur > roa_prior:
                    score += 1
        except Exception:
            pass

        # CFO > NI
        if cfo_cur > net_income_cur:
            score += 1

        # Leverage + liquidity + dilution (3)
        try:
            if "Long Term Debt" in balance_sheet.index:
                debt_cur = safe_num(balance_sheet.loc["Long Term Debt", cur_b], 0)
                debt_prior = safe_num(balance_sheet.loc["Long Term Debt", prior_b], 0)
                if debt_cur < debt_prior:
                    score += 1
        except Exception:
            pass

        try:
            if "Current Assets" in balance_sheet.index and "Current Liabilities" in balance_sheet.index:
                ca_cur = safe_num(balance_sheet.loc["Current Assets", cur_b], 0)
                cl_cur = safe_num(balance_sheet.loc["Current Liabilities", cur_b], 0)
                ca_prior = safe_num(balance_sheet.loc["Current Assets", prior_b], 0)
                cl_prior = safe_num(balance_sheet.loc["Current Liabilities", prior_b], 0)
                if cl_cur > 0 and cl_prior > 0:
                    cr_cur = ca_cur / cl_cur
                    cr_prior = ca_prior / cl_prior
                    if cr_cur > cr_prior:
                        score += 1
        except Exception:
            pass

        # No dilution (best effort)
        diluted_point = 0
        try:
            if "Ordinary Shares Number" in balance_sheet.index:
                sh_cur = safe_num(balance_sheet.loc["Ordinary Shares Number", cur_b], np.nan)
                sh_prior = safe_num(balance_sheet.loc["Ordinary Shares Number", prior_b], np.nan)
                if not np.isnan(sh_cur) and not np.isnan(sh_prior) and sh_cur <= sh_prior:
                    diluted_point = 1
        except Exception:
            pass
        score += diluted_point

        # Operating efficiency (2)
        try:
            if "Gross Profit" in financials.index and "Total Revenue" in financials.index:
                gp_cur = safe_num(financials.loc["Gross Profit", cur_f], 0)
                rev_cur = safe_num(financials.loc["Total Revenue", cur_f], 0)
                gp_prior = safe_num(financials.loc["Gross Profit", prior_f], 0)
                rev_prior = safe_num(financials.loc["Total Revenue", prior_f], 0)
                if rev_cur > 0 and rev_prior > 0:
                    gm_cur = gp_cur / rev_cur
                    gm_prior = gp_prior / rev_prior
                    if gm_cur > gm_prior:
                        score += 1
        except Exception:
            pass

        try:
            if "Total Revenue" in financials.index and "Total Assets" in balance_sheet.index:
                rev_cur = safe_num(financials.loc["Total Revenue", cur_f], 0)
                rev_prior = safe_num(financials.loc["Total Revenue", prior_f], 0)
                ta_cur = safe_num(balance_sheet.loc["Total Assets", cur_b], 0)
                ta_prior = safe_num(balance_sheet.loc["Total Assets", prior_b], 0)
                if ta_cur > 0 and ta_prior > 0:
                    at_cur = rev_cur / ta_cur
                    at_prior = rev_prior / ta_prior
                    if at_cur > at_prior:
                        score += 1
        except Exception:
            pass

        return int(score)
    except Exception:
        return None

def calculate_magic_formula_metrics(info: dict) -> dict:
    try:
        ebit = safe_num(info.get("ebit"), 0)
        enterprise_value = safe_num(info.get("enterpriseValue"), 0)
        earnings_yield = (ebit / enterprise_value * 100) if enterprise_value > 0 else 0

        total_assets = safe_num(info.get("totalAssets"), 0)
        roic = (ebit / total_assets * 100) if total_assets > 0 else 0

        return {
            "earnings_yield": earnings_yield,
            "roic": roic,
            "magic_score": 0.5 * earnings_yield + 0.5 * roic,
        }
    except Exception:
        return {"earnings_yield": 0, "roic": 0, "magic_score": 0}

def calculate_financial_health(info: dict) -> dict:
    metrics = {}

    fcf = safe_num(info.get("freeCashflow"), 0)
    market_cap = safe_num(info.get("marketCap"), 0)
    metrics["fcf_yield"] = (fcf / market_cap * 100) if market_cap > 0 else 0

    total_debt = safe_num(info.get("totalDebt"), 0)
    total_equity = safe_num(info.get("totalStockholderEquity"), 0)
    metrics["debt_to_equity"] = (total_debt / total_equity) if total_equity > 0 else 0

    roa = info.get("returnOnAssets", None)
    roe = info.get("returnOnEquity", None)
    metrics["roa"] = safe_num(roa, 0) * 100 if roa is not None else 0
    metrics["roe"] = safe_num(roe, 0) * 100 if roe is not None else 0

    metrics["pe"] = safe_num(info.get("trailingPE"), 0)
    metrics["peg"] = safe_num(info.get("pegRatio"), 0)
    metrics["ev_ebitda"] = safe_num(info.get("enterpriseToEbitda"), 0)

    return metrics

def calculate_technical_indicators(hist_data: pd.DataFrame) -> dict:
    if hist_data is None or hist_data.empty or "Close" not in hist_data.columns:
        return {}

    close_prices = hist_data["Close"].dropna()
    if close_prices.empty:
        return {}

    ma_50 = close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else close_prices.mean()
    ma_200 = close_prices.rolling(200).mean().iloc[-1] if len(close_prices) >= 200 else close_prices.mean()

    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_last = safe_num(rsi.iloc[-1], 50) if not rsi.empty else 50

    cumulative = (1 + close_prices.pct_change().fillna(0)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max.replace(0, np.nan)
    max_drawdown = safe_num(drawdown.min(), 0) * 100

    return {
        "ma_50": float(ma_50),
        "ma_200": float(ma_200),
        "rsi": float(rsi_last),
        "max_drawdown": abs(float(max_drawdown)),
        "current_price": float(close_prices.iloc[-1]),
        "golden_cross": bool(ma_50 > ma_200),
    }

def calculate_conviction_score(piotroski: int | None, financial_metrics: dict, magic_formula: dict, technical: dict) -> float:
    score = 50.0

    if piotroski is not None:
        score += (piotroski - 5) * 3

    fcf_yield = safe_num(financial_metrics.get("fcf_yield"), 0)
    if fcf_yield > 5:
        score += 10
    elif fcf_yield > 3:
        score += 5

    peg = safe_num(financial_metrics.get("peg"), 999)
    if 0 < peg < 1.5:
        score += 10
    elif 0 < peg < 2:
        score += 5

    dte = safe_num(financial_metrics.get("debt_to_equity"), 999)
    if dte < 0.5:
        score += 5
    elif dte < 1:
        score += 2

    magic_score = safe_num(magic_formula.get("magic_score"), 0)
    if magic_score > 15:
        score += 10
    elif magic_score > 10:
        score += 5

    if technical.get("golden_cross", False):
        score += 5

    rsi = safe_num(technical.get("rsi"), 50)
    if 40 < rsi < 60:
        score += 5

    return float(min(100, max(0, score)))

def generate_ai_analysis(ticker: str, info: dict, conviction_score: float, financial_metrics: dict, piotroski: int | None) -> dict:
    if conviction_score > 80:
        rating, color = "STRONG BUY", "🟢"
    elif conviction_score > 65:
        rating, color = "BUY", "🟢"
    elif conviction_score > 50:
        rating, color = "HOLD", "🟡"
    elif conviction_score > 35:
        rating, color = "REDUCE", "🟠"
    else:
        rating, color = "SELL", "🔴"

    kelly_percentage = (conviction_score / 100) * 5
    sector = info.get("sector", "Unknown")
    fcf_yield = safe_num(financial_metrics.get("fcf_yield"), 0)
    roe = safe_num(financial_metrics.get("roe"), 0)
    dte = safe_num(financial_metrics.get("debt_to_equity"), 0)

    ptxt = f"{piotroski}/9" if piotroski is not None else "N/A"
    health_word = "strong" if (piotroski is not None and piotroski >= 7) else "moderate"

    thesis = (
        f"{ticker} shows {health_word} financial quality with a Piotroski F Score of {ptxt}. "
        f"FCF yield is {fcf_yield:.2f}% which is {'excellent' if fcf_yield > 5 else 'solid'} for cash generation."
    )

    catalyst = (
        f"{sector} positioning plus {'strong' if roe > 15 else 'steady'} ROE ({roe:.1f}%) "
        "supports the operating story if fundamentals hold."
    )

    if dte > 1:
        risk = f"Primary concern: debt to equity is elevated at {dte:.2f}, which amplifies downside in stress."
    else:
        risk = "Primary concern: market regime risk and sector crowding. Size the position like you can be wrong."

    current_price = safe_num(info.get("currentPrice"), safe_num(info.get("regularMarketPrice"), 0))
    support_level = current_price * 0.95 if current_price > 0 else 0
    stop_loss = current_price * 0.925 if current_price > 0 else 0

    return {
        "conviction_score": conviction_score,
        "rating": rating,
        "color": color,
        "kelly_position": kelly_percentage,
        "thesis": thesis,
        "catalyst": catalyst,
        "risk": risk,
        "entry_price": current_price,
        "support_level": support_level,
        "stop_loss": stop_loss,
    }

# =============================================================================
# UNIVERSE
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        r = session.get(url, timeout=15)
        tables = pd.read_html(StringIO(r.text))
        symbols = tables[0]["Symbol"].tolist()
        return [norm_ticker(x) for x in symbols]
    except Exception:
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "V", "JNJ", "WMT", "JPM", "MA", "PG", "UNH", "HD", "DIS", "BAC",
            "XOM", "ABBV",
        ]

def scan_stocks(tickers: list[str], max_stocks: int = 50) -> pd.DataFrame:
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    universe = tickers[:max_stocks]
    for i, t in enumerate(universe):
        status_text.text(f"Scanning {t} ({i+1}/{len(universe)})")
        progress_bar.progress(clamp01((i + 1) / max(1, len(universe))))

        try:
            data = get_enriched_stock_data(t)
            info = data["info"]
            if not info or info.get("marketCap") is None:
                continue

            fcf = safe_num(info.get("freeCashflow"), 0)
            mcap = safe_num(info.get("marketCap"), 0)
            fcf_yield = (fcf / mcap * 100) if mcap > 0 else 0

            pe = safe_num(info.get("trailingPE"), 0)

            dte_raw = info.get("debtToEquity", None)
            dte = safe_num(dte_raw, 999)
            if dte > 20:
                dte = dte / 100

            if fcf_yield > 3 and 0 < pe < 30 and dte < 1.5:
                p = calculate_piotroski_score(data["financials"], data["balance_sheet"], data["cashflow"])
                if p is not None and p >= 7:
                    results.append(
                        {
                            "Ticker": norm_ticker(t),
                            "Price": safe_num(info.get("currentPrice"), 0),
                            "FCF Yield %": fcf_yield,
                            "P/E": pe,
                            "Debt/Equity": dte,
                            "Piotroski": p,
                            "Sector": info.get("sector", "N/A"),
                        }
                    )

            time.sleep(0.05)
        except Exception:
            continue

    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

def compare_stocks(ticker_list: list[str]) -> pd.DataFrame:
    rows = []
    for t in ticker_list:
        try:
            data = get_enriched_stock_data(t)
            info = data["info"]
            if not info or info.get("marketCap") is None:
                continue

            fin = calculate_financial_health(info)
            p = calculate_piotroski_score(data["financials"], data["balance_sheet"], data["cashflow"])

            rows.append(
                {
                    "Ticker": norm_ticker(t),
                    "Price": safe_num(info.get("currentPrice"), 0),
                    "P/E": safe_num(fin.get("pe"), 0),
                    "PEG": safe_num(fin.get("peg"), 0),
                    "FCF Yield %": safe_num(fin.get("fcf_yield"), 0),
                    "Debt/Equity": safe_num(fin.get("debt_to_equity"), 0),
                    "ROE %": safe_num(fin.get("roe"), 0),
                    "Piotroski": p if p is not None else 0,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)

# =============================================================================
# BACKTEST (TZ SAFE)
# =============================================================================
def backtest_strategy(ticker: str, months_back: int = 6) -> dict | None:
    try:
        t = norm_ticker(ticker)
        data = get_enriched_stock_data(t)
        hist = data["hist"]

        if hist is None or hist.empty or "Close" not in hist.columns:
            return None

        close = hist["Close"].dropna()
        if close.empty or len(close) < 2:
            return None

        end = close.index.max()
        if not isinstance(end, pd.Timestamp):
            return None

        lookback = end - pd.DateOffset(months=months_back)
        close_window = close.loc[close.index >= lookback]
        if close_window.empty or len(close_window) < 2:
            return None

        entry_price = float(close_window.iloc[0])
        current_price = float(close_window.iloc[-1])
        returns = ((current_price - entry_price) / entry_price) * 100

        spy_data = get_enriched_stock_data("SPY")
        spy_hist = spy_data["hist"]
        spy_returns = 0.0

        if spy_hist is not None and not spy_hist.empty and "Close" in spy_hist.columns:
            spy_close = spy_hist["Close"].dropna()
            if not spy_close.empty and len(spy_close) >= 2:
                spy_end = spy_close.index.max()
                spy_lookback = spy_end - pd.DateOffset(months=months_back)
                spy_window = spy_close.loc[spy_close.index >= spy_lookback]
                if not spy_window.empty and len(spy_window) >= 2:
                    spy_entry = float(spy_window.iloc[0])
                    spy_current = float(spy_window.iloc[-1])
                    spy_returns = ((spy_current - spy_entry) / spy_entry) * 100

        return {
            "ticker": t,
            "entry_price": entry_price,
            "current_price": current_price,
            "returns": returns,
            "spy_returns": spy_returns,
            "alpha": returns - spy_returns,
        }
    except Exception:
        return None

# =============================================================================
# APP
# =============================================================================
def main():
    st.markdown(
        """
        <style>
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       padding: 20px; border-radius: 10px; color: white; }
        .conviction-high { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
        .conviction-medium { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); }
        .conviction-low { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h1 style='text-align: center; color: #667eea;'>🏛️ Brahman Terminal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Astrology for Stock Market</p>", unsafe_allow_html=True)

    st.sidebar.title("⚙️ Control Panel")
    mode = st.sidebar.radio("Mode", ["Single Stock Analysis", "Compare Stocks", "Stock Scanner", "Backtest"])

    if mode == "Single Stock Analysis":
        ticker_input = norm_ticker(st.sidebar.text_input("Enter US Ticker", value="AAPL"))

        if st.sidebar.button("🚀 Analyze Stock", type="primary"):
            with st.spinner(f"Analyzing {ticker_input}..."):
                data = get_enriched_stock_data(ticker_input)
                info = data["info"]

                if not info or info.get("marketCap") is None:
                    st.error("Unable to fetch data. Check the ticker symbol, then try again.")
                    return

                financial_metrics = calculate_financial_health(info)
                piotroski = calculate_piotroski_score(data["financials"], data["balance_sheet"], data["cashflow"])
                magic_formula = calculate_magic_formula_metrics(info)
                technical = calculate_technical_indicators(data["hist"])
                conviction = calculate_conviction_score(piotroski, financial_metrics, magic_formula, technical)
                analysis = generate_ai_analysis(ticker_input, info, conviction, financial_metrics, piotroski)

                st.session_state["bundle"] = {
                    "analysis": analysis,
                    "financial_metrics": financial_metrics,
                    "piotroski": piotroski,
                    "magic_formula": magic_formula,
                    "technical": technical,
                    "info": info,
                    "hist": data["hist"],
                    "news": data["news"],
                    "ticker": ticker_input,
                }

        if "bundle" in st.session_state:
            b = st.session_state["bundle"]
            analysis = b["analysis"]
            financial_metrics = b["financial_metrics"]
            piotroski = b["piotroski"]
            magic_formula = b["magic_formula"]
            technical = b["technical"]
            info = b["info"]
            hist = b["hist"]
            news = b["news"]

            conviction_class = (
                "conviction-high" if analysis["conviction_score"] > 70 else
                "conviction-medium" if analysis["conviction_score"] > 50 else
                "conviction-low"
            )

            st.markdown(
                f"""
                <div class='metric-card {conviction_class}' style='margin: 20px 0;'>
                    <h2 style='margin:0;'>{analysis['color']} CONVICTION SCORE: {analysis['conviction_score']:.0f}/100</h2>
                    <h3 style='margin:10px 0 0 0;'>{analysis['rating']}</h3>
                    <p style='margin:5px 0 0 0;'>Recommended Position Size: {analysis['kelly_position']:.2f}% (Kelly)</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💰 Fundamentals", "📈 Technical", "🤖 AI Verdict"])

            with tab1:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Current Price",
                        f"${safe_num(info.get('currentPrice'), 0):.2f}",
                        f"{safe_num(info.get('regularMarketChangePercent'), 0):.2f}%",
                    )

                with col2:
                    pe = safe_num(financial_metrics.get("pe"), 0)
                    st.metric("P/E Ratio", f"{pe:.2f}", "Strong" if 0 < pe < 20 else "High" if pe < 30 else "Elevated")

                with col3:
                    fcfy = safe_num(financial_metrics.get("fcf_yield"), 0)
                    st.metric("FCF Yield", f"{fcfy:.2f}%", "Excellent" if fcfy > 5 else "Good" if fcfy > 3 else "Low")

                with col4:
                    st.metric("Beta", f"{safe_num(info.get('beta'), 0):.2f}")

                st.markdown("---")
                col_left, col_right = st.columns([2, 1])

                with col_left:
                    st.subheader("📊 Price Momentum (1 Year)")
                    if hist is not None and not hist.empty and all(c in hist.columns for c in ["Open", "High", "Low", "Close"]):
                        fig = go.Figure(
                            data=[
                                go.Candlestick(
                                    x=hist.index,
                                    open=hist["Open"],
                                    high=hist["High"],
                                    low=hist["Low"],
                                    close=hist["Close"],
                                    name="Price",
                                )
                            ]
                        )
                        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(50).mean(), name="50 Day MA"))
                        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(200).mean(), name="200 Day MA"))
                        fig.update_layout(
                            height=400,
                            template="plotly_dark",
                            xaxis_rangeslider_visible=False,
                            margin=dict(l=0, r=0, t=0, b=0),
                        )
                        st_plot(fig)
                    else:
                        st.info("No price history available.")

                with col_right:
                    st.subheader("📰 Latest Intelligence")
                    if news:
                        for item in news[:5]:
                            title = item.get("title", "No title")
                            publisher = item.get("publisher", "Unknown")
                            link = item.get("link", None)

                            st.markdown(f"**{publisher}**")
                            if link:
                                st.markdown(f"[{title}]({link})")
                            else:
                                st.write(title)
                            st.markdown("---")
                    else:
                        st.info("No recent news available.")

            with tab2:
                st.subheader("Quality Scores")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Piotroski F Score",
                        f"{piotroski}/9" if piotroski is not None else "N/A",
                        "Strong" if (piotroski is not None and piotroski >= 7) else "Weak",
                    )
                    st.progress(clamp01((piotroski / 9) if piotroski is not None else 0.0))
                    st.caption("Financial strength indicator")

                with col2:
                    magic_rank = safe_num(magic_formula.get("magic_score"), 0)
                    st.metric("Magic Formula Score", f"{magic_rank:.1f}")
                    st.progress(clamp01(magic_rank / 30))
                    st.caption("Quality and value proxy")

                with col3:
                    peg = safe_num(financial_metrics.get("peg"), 0)
                    st.metric(
                        "PEG Ratio",
                        f"{peg:.2f}" if peg > 0 else "N/A",
                        "Attractive" if 0 < peg < 1.5 else "Fair" if 0 < peg < 2 else "Rich",
                    )
                    # lower peg is better. clamp for negatives
                    st.progress(clamp01(1 - min(max(peg, 0) / 3, 1)) if peg > 0 else 0.0)
                    st.caption("Growth adjusted valuation")

                st.markdown("---")
                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader("💪 Profitability")
                    roa = safe_num(financial_metrics.get("roa"), 0)
                    roe = safe_num(financial_metrics.get("roe"), 0)

                    st.metric("Return on Assets", f"{roa:.2f}%")
                    st.progress(clamp01(max(roa, 0) / 25))

                    st.metric("Return on Equity", f"{roe:.2f}%")
                    st.progress(clamp01(max(roe, 0) / 30))

                with col_right:
                    st.subheader("💵 Valuation")
                    ev_ebitda = safe_num(financial_metrics.get("ev_ebitda"), 0)
                    fcf_yield = safe_num(financial_metrics.get("fcf_yield"), 0)

                    st.metric("EV/EBITDA", f"{ev_ebitda:.2f}" if ev_ebitda > 0 else "N/A")
                    # lower is better; if negative or zero, show 0 progress
                    st.progress(clamp01(1 - min(max(ev_ebitda, 0) / 25, 1)) if ev_ebitda > 0 else 0.0)

                    st.metric("Free Cash Flow Yield", f"{fcf_yield:.2f}%")
                    st.progress(clamp01(max(fcf_yield, 0) / 10))

            with tab3:
                st.subheader("Technical Analysis")
                col1, col2, col3 = st.columns(3)

                with col1:
                    rsi = safe_num(technical.get("rsi"), 50)
                    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    st.metric("RSI (14)", f"{rsi:.2f}", rsi_status)

                with col2:
                    st.metric("50 Day MA", f"${safe_num(technical.get('ma_50'), 0):.2f}")

                with col3:
                    st.metric("200 Day MA", f"${safe_num(technical.get('ma_200'), 0):.2f}")

                st.markdown("---")
                st.subheader("Moving Average Read")

                current_price = safe_num(technical.get("current_price"), 0)
                ma_50 = safe_num(technical.get("ma_50"), 0)
                ma_200 = safe_num(technical.get("ma_200"), 0)

                if ma_50 > 0 and current_price > ma_50:
                    st.success(f"Price above 50 Day MA (+{((current_price - ma_50) / ma_50 * 100):.2f}%)")
                elif ma_50 > 0:
                    st.error(f"Price below 50 Day MA ({((current_price - ma_50) / ma_50 * 100):.2f}%)")
                else:
                    st.info("Insufficient data for MA comparison.")

                if technical.get("golden_cross", False):
                    st.success("Golden Cross: 50 Day MA above 200 Day MA")
                else:
                    st.warning("Death Cross: 50 Day MA below 200 Day MA")

                st.markdown("---")
                st.metric("Max Drawdown (1 Year)", f"{safe_num(technical.get('max_drawdown'), 0):.1f}%")

            with tab4:
                st.subheader("Decision Brief")
                st.write(f"**Thesis:** {analysis['thesis']}")
                st.write(f"**Catalyst:** {analysis['catalyst']}")
                st.write(f"**Primary Risk:** {analysis['risk']}")
                st.markdown("---")
                st.write(
                    f"**Entry:** ${analysis['entry_price']:.2f}  |  "
                    f"**Support:** ${analysis['support_level']:.2f}  |  "
                    f"**Stop:** ${analysis['stop_loss']:.2f}"
                )

    elif mode == "Compare Stocks":
        spx = get_sp500_tickers()
        picks = st.sidebar.multiselect("Select tickers (2 to 6)", spx, default=["AAPL", "MSFT"], max_selections=6)
        picks = [norm_ticker(x) for x in picks]

        if st.sidebar.button("📊 Compare", type="primary"):
            df = compare_stocks(picks)
            if df.empty:
                st.warning("No comparison data returned.")
            else:
                st_df(df)

    elif mode == "Stock Scanner":
        spx = get_sp500_tickers()
        n = st.sidebar.slider("How many stocks to scan", 10, 200, 60, step=10)

        if st.sidebar.button("🔎 Run Scan", type="primary"):
            with st.spinner("Scanning..."):
                df = scan_stocks(spx, max_stocks=n)
                if df.empty:
                    st.warning("No matches found with the current filters.")
                else:
                    df = df.sort_values(["Piotroski", "FCF Yield %"], ascending=[False, False])
                    st_df(df)

    elif mode == "Backtest":
        t = norm_ticker(st.sidebar.text_input("Ticker", value="CELH"))
        months = st.sidebar.slider("Months back", 1, 12, 2)

        if st.sidebar.button("⏱️ Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                r = backtest_strategy(t, months_back=months)
                if not r:
                    st.warning("Not enough data to backtest.")
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Return", f"{r['returns']:.2f}%")
                    col2.metric("SPY Return", f"{r['spy_returns']:.2f}%")
                    col3.metric("Alpha", f"{r['alpha']:.2f}%")
                    st.caption(f"Entry: ${r['entry_price']:.2f} → Now: ${r['current_price']:.2f}")

if __name__ == "__main__":
    main()
