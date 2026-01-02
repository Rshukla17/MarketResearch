import pandas as pd
import requests

NASDAQLISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHERLISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

def _read_pipe_file(session: requests.Session, url: str) -> pd.DataFrame:
    r = session.get(url, timeout=20)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    # last line is a footer like "File Creation Time: ..."
    if len(lines) > 2:
        lines = lines[:-1]
    header = lines[0].split("|")
    rows = [ln.split("|") for ln in lines[1:]]
    df = pd.DataFrame(rows, columns=header)
    return df

def get_us_listed_universe(session: requests.Session) -> pd.DataFrame:
    """
    Builds a US listed universe using Nasdaq Trader symbol directory files.
    This is "listed issues" not "broker tradable flags".
    """
    df_n = _read_pipe_file(session, NASDAQLISTED_URL)
    df_o = _read_pipe_file(session, OTHERLISTED_URL)

    df_n = df_n.rename(columns={"Symbol": "ticker", "Security Name": "security_name", "ETF": "etf", "Test Issue": "test_issue"})
    df_o = df_o.rename(columns={"ACT Symbol": "ticker", "Security Name": "security_name", "ETF": "etf", "Test Issue": "test_issue", "Exchange": "exchange"})

    df_n["exchange"] = "NASDAQ"
    if "exchange" not in df_o.columns:
        df_o["exchange"] = "OTHER"

    keep = ["ticker", "security_name", "exchange", "etf", "test_issue"]
    for c in keep:
        if c not in df_n.columns:
            df_n[c] = None
        if c not in df_o.columns:
            df_o[c] = None

    df = pd.concat([df_n[keep], df_o[keep]], ignore_index=True)
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["security_name"] = df["security_name"].astype(str).str.strip()
    df = df[df["ticker"].str.fullmatch(r"[A-Z.\-]{1,10}", na=False)]
    df = df.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
    return df

def filter_universe(df: pd.DataFrame, include_etfs: bool, include_test_issues: bool) -> pd.DataFrame:
    out = df.copy()
    if not include_etfs and "etf" in out.columns:
        out = out[out["etf"].astype(str).str.upper().ne("Y")]
    if not include_test_issues and "test_issue" in out.columns:
        out = out[out["test_issue"].astype(str).str.upper().ne("Y")]
    return out.reset_index(drop=True)
