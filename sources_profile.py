import yfinance as yf

def get_profile_best_effort(ticker: str) -> dict:
    """
    Best effort company profile. Uses Yahoo profile fields via yfinance.
    This is not a paid API but can be flaky. If it breaks, tool still works via SEC.
    """
    tk = yf.Ticker(ticker)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}

    # CEO best effort
    ceo = None
    officers = info.get("companyOfficers")
    if isinstance(officers, list) and officers:
        # try to find CEO title
        for o in officers:
            title = str(o.get("title", "")).lower()
            if "ceo" in title:
                ceo = o.get("name")
                break
        if ceo is None:
            ceo = officers[0].get("name")

    # earnings date best effort
    next_earnings = None
    try:
        cal = tk.calendar
        if cal is not None and not cal.empty and "Earnings Date" in cal.index:
            next_earnings = str(cal.loc["Earnings Date"][0])
    except Exception:
        pass

    return {
        "name": info.get("shortName") or info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "website": info.get("website"),
        "summary": info.get("longBusinessSummary"),
        "employees": info.get("fullTimeEmployees"),
        "ceo": ceo,
        "founded": None,  # not reliable from yfinance
        "market_cap": info.get("marketCap"),
        "current_price": info.get("currentPrice"),
        "currency": info.get("currency"),
        "next_earnings_date": next_earnings,
    }
