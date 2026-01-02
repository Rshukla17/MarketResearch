import pandas as pd

def rsi(close: pd.Series, period: int = 14) -> float | None:
    if close is None or close.empty or close.shape[0] < period + 2:
        return None
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(period).mean()
    loss = down.rolling(period).mean()
    rs = gain / loss
    val = 100 - (100 / (1 + rs))
    out = val.iloc[-1]
    try:
        return float(out)
    except Exception:
        return None

def conviction_score(piotroski: int, price_df: pd.DataFrame) -> dict:
    """
    Transparent 0 to 100 score.
    Quality 30: scaled Piotroski.
    Momentum 40: 12m return + trend + RSI band.
    Risk 30: drawdown.
    """
    parts = {"quality": 0, "momentum": 0, "risk": 0}
    total = 0

    q = int(round((max(0, min(9, piotroski)) / 9) * 30))
    parts["quality"] = q
    total += q

    if price_df is None or price_df.empty or "Close" not in price_df.columns:
        return {"score": total, "parts": parts}

    df = price_df.copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    if df.shape[0] < 260:
        return {"score": total, "parts": parts}

    close = df["Close"].reset_index(drop=True)

    # 12m return
    ret_12m = (close.iloc[-1] / close.iloc[-252]) - 1
    # map -50%..+100% into 0..25
    x = max(-0.5, min(1.0, float(ret_12m)))
    mom = int(round(((x + 0.5) / 1.5) * 25))

    # trend
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    if pd.notna(ma50) and pd.notna(ma200) and ma50 > ma200:
        mom += 10

    # RSI band 40 to 60
    r = rsi(close, 14)
    if r is not None and 40 <= r <= 60:
        mom += 5

    mom = min(40, max(0, mom))
    parts["momentum"] = mom
    total += mom

    # risk via max drawdown
    roll_max = close.cummax()
    dd = (close / roll_max) - 1
    max_dd = float(dd.min())  # negative
    # map -70%..0% into 0..30
    z = max(-0.7, min(0.0, max_dd))
    risk = int(round(((z + 0.7) / 0.7) * 30))
    risk = min(30, max(0, risk))
    parts["risk"] = risk
    total += risk

    return {
        "score": int(max(0, min(100, total))),
        "parts": parts,
        "ret_12m": float(ret_12m),
        "max_drawdown": max_dd,
        "rsi": r,
    }
