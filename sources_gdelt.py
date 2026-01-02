import pandas as pd
import requests

def gdelt_doc_search(session: requests.Session, query: str, max_records: int = 25) -> pd.DataFrame:
    """
    GDELT DOC API provides near real time news search.
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "sort": "HybridRel",
    }
    r = session.get(url, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    arts = j.get("articles", []) or []
    rows = []
    for a in arts:
        rows.append({
            "seen": a.get("seendate"),
            "title": a.get("title"),
            "source": a.get("domain"),
            "url": a.get("url"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["seen"] = pd.to_datetime(df["seen"], errors="coerce")
        df = df.sort_values("seen", ascending=False).reset_index(drop=True)
    return df
