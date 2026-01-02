def _to_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def compute_piotroski(sec_facts: dict, helpers) -> dict:
    """
    9 point Piotroski F score using SEC facts when possible.
    Missing items are treated as 0 and reported.
    helpers must supply latest_annual_two(facts, tag)
    """
    missing = []

    def two(tag, name):
        v = helpers.latest_annual_two(sec_facts, tag)
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

    def roa(x_ni, x_assets):
        if not x_ni or not x_assets:
            return None, None
        ni_curr = _to_float(x_ni["curr"])
        ni_prev = _to_float(x_ni["prev"])
        a_curr = _to_float(x_assets["curr"])
        a_prev = _to_float(x_assets["prev"])
        if not a_curr or not a_prev:
            return None, None
        return ni_curr / a_curr if ni_curr is not None else None, ni_prev / a_prev if ni_prev is not None else None

    roa_curr, roa_prev = roa(ni, assets)

    checks["ROA positive"] = 1 if (roa_curr is not None and roa_curr > 0) else 0
    checks["CFO positive"] = 1 if (_to_float(cfo["curr"]) is not None and _to_float(cfo["curr"]) > 0) else 0
    checks["ROA improved"] = 1 if (roa_curr is not None and roa_prev is not None and (roa_curr - roa_prev) > 0) else 0
    checks["CFO exceeds NI"] = 1 if (_to_float(cfo["curr"]) is not None and _to_float(ni["curr"]) is not None and _to_float(cfo["curr"]) > _to_float(ni["curr"])) else 0

    # leverage down
    checks["Leverage down"] = 1 if (_to_float(ltd["curr"]) is not None and _to_float(ltd["prev"]) is not None and (_to_float(ltd["curr"]) - _to_float(ltd["prev"])) < 0) else 0

    # current ratio up
    cr_curr = None
    cr_prev = None
    if ca and cl and _to_float(ca["curr"]) and _to_float(cl["curr"]):
        cr_curr = _to_float(ca["curr"]) / _to_float(cl["curr"])
    if ca and cl and _to_float(ca["prev"]) and _to_float(cl["prev"]):
        cr_prev = _to_float(ca["prev"]) / _to_float(cl["prev"])
    checks["Current ratio up"] = 1 if (cr_curr is not None and cr_prev is not None and (cr_curr - cr_prev) > 0) else 0

    # no dilution
    checks["No dilution"] = 1 if (_to_float(shares["curr"]) is not None and _to_float(shares["prev"]) is not None and (_to_float(shares["curr"]) - _to_float(shares["prev"])) <= 0) else 0

    # gross margin up
    gm_curr = None
    gm_prev = None
    if gross and rev and _to_float(gross["curr"]) is not None and _to_float(rev["curr"]):
        gm_curr = _to_float(gross["curr"]) / _to_float(rev["curr"])
    if gross and rev and _to_float(gross["prev"]) is not None and _to_float(rev["prev"]):
        gm_prev = _to_float(gross["prev"]) / _to_float(rev["prev"])
    checks["Gross margin up"] = 1 if (gm_curr is not None and gm_prev is not None and (gm_curr - gm_prev) > 0) else 0

    # asset turnover up
    at_curr = None
    at_prev = None
    if rev and assets and _to_float(rev["curr"]) is not None and _to_float(assets["curr"]):
        at_curr = _to_float(rev["curr"]) / _to_float(assets["curr"])
    if rev and assets and _to_float(rev["prev"]) is not None and _to_float(assets["prev"]):
        at_prev = _to_float(rev["prev"]) / _to_float(assets["prev"])
    checks["Asset turnover up"] = 1 if (at_curr is not None and at_prev is not None and (at_curr - at_prev) > 0) else 0

    score = sum(checks.values())
    asof = assets["curr_end"] if assets else None

    return {"score": score, "checks": checks, "missing": missing, "asof": asof}
