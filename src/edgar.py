"""Boeing OPERATIONAL fundamentals (Ideas #1,#2) — SEC EDGAR company facts.

Pulls quarterly financials for Boeing (CIK 0000012927) and turns them into CAUSAL daily
features: a figure is only known on/after its SEC FILING date (never backfilled), plus a
`days_since_report` recency feature. This is the biggest structural gap in the model.

Features:
  op_revenue_yoy    revenue vs year-ago quarter
  op_ocf            operating cash flow (proxy for FCF direction), scaled
  op_days_since     trading days since the last filing (recency / staleness)

Offline fallback returns None -> pipeline still runs. Network needed for the real fetch.

    python -m src.operational
"""
from __future__ import annotations
import numpy as np, pandas as pd, requests
from .util import PROC, log

CIK = "0000012927"   # The Boeing Company
FACTS = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"
UA = {"User-Agent": "boeing-anomaly-engine singhm@beloit.edu"}
CACHE = PROC / "operational.parquet"
CACHE_MAX_AGE_DAYS = 30


def _concept(facts, tags):
    best = None
    for t in tags:
        node = facts.get("us-gaap", {}).get(t)
        if not node:
            continue
        rows = []
        for unit, arr in node.get("units", {}).items():
            for it in arr:
                form = it.get("form", "")
                if it.get("filed") and it.get("val") is not None and form.startswith("10"):
                    rows.append((pd.Timestamp(it["filed"]), pd.Timestamp(it["end"]), float(it["val"])))
        if rows:
            df = pd.DataFrame(rows, columns=["filed", "end", "val"]).sort_values("filed")
            df = df.drop_duplicates("end", keep="last")
            if best is None or len(df) > len(best):
                best = df
    return best


def _load_cache(index: pd.DatetimeIndex, max_age_days=CACHE_MAX_AGE_DAYS) -> pd.DataFrame | None:
    if not CACHE.exists():
        return None
    try:
        age = (pd.Timestamp.today() - pd.Timestamp(CACHE.stat().st_mtime, unit="s")).days
        if age > max_age_days:
            return None
        f = pd.read_parquet(CACHE)
        f.index = pd.to_datetime(f.index)
        log(f"EDGAR operational loaded from cache (age {age}d)")
        return f.reindex(index)
    except Exception as e:
        log(f"EDGAR cache read failed: {e}")
    return None


def operational_features(index: pd.DatetimeIndex, force_refresh=False) -> pd.DataFrame | None:
    if not force_refresh:
        cached = _load_cache(index)
        if cached is not None and cached[OP_FEATS].notna().any().any():
            return cached
    try:
        r = requests.get(FACTS, headers=UA, timeout=30)
        r.raise_for_status()
        facts = r.json().get("facts", {})
    except Exception as e:
        log(f"EDGAR fetch failed: {e}")
        return _load_cache(index, max_age_days=10_000)
    rev = _concept(facts, ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"])
    ocf = _concept(facts, ["NetCashProvidedByUsedInOperatingActivities"])
    if rev is None:
        return _load_cache(index, max_age_days=10_000)
    f = pd.DataFrame(index=index)
    # revenue YoY, known only on filing date -> forward fill from 'filed'
    rev = rev.set_index("filed")
    rev = rev[~rev.index.duplicated(keep="last")]
    rev["yoy"] = rev["val"] / rev["val"].shift(4) - 1
    f["op_revenue_yoy"] = rev["yoy"].reindex(index.union(rev.index)).sort_index().ffill().reindex(index)
    if ocf is not None:
        ocf = ocf.set_index("filed")
        ocf = ocf[~ocf.index.duplicated(keep="last")]
        s = ocf["val"] / 1e9
        f["op_ocf"] = s.reindex(index.union(ocf.index)).sort_index().ffill().reindex(index)
    last_filed = rev.index.to_series()
    f["op_days_since"] = pd.Series(index, index=index).apply(
        lambda d: (d - last_filed[last_filed <= d].max()).days if (last_filed <= d).any() else np.nan)
    f.to_parquet(CACHE)
    log(f"EDGAR operational fetched ({rev.index.min().date()} .. {rev.index.max().date()})")
    return f


OP_FEATS = ["op_revenue_yoy", "op_ocf", "op_days_since"]


if __name__ == "__main__":
    idx = pd.bdate_range("2016-06-24", pd.Timestamp.today())
    d = operational_features(idx, force_refresh=True)
    if d is None or not d[OP_FEATS].notna().any().any():
        print("FAILED: no EDGAR data (check network / User-Agent)")
        raise SystemExit(1)
    print(d[OP_FEATS].dropna().tail())
