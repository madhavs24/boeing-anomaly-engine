"""Boeing operational features — deliveries, orders, SEC fundamentals, FAA events.

All series are indexed by *release date* (when the figure becomes public), then forward-filled
to a daily grid with days_since_release counters. No backfill into the past.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import requests
from .util import DATA, log

RAW = DATA / "raw"
SEC_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000012927.json"
SEC_CACHE = RAW / "sec_companyfacts.json"
OD_CSV = DATA / "boeing_od_monthly.csv"
FAA_CSV = DATA / "faa_events.csv"
UA = {"User-Agent": "BoeingAnomalyEngine research contact@example.com"}

OD_URLS = [
    "https://www.boeing.com/resources/boeingdotcom/company/about_bca/stat/summary/files/orders_deliveries.xlsx",
]


def _causal_daily(sparse: pd.DataFrame, daily_index: pd.DatetimeIndex, prefix: str) -> pd.DataFrame:
    if sparse.empty:
        return pd.DataFrame(index=daily_index)
    s = sparse.sort_index().reindex(daily_index, method="ffill")
    rel = pd.Series(sparse.index, index=sparse.index).reindex(daily_index, method="ffill")
    out = s.copy()
    out[f"days_since_{prefix}"] = (daily_index - rel.values).days.astype(float)
    return out


def _load_od_monthly() -> pd.DataFrame:
    for url in OD_URLS:
        try:
            r = requests.get(url, headers=UA, timeout=20)
            if r.status_code == 200 and len(r.content) > 5000:
                RAW.mkdir(parents=True, exist_ok=True)
                (RAW / "orders_deliveries.xlsx").write_bytes(r.content)
                log(f"Boeing O&D xlsx downloaded ({len(r.content)} bytes)")
                break
        except Exception as e:
            log(f"Boeing O&D fetch {url}: {e}")
    if not OD_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(OD_CSV, parse_dates=["period_end", "release_date"])
    return df.set_index("release_date").sort_index()[["deliveries", "net_orders"]]


def _sec_facts() -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    if SEC_CACHE.exists() and SEC_CACHE.stat().st_size > 1_000_000:
        return json.loads(SEC_CACHE.read_text())
    r = requests.get(SEC_URL, headers=UA, timeout=60)
    r.raise_for_status()
    SEC_CACHE.write_text(r.text)
    log(f"SEC companyfacts cached ({len(r.text)} bytes)")
    return r.json()


def _sec_quarterly(tag: str, unit: str = "USD") -> pd.DataFrame:
    facts = _sec_facts()
    block = facts.get("facts", {}).get("us-gaap", {}).get(tag, {})
    rows = block.get("units", {}).get(unit, [])
    if not rows:
        return pd.DataFrame()
    recs = []
    for row in rows:
        if row.get("form") not in ("10-K", "10-Q", "8-K", None):
            continue
        end = pd.Timestamp(row.get("end"))
        filed = pd.Timestamp(row.get("filed"))
        if end.year < 2015:
            continue
        recs.append({"release_date": filed.normalize(), "period_end": end.normalize(), "value": float(row["val"])})
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs).sort_values("release_date")
    df = df.groupby("release_date", as_index=True)["value"].last().to_frame()
    return df.rename(columns={"value": tag})


def _sec_fcf_quarterly() -> pd.DataFrame:
    ocf = _sec_quarterly("NetCashProvidedByUsedInOperatingActivities")
    capex = _sec_quarterly("PaymentsToAcquirePropertyPlantAndEquipment")
    if ocf.empty:
        return pd.DataFrame()
    if capex.empty:
        return ocf.rename(columns={"NetCashProvidedByUsedInOperatingActivities": "fcf_q"})
    j = ocf.join(capex, how="outer").sort_index()
    j["fcf_q"] = j.iloc[:, 0] - j.iloc[:, 1].abs()
    return j[["fcf_q"]]


def _faa_features(daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    if not FAA_CSV.exists():
        return pd.DataFrame(index=daily_index)
    ev = pd.read_csv(FAA_CSV, parse_dates=["event_date"]).sort_values("event_date")
    out = pd.DataFrame(index=daily_index)
    out["faa_event_count"] = 0.0
    for _, row in ev.iterrows():
        d = row["event_date"].normalize()
        if d <= daily_index[-1]:
            out.loc[d:, "faa_event_count"] += 1.0
    out["days_since_faa_event"] = np.nan
    for d in ev["event_date"]:
        mask = daily_index >= d
        out.loc[mask, "days_since_faa_event"] = (daily_index[mask] - d).days.astype(float)
    return out


def build_operational_features(daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    daily_index = pd.DatetimeIndex(daily_index).normalize()
    parts = []

    od = _load_od_monthly()
    if not od.empty:
        od = od.rename(columns={"deliveries": "deliveries_m", "net_orders": "orders_m"})
        parts.append(_causal_daily(od, daily_index, "od_release"))

    rev = _sec_quarterly("Revenues")
    if not rev.empty:
        parts.append(_causal_daily(rev.rename(columns={"Revenues": "revenue_q"}), daily_index, "10q_release"))

    backlog = _sec_quarterly("RevenueRemainingPerformanceObligation")
    if not backlog.empty:
        parts.append(_causal_daily(backlog.rename(columns={"RevenueRemainingPerformanceObligation": "backlog_q"}), daily_index, "backlog_release"))

    fcf = _sec_fcf_quarterly()
    if not fcf.empty:
        parts.append(_causal_daily(fcf, daily_index, "fcf_release"))

    faa = _faa_features(daily_index)
    if not faa.empty:
        parts.append(faa)

    if not parts:
        return pd.DataFrame(index=daily_index)
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]
