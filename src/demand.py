"""Air-travel DEMAND features — the airline-industry signal (Idea #4).

Source: TSA daily checkpoint passenger volumes (public, free, daily). Demand -> airline health
-> aircraft orders/deliveries, so a demand shock is a genuine industry signal for BA.

Causal daily features:
  demand_yoy   passengers vs the same day last year (the standard TSA comparison)
  demand_mom   21-day momentum of passenger volume
  demand_z     z-score of demand vs its trailing year (shock detector)

Robust: network failure or missing data -> returns None (pipeline still runs). A synthetic
fallback is provided for offline testing so downstream code paths are exercised.

    python -m src.demand            # fetch + cache (real data only)
"""
from __future__ import annotations
import io, re
import numpy as np, pandas as pd, requests
from .util import PROC, log

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
CACHE = PROC / "demand_raw.parquet"
CACHE_FEATS = PROC / "demand.parquet"
CACHE_MAX_AGE_DAYS = 7
TSA_PAGE = "https://www.tsa.gov/travel/passenger-volumes"


def _parse_tsa_series(df: pd.DataFrame) -> pd.Series | None:
    """Heuristically find date + passenger-count columns in a TSA table/CSV."""
    cols = list(df.columns)
    dcol = next((c for c in cols if "date" in str(c).lower()), None)
    if dcol is None:
        for c in cols:
            if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.8:
                dcol = c
                break
    if dcol is None:
        return None
    ncol = None
    for c in cols:
        if c == dcol:
            continue
        nums = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
        if nums.notna().mean() > 0.8 and nums.median() > 50_000:
            ncol = c
            break
    if ncol is None:
        return None
    vals = pd.to_numeric(df[ncol].astype(str).str.replace(",", "", regex=False), errors="coerce")
    idx = pd.to_datetime(df[dcol], errors="coerce")
    s = pd.Series(vals.values, index=idx).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.index = s.index.normalize()
    return s if len(s) > 300 else None


def _fetch_tsa_gov() -> pd.Series | None:
    """Scrape TSA passenger-volumes pages (current + yearly archives)."""
    series = []
    years = list(range(2016, pd.Timestamp.today().year + 1))
    for year in years:
        url = TSA_PAGE if year == pd.Timestamp.today().year else f"{TSA_PAGE}/{year}"
        try:
            r = requests.get(url, headers=UA, timeout=20)
            r.raise_for_status()
            rows = re.findall(r"(\d{1,2}/\d{1,2}/\d{4})\s+([\d,]+)", r.text)
            if len(rows) < 30:
                continue
            s = pd.Series(
                [float(v.replace(",", "")) for _, v in rows],
                index=pd.to_datetime([d for d, _ in rows]),
            ).sort_index()
            s = s[~s.index.duplicated(keep="last")]
            series.append(s)
        except Exception as e:
            log(f"TSA year {year} scrape failed: {e}")
    if not series:
        return None
    out = pd.concat(series).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if len(out) > 300:
        log(f"TSA demand fetched from tsa.gov pages ({len(out)} days)")
        return out
    return None


def _fetch_tsa_mirrors() -> pd.Series | None:
    urls = [
        "https://raw.githubusercontent.com/bcantoni/tsa-data/main/tsa.csv",
    ]
    for u in urls:
        try:
            r = requests.get(u, headers=UA, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            s = _parse_tsa_series(df)
            if s is not None:
                log(f"TSA demand fetched from mirror ({len(s)} days): {u[:60]}")
                return s
        except Exception as e:
            log(f"TSA mirror failed ({u[:50]}...): {e}")
    return None


def _load_cached_raw(max_age_days=CACHE_MAX_AGE_DAYS) -> pd.Series | None:
    if not CACHE.exists():
        return None
    try:
        age = (pd.Timestamp.today() - pd.Timestamp(CACHE.stat().st_mtime, unit="s")).days
        if age > max_age_days:
            return None
        s = pd.read_parquet(CACHE).squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        if len(s) > 300:
            log(f"TSA demand loaded from cache ({len(s)} days, age {age}d)")
            return s.sort_index()
    except Exception as e:
        log(f"TSA cache read failed: {e}")
    return None


def _fetch_tsa() -> pd.Series | None:
    """Return daily TSA passenger counts from cache, mirror, or scrape."""
    cached = _load_cached_raw()
    if cached is not None:
        return cached
    for fn in (_fetch_tsa_mirrors, _fetch_tsa_gov):
        s = fn()
        if s is not None:
            s.to_frame("passengers").to_parquet(CACHE)
            return s
    return None


def _synthetic(index) -> pd.Series:
    rng = np.random.default_rng(3)
    base = 2.2e6 + 4e5 * np.sin(np.linspace(0, 30, len(index)))
    covid = np.where((index >= "2020-03-15") & (index < "2021-06-01"), 0.25, 1.0)
    return pd.Series(base * covid * (1 + rng.normal(0, 0.05, len(index))), index=index)


def demand_features(index: pd.DatetimeIndex, allow_synthetic=True) -> pd.DataFrame | None:
    s = _fetch_tsa()
    if s is None:
        if not allow_synthetic:
            return None
        log("TSA demand unavailable — using synthetic demand for offline testing")
        s = _synthetic(index)
    s = s.reindex(index.union(s.index)).sort_index().ffill(limit=5)
    f = pd.DataFrame(index=s.index)
    f["demand_yoy"] = s / s.shift(365) - 1
    f["demand_mom"] = s.pct_change(21)
    f["demand_z"] = (s - s.rolling(252, min_periods=60).mean()) / s.rolling(252, min_periods=60).std()
    out = f.reindex(index)
    out.to_parquet(CACHE_FEATS)
    return out


DEMAND_FEATS = ["demand_yoy", "demand_mom", "demand_z"]


if __name__ == "__main__":
    idx = pd.bdate_range("2016-01-01", pd.Timestamp.today())
    d = demand_features(idx, allow_synthetic=False)
    if d is None:
        print("FAILED: no real TSA data (check network / sources)")
        raise SystemExit(1)
    print(d.dropna().tail())
    print(f"rows={d.notna().any(axis=1).sum()}  range={d.dropna().index.min().date()} .. {d.dropna().index.max().date()}")
