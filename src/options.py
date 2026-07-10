"""BA-specific options implied volatility (30d ATM IV + put-call skew).

Uses yfinance option chains. Historical depth is limited (~1-2y of expiries); series is cached
in data/processed/ba_iv.parquet and forward-filled causally.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .util import PROC, log

BA_IV_PATH = PROC / "ba_iv.parquet"


def _atm_iv(chain, spot: float, expiry_days: int) -> tuple[float, float]:
    """Return (atm_iv, skew) from a single expiry chain."""
    if chain is None or spot <= 0:
        return np.nan, np.nan
    calls = getattr(chain, "calls", chain)
    puts = getattr(chain, "puts", None)
    if puts is None or calls is None:
        return np.nan, np.nan
    calls = calls.copy(); puts = puts.copy()
    if calls.empty or puts.empty:
        return np.nan, np.nan
    calls["dist"] = (calls["strike"] - spot).abs()
    puts["dist"] = (puts["strike"] - spot).abs()
    atm_c = calls.loc[calls["dist"].idxmin()]
    atm_p = puts.loc[puts["dist"].idxmin()]
    atm_iv = float(np.nanmean([atm_c.get("impliedVolatility", np.nan), atm_p.get("impliedVolatility", np.nan)]))
    otm_puts = puts[puts["strike"] < spot * 0.95]
    otm_calls = calls[calls["strike"] > spot * 1.05]
    if otm_puts.empty or otm_calls.empty:
        skew = np.nan
    else:
        put_iv = float(otm_puts.iloc[(otm_puts["strike"] - spot * 0.95).abs().argmin()].get("impliedVolatility", np.nan))
        call_iv = float(otm_calls.iloc[(otm_calls["strike"] - spot * 1.05).abs().argmin()].get("impliedVolatility", np.nan))
        skew = put_iv - call_iv if np.isfinite(put_iv) and np.isfinite(call_iv) else np.nan
    return atm_iv, skew


def _norm_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize(None).normalize() if t.tzinfo else t.normalize()


def fetch_ba_iv_snapshot(as_of: pd.Timestamp | None = None) -> dict:
    import yfinance as yf
    as_of = _norm_ts(as_of or pd.Timestamp.today())
    tkr = yf.Ticker("BA")
    spot = float(tkr.history(period="5d")["Close"].iloc[-1])
    expiries = tkr.options
    if not expiries:
        return {"as_of": as_of, "ba_iv": np.nan, "ba_iv_skew": np.nan}
    best_exp, best_d = None, 10_000
    for exp in expiries:
        d = (_norm_ts(exp) - as_of).days
        if 20 <= d <= 45 and abs(d - 30) < abs(best_d - 30):
            best_exp, best_d = exp, d
    if best_exp is None:
        for exp in expiries:
            d = (_norm_ts(exp) - as_of).days
            if d > 7 and abs(d - 30) < abs(best_d - 30):
                best_exp, best_d = exp, d
    if best_exp is None:
        return {"as_of": as_of, "ba_iv": np.nan, "ba_iv_skew": np.nan}
    chain = tkr.option_chain(best_exp)
    iv, skew = _atm_iv(chain, spot, best_d)
    return {"as_of": as_of, "ba_iv": iv, "ba_iv_skew": skew}


def build_ba_iv_series(panel_index: pd.DatetimeIndex, refresh: bool = False) -> pd.DataFrame:
    """Build or load cached BA IV series aligned to panel_index."""
    if BA_IV_PATH.exists() and not refresh:
        cached = pd.read_parquet(BA_IV_PATH)
        cached.index = pd.DatetimeIndex(cached.index).normalize()
        return cached.reindex(panel_index, method="ffill")

    import yfinance as yf
    hist = yf.Ticker("BA").history(period="2y")
    if not hist.empty:
        hist.index = pd.DatetimeIndex([_norm_ts(x) for x in hist.index])
    if hist.empty:
        log("options: no BA price history")
        return pd.DataFrame(index=panel_index, columns=["ba_iv", "ba_iv_skew"])

    rows = []
    snap = fetch_ba_iv_snapshot(_norm_ts(hist.index[-1]))
    rows.append({"date": snap["as_of"], "ba_iv": snap["ba_iv"], "ba_iv_skew": snap["ba_iv_skew"]})

    tkr = yf.Ticker("BA")
    for exp in list(tkr.options or [])[:5]:
        try:
            chain = tkr.option_chain(exp)
            exp_dt = _norm_ts(exp)
            if exp_dt > _norm_ts(hist.index[-1]):
                exp_dt = _norm_ts(hist.index[-1])
            spot = float(hist["Close"].asof(exp_dt))
            if not np.isfinite(spot):
                continue
            iv, skew = _atm_iv(chain, spot, 30)
            if np.isfinite(iv):
                rows.append({"date": exp_dt, "ba_iv": iv, "ba_iv_skew": skew})
        except Exception as e:
            log(f"options chain {exp}: {e}")

    if not rows:
        return pd.DataFrame(index=panel_index, columns=["ba_iv", "ba_iv_skew"])

    df = pd.DataFrame(rows).dropna(subset=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    daily = df.reindex(panel_index, method="ffill")
    daily.to_parquet(BA_IV_PATH)
    log(f"ba_iv cached: {len(df)} snapshots -> {BA_IV_PATH}")
    return daily
