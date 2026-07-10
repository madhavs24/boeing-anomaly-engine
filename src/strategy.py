"""Turn the signals into economic value — a causal, cost-aware backtest.

Raw direction prediction is a coin flip, but the anomaly + volatility signals can still ADD
VALUE by managing risk (the same lesson as the original turbulence project). Each strategy sets
a daily position in BA (0..1, rest in cash) using ONLY prior-day info, pays transaction costs on
turnover, and is compared to buy & hold on return, Sharpe, and — the real prize — max drawdown.

Strategies:
  buy_hold        always 100% invested (benchmark)
  derisk_anomaly  go to cash for N days after an anomaly fires (uses the event-conditional edge)
  vol_target      scale position to a target volatility using a causal vol estimate
  combo           vol_target, then flat during the post-anomaly window

    python -m src.strategy
"""
from __future__ import annotations
import numpy as np, pandas as pd
from .features import build_features
from . import anomaly as A
TRADING = 252
COST = 0.0005   # 5 bps per unit turnover


def _stats(ret, start=10000):
    eq = start * (1 + ret).cumprod()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / start) ** (1 / yrs) - 1
    vol = ret.std() * np.sqrt(TRADING)
    sharpe = ret.mean() / ret.std() * np.sqrt(TRADING) if ret.std() > 0 else np.nan
    downside = ret[ret < 0].std() * np.sqrt(TRADING)
    sortino = ret.mean() * TRADING / downside if downside > 0 else np.nan
    dd = float((eq / eq.cummax() - 1).min())
    calmar = (cagr / abs(dd)) if dd < 0 else np.nan
    return {"CAGR_%": round(cagr * 100, 1), "vol_%": round(vol * 100, 1),
            "sharpe": round(float(sharpe), 2), "sortino": round(float(sortino), 2),
            "max_drawdown_%": round(dd * 100, 1), "calmar": round(float(calmar), 2),
            "final_$": int(eq.iloc[-1])}


def oos_compare(feats, anom, train_frac=0.6, **kw):
    n=len(feats); cut=int(n*train_frac); f2,a2=feats.iloc[cut:],anom.iloc[cut:]
    return backtest(f2,a2,**kw).loc[["buy_hold","derisk_anomaly","combo"]]


def backtest(feats, anom, derisk_days=5, target_vol=0.30):
    r = feats["ret"].fillna(0.0)
    flag = anom["flag"].reindex(feats.index).fillna(0).astype(int)
    # causal post-anomaly window: in cash for `derisk_days` after a flag
    derisk = flag.rolling(derisk_days, min_periods=1).max().shift(1).fillna(0)
    # causal vol estimate (yesterday's 21d realized) for vol targeting
    rv = (r.rolling(21).std() * np.sqrt(TRADING)).shift(1)
    vt = (target_vol / rv).clip(0, 1.0).fillna(1.0)
    pos = {
        "buy_hold": pd.Series(1.0, index=r.index),
        "derisk_anomaly": (1 - derisk).astype(float),
        "vol_target": vt,
        "combo": (vt * (1 - derisk)).clip(0, 1),
    }
    out = {}
    for name, p in pos.items():
        p = p.shift(1).fillna(1.0 if name == "buy_hold" else 0.0).clip(0, 1)  # act next day
        turn = p.diff().abs().fillna(0)
        net = p * r - turn * COST
        s = _stats(net); s["pct_invested"] = round(float(p.mean()) * 100)
        s["turnover_/yr"] = round(float(turn.sum() / (len(turn) / TRADING)), 1)
        out[name] = s
    return pd.DataFrame(out).T


if __name__ == "__main__":
    from .data import get_panel
    feats = build_features(get_panel("cached"))
    anom = A.detect(feats, min_votes=2)
    print(backtest(feats, anom).to_string())
