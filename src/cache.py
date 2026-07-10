"""Real-time cache layer — train once, predict in milliseconds.

The slow part (fitting models over 10y of history) is done ONCE by `train_and_cache()` and
persisted with joblib. Thereafter `predict_now()` loads the cached models and the latest
cached features and returns today's anomaly + volatility + flare prediction in a fraction of
a second — suitable for an intraday/near-real-time loop.

    python -m src.cache train     # fit + persist models  (results/models.joblib)
    python -m src.cache now       # fast prediction on the latest cached bar

For genuine real-time use, call data.update_latest() (network, on your machine) to append the
newest bars to panel.parquet, then predict_now() — the recompute is incremental and quick.
"""
from __future__ import annotations
import time, sys, threading
import numpy as np
import pandas as pd
import joblib
from .util import RESULTS, load_config
from .features import build_features, rolling_residual, _ret
from .direction_pro import rich_features, RICH
from .models import make_models, _meta_features, _target
from . import anomaly as A

CACHE = RESULTS / "models.joblib"
_FEATS = {"ts": 0.0, "df": None}
_FEATS_LOCK = threading.Lock()
FEATS_TTL = 300


def _get_feats(force=False):
    """Reuse built features for a few minutes — rolling_residual rebuild takes ~30s."""
    if not force:
        with _FEATS_LOCK:
            if _FEATS["df"] is not None and (time.time() - _FEATS["ts"]) < FEATS_TTL:
                return _FEATS["df"]
    from .data import get_panel
    feats = build_features(get_panel("cached"))
    with _FEATS_LOCK:
        _FEATS.update(ts=time.time(), df=feats)
    return feats


def train_and_cache(feats=None):
    if feats is None:
        feats = _get_feats(force=True)
    cfg = load_config(); alpha = float(cfg.get("anomaly_alpha", 0.01))
    F = rich_features(feats).join(_meta_features(feats))
    cols = [c for c in RICH if c in F.columns] + [c for c in _meta_features(feats).columns]

    # --- flare hybrid model (fit on all history) ---
    y, _ = _target(feats, "flare")
    df = F[cols].join(y.rename("y")).replace([np.inf, -np.inf], np.nan).dropna()
    flare = make_models()["hybrid_vote"].fit(df[cols].values, df["y"].astype(int).values)

    # --- anomaly: conformal calibration array + IsolationForest on non-residual feats ---
    from sklearn.ensemble import IsolationForest
    Xif = feats[[c for c in A.IF_FEATS if c in feats.columns]].replace([np.inf, -np.inf], np.nan).ffill().dropna()
    iso = IsolationForest(n_estimators=200, contamination=alpha, random_state=0).fit(Xif.values)
    calib = feats["resid_z"].abs().dropna().values

    # --- HARX volatility coefficients (fit on all history) ---
    ret = feats["ret"]; h = int(cfg.get("vol_horizon", 10))
    rv_d = ret.rolling(5).std() * np.sqrt(252); rv_w = ret.rolling(21).std() * np.sqrt(252)
    rv_m = ret.rolling(63).std() * np.sqrt(252)
    fwd = (ret.rolling(h).std() * np.sqrt(252)).shift(-h)
    vx = feats["vix"] if "vix" in feats else pd.Series(0, index=feats.index)
    vdf = pd.concat({"d": rv_d, "w": rv_w, "m": rv_m, "vix": vx, "y": fwd}, axis=1).dropna()
    Xv = np.column_stack([np.ones(len(vdf)), vdf["d"], vdf["w"], vdf["m"], vdf["vix"]])
    harx_coef, *_ = np.linalg.lstsq(Xv, vdf["y"].values, rcond=None)

    blob = {"flare": flare, "flare_cols": cols, "iso": iso, "iso_cols": list(Xif.columns),
            "calib": calib, "alpha": alpha, "harx_coef": harx_coef, "vol_horizon": h,
            "trained_at": str(pd.Timestamp.now()), "rows": int(len(feats))}
    joblib.dump(blob, CACHE)
    print(f"cached models -> {CACHE} (trained on {len(feats)} rows)")
    return blob


def predict_now(feats=None, blob=None):
    t0 = time.perf_counter()
    if blob is None:
        blob = joblib.load(CACHE)
    if feats is None:
        feats = _get_feats()
    F = rich_features(feats).join(_meta_features(feats))
    row = F.iloc[[-1]]
    # flare probability
    xf = row[blob["flare_cols"]].replace([np.inf, -np.inf], np.nan)
    flare_p = float(blob["flare"].predict_proba(np.nan_to_num(xf.values))[:, 1][0])
    # anomaly: conformal p-value of today's |resid_z| vs calibration + IsoForest flag
    rz = abs(float(feats["resid_z"].iloc[-1]))
    p_val = (1 + np.sum(blob["calib"] >= rz)) / (len(blob["calib"]) + 1)
    xif = feats[blob["iso_cols"]].iloc[[-1]].replace([np.inf, -np.inf], np.nan).ffill()
    iso_flag = int(blob["iso"].predict(np.nan_to_num(xif.values))[0] == -1)
    votes = int(p_val <= blob["alpha"]) + iso_flag
    # volatility forecast (HARX)
    ret = feats["ret"]; rv_d = ret.rolling(5).std().iloc[-1] * np.sqrt(252)
    rv_w = ret.rolling(21).std().iloc[-1] * np.sqrt(252); rv_m = ret.rolling(63).std().iloc[-1] * np.sqrt(252)
    vix = float(feats["vix"].iloc[-1]) if "vix" in feats else 0.0
    vol_fc = float(blob["harx_coef"] @ np.array([1, rv_d, rv_w, rv_m, vix]))
    ms = (time.perf_counter() - t0) * 1000
    return {"as_of": str(feats.index[-1].date()), "latency_ms": round(ms, 1),
            "flare_prob_5d": round(flare_p, 3), "anomaly_votes": votes,
            "anomaly_conformal_p": round(float(p_val), 4), "resid_z_today": round(rz, 2),
            "vol_forecast_10d_annualized": round(vol_fc, 3)}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "now"
    if cmd == "train":
        train_and_cache()
    else:
        import json; print(json.dumps(predict_now(), indent=2))
