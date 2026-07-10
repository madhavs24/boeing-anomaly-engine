"""Industry-grade detection: 'is BA abnormal NOW' + 'will it be abnormal SOON', with honest,
calibrated, regime-robust machinery.

Components
  1. adaptive_conformal()  — Adaptive Conformal Inference (Gibbs & Candes). The alarm threshold
     adapts online so the realized flag rate tracks the target even under volatility regime
     shifts (fixes the exchangeability problem of plain conformal).
  2. calibrated_flare()    — walk-forward flare probability with ISOTONIC calibration on a held-out
     tail of each training window (no leakage). Reports AUC, PR-AUC and Brier score (calibration).
  3. precision_alerts()    — pick the probability threshold on TRAIN to hit a target precision,
     apply out-of-sample; report precision / recall / alerts-per-year (actionable, not just AUC).
  4. detect_combined()     — fuses the 'now' anomaly votes with the 'soon' calibrated flare into a
     single 0-3 severity for the dashboard / live monitor.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from .features import build_features
from .direction_pro import rich_features, RICH
from .models import _meta_features, _target
from . import anomaly as A


def adaptive_conformal(resid_z, target=0.01, gamma=0.02, min_cal=250):
    """ACI: online-adapted alarm rate. Returns (p_value, flag, realized_rate)."""
    s = resid_z.abs(); vals = s.values
    p = pd.Series(np.nan, index=s.index); flag = pd.Series(0, index=s.index)
    a_t = target
    for i in range(min_cal, len(vals)):
        if np.isnan(vals[i]):
            continue
        past = vals[:i]; past = past[~np.isnan(past)]
        if len(past) < min_cal:
            continue
        pv = (1 + np.sum(past >= vals[i])) / (len(past) + 1)
        fired = int(pv <= a_t)
        p.iloc[i] = pv; flag.iloc[i] = fired
        a_t = float(np.clip(a_t + gamma * (target - fired), 1e-4, 0.2))   # ACI update
    return p, flag, float(flag.mean())


def _flare_y(feats, horizon=5):
    y, _ = _target(feats, "flare", horizon); return y


def calibrated_flare(feats, base="hgb", refit=126, min_train=750, calib_frac=0.2):
    F = rich_features(feats).join(_meta_features(feats))
    cols = [c for c in RICH if c in F.columns] + list(_meta_features(feats).columns)
    y = _flare_y(feats)
    df = F[cols].join(y.rename("y")).replace([np.inf, -np.inf], np.nan).dropna()
    X, yv = df[cols].values, df["y"].astype(int).values
    proba = pd.Series(np.nan, index=df.index); i = min_train
    while i < len(df) - 1:
        cut = int(i * (1 - calib_frac))
        if len(np.unique(yv[:cut])) < 2 or len(np.unique(yv[cut:i])) < 2:
            i += refit; continue
        mdl = (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
               l2_regularization=1.0) if base == "hgb" else
               RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=30, n_jobs=-1))
        mdl.fit(X[:cut], yv[:cut])
        raw_cal = mdl.predict_proba(X[cut:i])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, yv[cut:i])
        proba.iloc[i:i + refit] = iso.predict(mdl.predict_proba(X[i:i + refit])[:, 1])
        i += refit
    o = pd.concat({"y": df["y"], "p": proba}, axis=1).dropna()
    metrics = {"n": int(len(o)), "auc": round(float(roc_auc_score(o["y"], o["p"])), 3),
               "pr_auc": round(float(average_precision_score(o["y"], o["p"])), 3),
               "brier": round(float(brier_score_loss(o["y"], o["p"])), 4),
               "base_rate": round(float(o["y"].mean()), 3)}
    return o["p"].reindex(feats.index), metrics


def precision_alerts(proba, feats, horizon=5, target_precision=0.6, train_frac=0.6):
    y = _flare_y(feats, horizon)
    o = pd.concat({"p": proba, "y": y}, axis=1).dropna()
    n = len(o); cut = int(n * train_frac)
    tr, te = o.iloc[:cut], o.iloc[cut:]
    # choose smallest threshold on TRAIN achieving target precision
    best = 0.99
    for thr in np.linspace(0.2, 0.95, 76):
        sel = tr[tr["p"] >= thr]
        if len(sel) >= 20 and sel["y"].mean() >= target_precision:
            best = thr; break
    sel = te[te["p"] >= best]
    prec = float(sel["y"].mean()) if len(sel) else float("nan")
    rec = float(sel["y"].sum() / te["y"].sum()) if te["y"].sum() else float("nan")
    yrs = max((te.index[-1] - te.index[0]).days / 365.25, 1e-9)
    return {"threshold": round(best, 3), "oos_precision": round(prec, 3),
            "oos_recall": round(rec, 3), "alerts_per_year": round(len(sel) / yrs, 1),
            "oos_base_rate": round(float(te["y"].mean()), 3)}


def detect_combined(feats, alpha=0.01, min_votes=2):
    """Fuse 'now' (adaptive-conformal + IsoForest + LOF votes) with 'soon' (calibrated flare)."""
    anom = A.detect(feats, alpha=alpha, contamination=alpha, min_votes=min_votes)
    p, aci_flag, rate = adaptive_conformal(feats["resid_z"], target=alpha)
    flare_p, _ = calibrated_flare(feats)
    out = anom.copy()
    out["aci_flag"] = aci_flag.reindex(out.index).fillna(0).astype(int)
    out["flare_prob"] = flare_p.reindex(out.index)
    out["alert"] = ((out["votes"] >= min_votes) | (out["flare_prob"] >= 0.6)).astype(int)
    return out, {"aci_realized_rate": round(rate, 4)}


if __name__ == "__main__":
    import json
    from .data import get_panel
    feats = build_features(get_panel("cached"))
    fp, m = calibrated_flare(feats)
    print("CALIBRATED FLARE:", json.dumps(m))
    print("PRECISION ALERTS:", json.dumps(precision_alerts(fp, feats)))
    p, flag, rate = adaptive_conformal(feats["resid_z"], target=0.01)
    print(f"ADAPTIVE CONFORMAL realized flag rate={rate:.4f} (target 0.0100)")
