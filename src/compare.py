"""See the impact of the new features — a BEFORE/AFTER comparison (Idea validation).

Runs the flare detector with the CORE feature set vs CORE + whatever optional industry features
are currently present (demand_*, op_*, news_*), on BOTH hgb and rf, and prints the deltas. Also
reports anomaly recall + volatility skill for context. Enforces the discipline rule: an addition
"passes" only if it improves flare AUC on BOTH models.

    python -m src.compare          # uses config flags to decide which optional feats are on
"""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from .data import get_panel
from .features import build_features
from .direction_pro import rich_features, RICH
from .models import _meta_features, _target
from . import anomaly as A, volatility as V

warnings.filterwarnings("ignore")
OPT_PREFIX = ("demand_", "op_", "news_")


def _flare(F, cols, model, y, refit=126, min_train=750):
    df = F[cols].join(y.rename("y")).replace([np.inf, -np.inf], np.nan).dropna()
    if df["y"].nunique() < 2:
        return None, None
    mt = max(60, min(min_train, len(df) // 3))
    if len(df) < mt + 50:
        return None, None
    X, yv = df[cols].values, df["y"].astype(int).values
    pr = pd.Series(np.nan, index=df.index); i = mt
    while i < len(df) - 1:
        te = max(1, i - 5)
        if len(np.unique(yv[:te])) < 2: i += refit; continue
        m = (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0)
             if model == "hgb" else RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=30, n_jobs=-1))
        m.fit(X[:te], yv[:te]); pr.iloc[i:i+refit] = m.predict_proba(X[i:i+refit])[:, 1]; i += refit
    o = pd.concat({"y": df["y"], "p": pr}, axis=1).dropna()
    if len(o) < 30 or o["y"].nunique() < 2:
        return None, None
    return roc_auc_score(o["y"], o["p"]), average_precision_score(o["y"], o["p"])


def main(mode="cached"):
    feats = build_features(get_panel(mode))
    F = rich_features(feats).join(_meta_features(feats))
    base = [c for c in RICH if c in F.columns] + list(_meta_features(feats).columns)
    optional = [c for c in F.columns if c.startswith(OPT_PREFIX)]
    y, _ = _target(feats, "flare")

    min_train = 750
    F_eval = F
    if optional:
        mask = F[optional].notna().all(axis=1)
        if mask.sum() >= 120:
            F_eval = F.loc[mask]
            y = y.loc[mask]
            min_train = max(60, min(750, len(F_eval) // 3))
            print(f"\n  eval window: {len(F_eval)} days with optional feats (min_train={min_train})")

    print("\n" + "=" * 62)
    print("  BEFORE / AFTER — impact of new industry features on flare detection")
    print("=" * 62)
    print(f"  optional features present: {optional or '(none — enable use_demand/use_operational/use_news + fetch data)'}")
    print(f"\n  {'model':5s} {'AUC before':>11s} {'AUC after':>10s} {'delta':>7s} {'AP before':>10s} {'AP after':>9s}")
    verdict = {}
    for model in ("hgb", "rf"):
        a0, p0 = _flare(F_eval, base, model, y, min_train=min_train)
        a1, p1 = _flare(F_eval, base + optional, model, y, min_train=min_train) if optional else (a0, p0)
        if a0 is None or a1 is None:
            print(f"  {model:5s}       (insufficient data for walk-forward)")
            verdict[model] = None
            continue
        verdict[model] = (a1 - a0, p1 - p0)
        print(f"  {model:5s} {a0:>11.3f} {a1:>10.3f} {a1-a0:>+7.3f} {p0:>10.3f} {p1:>9.3f}")
    improved_both = (
        optional
        and verdict.get("hgb") is not None
        and verdict.get("rf") is not None
        and verdict["hgb"][0] > 0.002 and verdict["rf"][0] > 0.002
        and verdict["hgb"][1] > 0 and verdict["rf"][1] > 0
    )
    print(f"\n  VERDICT: {'KEEP — improves AUC+AP on both models' if improved_both else 'DO NOT KEEP — not a robust improvement (or no optional feats on)'}")

    anom = A.detect(feats, min_votes=2); ae = A.evaluate(feats, anom)
    vol = V.evaluate(feats, horizon=10)
    print(f"\n  context — anomaly recall {ae['events_caught']}/{ae['n_events_known']} "
          f"(p={ae['event_recall_vs_random_p']}), idio_sep {ae['idio_forward_residual_separation']}x; "
          f"vol best {vol.get('best_model')} skill {vol.get('best_skill_vs_naive')}")
    return verdict


if __name__ == "__main__":
    import sys; main(sys.argv[1] if len(sys.argv) > 1 else "cached")
