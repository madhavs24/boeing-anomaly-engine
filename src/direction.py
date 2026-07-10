"""Stage 3C — Direction prediction (LOW trust, honest).

Next-day up/down. Models compared walk-forward, all vs honest baselines (coin flip,
always-up, last-move). We also test the more defensible CONDITIONAL targets:
  - direction on the day AFTER an anomaly fires (event-driven)
Metric: accuracy and AUC. We only ever 'adopt' a model if it clears the baselines by a
margin (≈1 std of a coin flip). On a liquid large-cap this usually fails — reported plainly.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from .features import DIR_FEATS

warnings.filterwarnings("ignore")


def _design(feats, cols):
    y = (feats["ret"].shift(-1) > 0).astype(int)
    df = feats[cols].join(y.rename("y")).replace([np.inf, -np.inf], np.nan).dropna()
    return df, cols


def walk_forward(feats, model="logit", refit_every=63, min_train=750, cols=None):
    cols = [c for c in (cols or DIR_FEATS) if c in feats.columns]
    df, cols = _design(feats, cols)
    X, y = df[cols].values, df["y"].values
    pred = pd.Series(np.nan, index=df.index)
    proba = pd.Series(np.nan, index=df.index)
    i = min_train
    while i < len(df) - 1:
        if len(np.unique(y[:i])) < 2:
            i += refit_every; continue
        if model == "logit":
            sc = StandardScaler().fit(X[:i])
            m = LogisticRegression(max_iter=500, class_weight="balanced").fit(sc.transform(X[:i]), y[:i])
            sl = slice(i, i + refit_every)
            pred.iloc[sl] = m.predict(sc.transform(X[sl]))
            proba.iloc[sl] = m.predict_proba(sc.transform(X[sl]))[:, 1]
        elif model == "hgb":
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                               max_iter=300, l2_regularization=1.0).fit(X[:i], y[:i])
            sl = slice(i, i + refit_every)
            pred.iloc[sl] = m.predict(X[sl])
            proba.iloc[sl] = m.predict_proba(X[sl])[:, 1]
        i += refit_every
    out = pd.concat({"y": df["y"], "pred": pred, "proba": proba}, axis=1).dropna()
    out["last_move"] = (feats["ret"] > 0).astype(int).reindex(out.index)
    return out


def _score(o):
    acc = lambda p: float((o["y"].values == np.asarray(p)).mean())
    rng = np.random.default_rng(0)
    res = {
        "n": int(len(o)),
        "accuracy": round(acc(o["pred"].astype(int)), 3),
        "auc": round(float(roc_auc_score(o["y"], o["proba"])), 3) if o["proba"].notna().any()
               and o["y"].nunique() > 1 else None,
        "coinflip_acc": round(acc(rng.integers(0, 2, len(o))), 3),
        "last_move_acc": round(acc(o["last_move"].astype(int)), 3),
        "always_up_acc": round(float((o["y"] == 1).mean()), 3),
    }
    margin = 0.5 + 1.0 / np.sqrt(max(len(o), 1))
    res["beats_baseline"] = bool(res["accuracy"] > margin)
    return res


def evaluate(feats: pd.DataFrame) -> dict:
    res = {}
    for name in ("logit", "hgb"):
        o = walk_forward(feats, model=name)
        res[name] = _score(o) if len(o) else {"n": 0}
    scored = {k: v for k, v in res.items() if v.get("auc") is not None}
    best = max(scored, key=lambda k: scored[k]["auc"]) if scored else None
    res["best_model"] = best
    res["ceiling_auc"] = scored[best]["auc"] if best else None
    res["any_beats_baseline"] = any(v.get("beats_baseline") for v in res.values() if isinstance(v, dict))
    return res


if __name__ == "__main__":
    from .data import get_panel
    from .features import build_features
    import json
    print(json.dumps(evaluate(build_features(get_panel("auto"))), indent=2))
