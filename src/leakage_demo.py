"""LEAKAGE DEMONSTRATION — why papers report 80-95% directional accuracy, and why it's fake.

Runs the SAME data three ways and prints the accuracy each produces:

  1. HONEST       walk-forward, causal features only            -> ~0.50 (the truth)
  2. LEAKY-shuffle random train/test split + global scaling     -> still ~0.50 here*
  3. LEAKY-future one feature that peeks 1 day ahead + shuffle  -> ~0.90 (the fake)

*Shuffling alone doesn't inflate much when the features genuinely carry no signal — the real
killer is a FUTURE-PEEKING FEATURE (a feature computed with a window that includes t+1, or any
quantity not knowable at prediction time). Combined with a shuffled split (so the leaked rows
land in both train and test), it manufactures 90%+ that collapses to 50% live.

    python -m src.leakage_demo
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .features import build_features, DIR_FEATS

warnings.filterwarnings("ignore")


def _honest(feats, cols, refit=63, min_train=750):
    px = feats["price"]; y = (px.shift(-1) > px).astype(int)
    df = feats[cols].join(y.rename("y")).replace([np.inf, -np.inf], np.nan).dropna()
    X, yv = df[cols].values, df["y"].values
    pred = pd.Series(np.nan, index=df.index); i = min_train
    while i < len(df) - 1:
        if len(np.unique(yv[:i])) < 2:
            i += refit; continue
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                           l2_regularization=1.0).fit(X[:i], yv[:i])
        pred.iloc[i:i + refit] = m.predict(X[i:i + refit])
        i += refit
    o = pd.concat({"y": df["y"], "pred": pred}, axis=1).dropna()
    return float((o["y"] == o["pred"]).mean()), len(o)


def _leaky_shuffle(feats, cols):
    px = feats["price"]; y = (px.shift(-1) > px).astype(int)
    df = feats[cols].join(y.rename("y")).replace([np.inf, -np.inf], np.nan).dropna()
    X = StandardScaler().fit_transform(df[cols].values)            # LEAK: scaler sees all data
    Xtr, Xte, ytr, yte = train_test_split(X, df["y"].values, test_size=0.2,
                                          random_state=0, shuffle=True)  # LEAK: shuffled time
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1).fit(Xtr, ytr)
    return float((m.predict(Xte) == yte).mean())


def _leaky_future(feats, cols):
    px = feats["price"]; r = feats["ret"]; y = (px.shift(-1) > px).astype(int)
    # LEAK: centered window includes t+1, then shifted -1 -> the feature literally knows tomorrow
    leak = ((r - r.rolling(11, center=True).mean()) / (r.rolling(11, center=True).std() + 1e-9)).shift(-1)
    df = feats[cols].join(leak.rename("LEAK")).join(y.rename("y")).replace([np.inf, -np.inf], np.nan).dropna()
    X = df[cols + ["LEAK"]].values
    Xtr, Xte, ytr, yte = train_test_split(X, df["y"].values, test_size=0.2,
                                          random_state=0, shuffle=True)
    m = HistGradientBoostingClassifier(max_iter=400).fit(Xtr, ytr)
    return float((m.predict(Xte) == yte).mean())


def run(feats=None):
    if feats is None:
        from .data import get_panel
        feats = build_features(get_panel("auto"))
    cols = [c for c in DIR_FEATS if c in feats.columns]
    honest, n = _honest(feats, cols)
    shuf = _leaky_shuffle(feats, cols)
    fut = _leaky_future(feats, cols)
    print("\n" + "=" * 60)
    print("  LEAKAGE DEMONSTRATION — Boeing next-day direction")
    print("=" * 60)
    print(f"  1. HONEST  (walk-forward, causal)            acc = {honest:.3f}   <- the truth (n={n})")
    print(f"  2. LEAKY   (shuffled split + global scaling)  acc = {shuf:.3f}")
    print(f"  3. LEAKY   (+ one future-peeking feature)     acc = {fut:.3f}   <- the fake 'high accuracy'")
    print("\n  Takeaway: the jump to ~0.90 comes from a feature that knows the future,")
    print("  not from skill. Honest walk-forward is the only number that survives live.")
    return {"honest": honest, "leaky_shuffle": shuf, "leaky_future": fut, "n": n}


if __name__ == "__main__":
    run()
