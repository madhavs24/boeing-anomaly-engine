"""Causal expanding-window Gaussian HMM market regime labels."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REGIME_NAMES = ("calm", "stress", "crisis")


def fit_expanding_regime(feats: pd.DataFrame, min_train: int = 500, refit: int = 63) -> pd.Series:
    cols = [c for c in ("rv21", "vix", "resid_rv21") if c in feats.columns]
    if len(cols) < 2:
        return pd.Series("calm", index=feats.index, dtype=object)
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return _rule_regime(feats)

    Xdf = feats[cols].replace([np.inf, -np.inf], np.nan).ffill().dropna()
    X = Xdf.values
    labels = pd.Series(index=Xdf.index, dtype=object)
    model, scaler, state_map = None, None, {}

    for i in range(min_train, len(Xdf)):
        if i % refit == 0 or model is None:
            try:
                scaler = StandardScaler().fit(X[:i])
                Xs = scaler.transform(X[:i])
                hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=0)
                hmm.fit(Xs)
                vix_i = cols.index("vix") if "vix" in cols else 0
                order = np.argsort(hmm.means_[:, vix_i])
                state_map = {int(order[j]): REGIME_NAMES[j] for j in range(3)}
                model = hmm
            except Exception:
                pass
        if model is not None and scaler is not None:
            st = int(model.predict(scaler.transform(X[i:i + 1]))[0])
            labels.iloc[i] = state_map.get(st, "stress")

    return labels.ffill().fillna("calm").reindex(feats.index).ffill().fillna("calm")


def _rule_regime(feats: pd.DataFrame) -> pd.Series:
    vix = feats["vix"] if "vix" in feats else pd.Series(20, index=feats.index)
    q33 = vix.expanding(min_periods=252).quantile(0.33)
    q66 = vix.expanding(min_periods=252).quantile(0.66)
    out = pd.Series("stress", index=feats.index, dtype=object)
    out[vix <= q33] = "calm"
    out[vix >= q66] = "crisis"
    return out


def regime_one_hot(regime: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=regime.index)
    for name in REGIME_NAMES:
        out[f"regime_{name}"] = (regime == name).astype(float)
    return out
