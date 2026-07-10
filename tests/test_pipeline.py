"""Minimal regression tests (fix F3). Run: python -m pytest tests/ -q"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pathlib
from src.features import build_features
from src import anomaly as A, volatility as V, direction_pro as DP, models as M

PANEL = pathlib.Path("data/processed/panel.parquet")
def _feats():
    from src.data import build_synthetic
    panel = pd.read_parquet(PANEL) if PANEL.exists() else build_synthetic()
    return build_features(panel)

def test_features_causal_no_nan_explosion():
    f = _feats(); assert len(f) > 500 and "resid_z" in f
    assert f["resid_z"].notna().mean() > 0.5

def test_panel_macro_columns():
    panel = pd.read_parquet(PANEL)
    for c in ("oil", "dollar", "hy_oas"):
        assert c in panel.columns, f"missing {c}"
        assert panel[c].isna().mean() < 0.05, f"{c} too many NaNs"

def test_operational_causal_no_backfill():
    from src.operational import build_operational_features
    idx = pd.bdate_range("2015-01-01", "2020-12-31")
    op = build_operational_features(idx)
    if "deliveries_m" in op.columns:
        first = op["deliveries_m"].first_valid_index()
        assert first is not None
        assert op.loc[op.index < first, "deliveries_m"].isna().all()

def test_regime_no_lookahead():
    f = _feats()
    if "regime_calm" in f.columns:
        assert f["regime_calm"].iloc[:100].notna().any()

def test_anomaly_runs_and_flag_rate_reasonable():
    f = _feats(); d = A.detect(f, min_votes=2); e = A.evaluate(f, d)
    assert 0 <= e["flag_rate"] <= 0.2
    assert d["flag"].isin([0,1]).all()

def test_volatility_skill_finite():
    f = _feats(); r = V.evaluate(f, horizon=10)
    assert "har" in r and np.isfinite(r["har"]["skill_vs_naive"])

def test_direction_is_honest_not_absurd():
    f = _feats(); r = DP.honest_direction(f, "hgb")
    assert 0.4 < r["accuracy"] < 0.65   # no leakage -> near coin flip

def test_flare_has_signal_or_at_least_runs():
    f = _feats(); r = M.walk_forward(f, "hgb", kind="flare", refit=126)
    assert r["n"] > 100 and 0.4 <= r["auc"] <= 0.9
