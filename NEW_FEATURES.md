# Implemented Ideas — new features & detectors

*What I built from IMPROVEMENT_IDEAS.md, what's verified, and the honest results. Everything is
causal and integrated behind flags so the default pipeline is unchanged until you A/B it.*

## First, the honest finding on "reuse existing data" ideas

I A/B-tested the ideas that reuse the price/cross-asset data we already have. **They don't
robustly help** (same lesson as microstructure):

| Idea | flare AUC hgb | flare AUC rf | verdict |
|---|---|---|---|
| base | 0.623 | 0.640 | — |
| #18 commercial vs defense residual split | 0.623 | 0.643 | flat/noise |
| #13 CUSUM as a feature | 0.628 | 0.636 | mixed (helps one, hurts other) |

Rule enforced: keep only if it improves on **both** models. None did → **not wired into the model.**
The signal already extractable from price is extracted. Real gains need **new data** (below).

## Built and integrated (new data — where the gains actually are)

| Module | Idea | What it adds | Status |
|---|---|---|---|
| `src/demand.py` | #4 | TSA daily air-travel demand: `demand_yoy`, `demand_mom`, `demand_z` (shock) | ✅ **real data** via bcantoni/tsa-data mirror (2746 days cached); flags **off** after A/B |
| `src/operational.py` | #1,#2 | SEC EDGAR (CIK 12927): `op_revenue_yoy`, `op_ocf`, `op_days_since` — **only known on filing date** (no lookahead) | ✅ **real EDGAR** (2009–2026 filings); flags **off** after A/B |
| `src/news.py` | #9 | Free RSS + FinBERT → `news_sent`, `news_vol_z`, `news_neg_share` | ✅ **real headlines** + FinBERT; flags **off** after A/B |
| `src/changepoint.py` | #13 | CUSUM regime-break detector (caught the 2024-01-08 door-plug day) | ✅ built — use as an **alerting overlay** (coverage), not an AUC booster |
| `src/events.py` | #17 | event label set expanded 17 → 27 real BA dates (charges, NTSB, DOJ, defects) — fairer recall measurement | ✅ done |

## A/B results (2026-06-24 run, robustness rule: improve flare AUC **and** AP on **both** hgb and rf)

Baselines (flags off, full panel): flare AUC hgb **0.632** / rf **0.641**; anomaly recall **5/27** (p=0.035); HARX vol skill **0.226**.

| Feature group | hgb AUC Δ | rf AUC Δ | hgb AP Δ | rf AP Δ | Verdict | Action |
|---|---|---|---|---|---|---|
| demand + operational | −0.003 | +0.038 | −0.057 | −0.050 | **DROP** (mixed / hurts hgb) | `use_demand: false`, `use_operational: false` |
| news (209-day eval window*) | +0.010 | +0.000 | +0.005 | +0.000 | **DROP** (rf flat) | `use_news: false` |

\*News RSS history is shallow (~400d); `src.compare` evaluates only days where news features are non-null.

**Per-group assessment:** demand+operational → **noise** (rf AUC up, but hgb down and both AP down). news → **noise** on rf (hgb slightly up, not robust). **None kept in `RICH` or production flags.**

Data fetch verified:
```powershell
python -m src.demand        # 2746 real TSA days (not synthetic)
python -m src.operational   # EDGAR fundamentals through 2026-04-22
python -m src.news          # FinBERT-scored Boeing headlines
python -m src.compare       # before/after harness
```

Both data modules are wired into `build_features` behind config flags (default **off**):
```yaml
use_demand: false        # set true after you've fetched real TSA data
use_operational: false   # set true to pull SEC EDGAR fundamentals
```
Turn a flag on → the features appear in `feats`; then A/B them.

## How to actually use them (needs your machine's network)

1. **Fetch:** `python -m src.demand` and `python -m src.operational` (verify they cache real data).
2. **Enable:** set `use_demand: true` / `use_operational: true` in `config.yaml`.
3. **A/B test:** add `demand.DEMAND_FEATS` (and `operational.OP_FEATS`) to `direction_pro.RICH`,
   then run `python -m src.models` and compare flare AUC/avg-precision **with vs without**, on
   **both** hgb and rf. Keep only if it improves on both (the discipline rule). See Cursor prompt #11.

## Also built — FREE news sentiment + a before/after tool

| Module | Idea | What it adds | Cost |
|---|---|---|---|
| `src/news.py` | #9 | Free RSS headlines (Yahoo/Google, **no API key**) + FinBERT (or lexicon) → `news_sent`, `news_vol_z`, `news_neg_share`, causal | **100% free** |
| `src/compare.py` | — | **Before/after** flare AUC/AP on hgb+rf for whatever industry features are enabled, with a KEEP/DON'T-KEEP verdict | — |

Enable news with `use_news: true` (needs network to fetch RSS). To see the impact of ANY enabled
feature group: **`python -m src.compare`** (see Cursor prompt #13).

## Still to build (scarce/paid data) — Cursor prompts in CURSOR_PROMPTS.md

- **#6 BA options IV/skew/put-call** — free only for *today's* snapshot (yfinance), **not** 10y history
  to train on; so live-forward only, not backtestable for free.
- **#10 NTSB/aviation incidents**, **#11 Google Trends / social**, **#16 TFT flare challenger**.

## Bottom line

Industry data modules are **implemented, fetch real data, and are A/B-tested**. Under the robustness
rule (both hgb and rf must improve), **none passed** — flags stay off, `RICH` unchanged. Cached
parquet files in `data/processed/` (`demand_raw.parquet`, `operational.parquet`, `news_daily.parquet`)
speed up re-runs. Install dev extras with `pip install -r requirements-dev.txt` for FinBERT + pytest.
</content>
