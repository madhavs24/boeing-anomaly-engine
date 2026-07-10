# Cursor Prompts — for the items that need your machine's network / data

These are the open items I can't do from my sandbox (no internet) or that are a build of their
own. Each block is a copy-paste prompt for Cursor. Run them in order; #1 is the quick win.

---

## 1. Re-fetch adjusted data (closes A1, A2, A3) — DO THIS FIRST

You actually don't need Cursor for this — just run, in the project folder:

```
python -m src.data live
python -m src.run cached
python -m src.cache train
```

If the FRED series time out again, give Cursor this:

> In `src/data.py`, the FRED CSV fetch (`_fred`) sometimes times out for `DCOILWTICO`,
> `DTWEXBGS`, and `BAMLH0A0HYM2`. Add a 3-try exponential backoff with a 30s timeout, and a
> fallback that pulls the same series from Yahoo where possible (e.g. `CL=F` for WTI, `DX-Y.NYB`
> for the dollar index). Keep everything causal and don't change the panel schema. Then run
> `python -m src.data live` and confirm `data/processed/panel.parquet` has columns
> `oil, dollar, hy_oas` with <5% NaN.

---

## 2. E1 — Boeing operational data (deliveries, FCF, FAA, backlog) [highest value]

> Create `src/operational.py` for The Boeing Company. Pull, from FREE sources, a monthly/
> quarterly history (2015–present) of: (a) monthly commercial aircraft deliveries and orders
> from Boeing's investor "Orders & Deliveries" data, (b) quarterly free cash flow, revenue, and
> backlog from SEC EDGAR company facts API
> (`https://data.sec.gov/api/xbrl/companyfacts/CIK0000012927.json`, use a descriptive
> User-Agent), and (c) a hand-maintained CSV `data/faa_events.csv` of FAA production-cap and
> certification dates. Resample everything to a daily index with forward-fill and a
> `days_since_release` column so the data is strictly causal (a figure is only known on/after its
> release date — never backfill into the past). Expose `build_operational_features()` returning a
> daily DataFrame, and join it into `features.build_features` behind a config flag
> `use_operational: true`. Add the new columns to `DIR_FEATS` and to the flare model's feature
> set in `src/models.py`. Re-run `python -m src.run cached` and `python -m src.models` and report
> whether flare AUC and anomaly recall improve. Keep all existing tests passing.

## 3. E2 — Boeing-specific options implied volatility

> Add `src/options.py` that fetches Boeing (BA) option-chain implied volatility from a free/
> low-cost source (try `yfinance` `Ticker('BA').option_chain()` for near-the-money IV, or the
> CBOE delayed quotes). Build a daily `ba_iv` (30-day ATM IV) and `ba_iv_skew` (25-delta put−call)
> series, cache to `data/processed/ba_iv.parquet`, and add them as features. Wire `ba_iv` into
> the HARX volatility model in `src/volatility.py` as an additional regressor (call it HARX+IV)
> and compare its walk-forward skill_vs_naive to plain HARX. Report whether BA-specific IV beats
> market VIX. Causal only; no lookahead.

## 4. (Optional) Deep sequence model — LSTM / GRU / TFT challenger

> In `src/models.py`, add a PyTorch LSTM classifier (`pip install torch`) for the `flare` target:
> input = last 20 days of the feature matrix, output = P(flare in next 5d). Train walk-forward
> with the same embargo and refit cadence as the other models, early stopping on a validation
> tail. Add it to `bench()` as model name `lstm`. Only recommend adopting it if its out-of-sample
> AUC beats the current best (`hybrid_vote`, AUC≈0.64) by more than 0.02 with overlapping
> confidence intervals checked via `validation.moving_block_bootstrap_ci`. Report honestly if it
> doesn't beat the simpler ensemble.

## 5. (Optional) Regime conditioning (closes G2)

> Add a market-regime label using the existing causal HMM idea: fit a 3-state Gaussian HMM on
> `[rv21, vix, resid_rv21]` with an expanding window (no lookahead), producing a daily
> `regime in {calm, stress, crisis}`. Add `regime` as a one-hot feature to the flare and
> direction models, and let `anomaly.detect` use a regime-specific conformal alpha. Re-run the
> bench and report whether regime conditioning improves flare AUC or anomaly precision.

---

## 6. F4 — get the project out of OneDrive's way (operational) — DONE

Moved to `C:\Users\singhm\Desktop\Econ project\boeing-anomaly-engine`. Run all commands from there.

---

## What's already DONE in code (no Cursor needed)

D3 (score-threshold Isolation Forest), F3 (pytest suite in `tests/`), the model zoo + hybrid
(`src/models.py`), and the real-time cache (`src/cache.py`). See `MODELS_AND_REALTIME.md`.
</content>
