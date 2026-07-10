# Boeing (BA) Anomaly Engine

A single-name research & near-real-time monitoring engine for **The Boeing Company (NYSE: BA)**.
It detects Boeing-specific anomalies, forecasts volatility, and tests direction prediction —
honestly. Built on free data, runs on a laptop. See **[RESEARCH.md](RESEARCH.md)** for the full
reasoning (drivers, features, data, model survey, time-frame decision).

> Research / education only. Not investment advice. No real trades.

## What it does, ranked by how much to trust it

1. **Anomaly detection (highest trust).** Strips the market + sector out of Boeing's return to
   get the *idiosyncratic residual*, then flags abnormal days with a calibrated false-alarm
   rate (conformal p-value) **plus** a multivariate Isolation Forest. Two votes = high
   confidence. This catches Boeing-specific events (groundings, FAA actions, strikes, surprises).
2. **Volatility forecast (medium-high).** HAR-RV baseline for next-N-day turbulence, scored
   out-of-sample against the naive baseline.
3. **Direction (low, shown honestly).** Logistic baseline vs. coin flip / last-move. Reported
   plainly even when it doesn't beat chance.

## Quick start

**Project path (local, not OneDrive):** `C:\Users\singhm\Desktop\Econ project\boeing-anomaly-engine`

```bash
cd "C:\Users\singhm\Desktop\Econ project\boeing-anomaly-engine"
.\setup.ps1                  # Windows: create .venv + install deps
# or: pip install -r requirements.txt

python -m src.data live      # one-time: fetch real BA data -> data/processed/panel.parquet
python -m src.run            # run the engine (anomaly + volatility + direction)
python -m src.experiment     # auto-tuner: sweep configs, leaderboard -> results/
python -m src.dashboard      # build the interactive dashboard.html (open in any browser)
```

Outputs land in `results/`: `summary.json`, `anomalies.csv`, `leaderboard.json`,
`experiments_*.csv`. The dashboard is a single self-contained `dashboard.html` (no server).
Run any stage alone, e.g. `python -m src.anomaly`, `python -m src.volatility`.

**Real-data results so far (see [RESULTS.md](RESULTS.md)):** volatility HARX OOS R²≈0.56 (10d);
anomaly detector flags ~1.5% of days at 4–4.5× the forward-vol of calm days and catches every
major idiosyncratic shock; next-day direction ≈ coin flip (honest).

## Data (free only)

- **Yahoo chart API** (browser User-Agent, not yfinance) — BA, peers (RTX, GE, Airbus/EADSY,
  LMT, NOC), SPY, sector ETF ITA, plus daily and recent intraday bars.
- **FRED** (CSV, no key) — oil, 10Y yield, dollar, high-yield credit spread.
- Synthetic generator for fully offline testing.

Edit `config.yaml` to change peers, macro series, history length, the anomaly false-alarm
rate, or the volatility horizon.

## Layout

```
boeing-anomaly-engine/
  RESEARCH.md         # the deep research report (read this)
  config.yaml         # tickers, macro series, knobs
  requirements.txt
  src/
    data.py           # free-data loader (Yahoo + FRED) + synthetic fallback
    features.py       # causal features; idiosyncratic-residual spine
    anomaly.py        # conformal + Isolation Forest ensemble
    volatility.py     # HAR-RV forecast vs naive
    direction.py      # honest direction baseline vs coinflip
    run.py            # orchestrator + status report
  results/            # generated outputs
```

## Roadmap (priority order, see RESEARCH.md §7)

1. LSTM-autoencoder detector → 3-way ensemble vote.
2. Finnhub news + FinBERT sentiment layer (needs your free API key) — biggest accuracy upgrade.
3. Intraday near-real-time monitor (1–15 min loop).
4. Deep challengers (TFT) for vol/direction — adopt only if they beat the baselines.
5. Interactive dashboard once the engine is trusted.

## Guardrails (why it's trustworthy)

Causal features only (no lookahead), walk-forward validation, honest baselines always shown,
calibrated false-alarm rates, and every output ships with its real, modest score.
</content>
