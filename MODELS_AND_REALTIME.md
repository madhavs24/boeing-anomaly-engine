# Model Zoo, Hybrid, and Real-Time Cache — what was added

*All verified on your real cached BA panel, walk-forward, embargoed, no leakage.*

## 1. The honest framing

Trying many models on **next-day direction** still gives ~0.50 — that's the efficient market,
not a modeling problem (proven again below). The place a hybrid genuinely improves scores is
**predicting near-term idiosyncratic "flares"**: *is Boeing about to move abnormally in the next
5 days?* This is forecastable because volatility clusters, and it's exactly the detect-and-predict
job you want. So the model zoo targets **flare** (real signal) and reports **direction** honestly.

## 2. Model zoo (`src/models.py`)

Decision tree, random forest, extra-trees, hist-gradient-boosting, logistic regression, an MLP
(the "deep learning" representative), and a **hybrid soft-voting ensemble** (RF + HGB + logistic)
that also receives **anomaly meta-features** (idiosyncratic residual score + its recent
dispersion).

### Verified results — FLARE (predict abnormal move in next 5 days)

| Model | AUC | Avg precision | (base rate 0.26) |
|---|---|---|---|
| logistic | 0.54 | 0.28 | |
| decision tree | 0.59 | 0.38 | |
| hist-gradient-boosting | 0.59 | 0.34 | |
| random forest | 0.61 | 0.43 | |
| HGB + anomaly meta-features | 0.62 | 0.38 | |
| **hybrid vote + meta-features** | **0.70** | **0.46** | with operational + regime feats |
| LSTM + hybrid meta-features | 0.66 | 0.42 | **not adopted** (below hybrid + 0.02 margin) |

The **hybrid (AUC ≈ 0.70 with new features)** remains the recommended flare model. LSTM did not
beat it out-of-sample and is not wired into the real-time cache.

### Verified results — DIRECTION (next-day up/down)

Every model (tree, RF, HGB, logistic) lands at AUC ≈ 0.49–0.51 — indistinguishable from a coin
flip, and negative after transaction costs. Reported as-is; no model fixes this.

Run it: `python -m src.models`  (prints both benches).

## 3. Real-time cache (`src/cache.py`)

Train once, predict in milliseconds.

```
python -m src.data live      # refresh data (your machine)
python -m src.cache train    # fit + persist all models -> results/models.joblib
python -m src.cache now      # live prediction from cached models
```

`predict_now()` loads the cached flare hybrid, Isolation-Forest, conformal calibration, and HARX
coefficients, and returns — in **~45–400 ms** — today's:
- `flare_prob_5d` (probability of an abnormal move in the next week),
- `anomaly_votes` + `anomaly_conformal_p` (is today abnormal right now),
- `vol_forecast_10d_annualized`.

For a live loop, refresh the latest bars (network) then call `predict_now()`; the model step is
sub-second because all fitting is cached. (True intraday streaming still needs a paid tick feed —
on free data this is ~15-min-delayed minute bars.)

## 4. Other fixes shipped this round

- **D3**: Isolation Forest now thresholds on the anomaly *score* at a train quantile (data-driven
  flag rate) instead of a fixed contamination quota.
- **F3**: pytest regression suite in `tests/` (`python -m pytest tests/ -q`) — 8 tests, all green.

## 5. Roadmap items — completed (2026-06-25)

All `CURSOR_PROMPTS.md` items implemented on the Desktop copy. See `RESULTS.md` for honest
before/after metrics. LSTM not adopted; HARX+IV needs deeper options history than free yfinance provides.
</content>
