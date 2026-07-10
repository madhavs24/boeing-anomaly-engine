# Direction Prediction — Deep Dive & Honest Findings

*You asked to chase 80% directional accuracy on Boeing. I read the literature and tested every
legitimate lever on your real BA data (2016–2026, walk-forward, no leakage). This documents
what's real, what's fake, and where the honest ceiling is. New code: `src/direction_pro.py`,
`src/leakage_demo.py`, `src/news.py`.*

## Bottom line

**80% genuine next-day directional accuracy on Boeing is not achievable.** Properly-validated
models in the literature reach ~50–60% (exceptional ensembles ~63–67%). On your data, every
honest configuration lands at ~0.50. The papers showing 85–95% have identifiable leakage. I
will not ship a leaky model that prints a fake 80% — it loses money the instant it's used.

## 1. What the research actually reports

- Legitimate daily direction: **~50–60%**; best sentiment-augmented ensembles **~63–67%**
  ([How effective is ML in markets](https://pmc.ncbi.nlm.nih.gov/articles/PMC10826674/),
  [ensemble VAE+Transformer+LSTM](https://arxiv.org/pdf/2503.22192)).
- The 80–95% claims come from **data leakage**: shuffling time-series before the split,
  scaling/normalizing across the whole dataset, or future-peeking features
  ([leakage & lookahead](https://medium.com/@kyle-t-jones/data-leakage-lookahead-bias-and-causality-in-time-series-analytics-76e271ba2f6b)).
- Finance replication crisis: **65% of anomalies fail single-test significance, 82% under
  multiple-testing**; returns fall ~58% post-publication
  ([backtest overfitting](https://www.researchgate.net/publication/275302374_Pseudo-Mathematics_and_Financial_Charlatanism_The_Effects_of_Backtest_Overfitting_on_Out-of-Sample_Performance)).
- Honest validation requires walk-forward + transaction costs; daily predictability often
  doesn't survive costs ([replication & costs](https://nature.com/articles/s41598-025-14872-6)).

## 2. What your Boeing data does (real, walk-forward)

### Honest models — adding features does NOT create signal

| Model | Features | Accuracy | AUC | Majority baseline |
|---|---|---|---|---|
| Logistic | 10 core | 0.499 | 0.493 | 0.505 |
| Gradient boosting | 10 core | 0.496 | 0.487 | 0.505 |
| Logistic | **24 rich** (lags, momentum, RSI, MA-gaps, peers, DOW) | 0.499 | 0.493 | 0.505 |
| Gradient boosting | **24 rich** | 0.496 | 0.487 | 0.505 |

Going from 10 to 24 features changed nothing — the signal isn't there to find.

### Selective prediction doesn't rescue it

Trading only the model's most-confident days makes it *no better* (often worse): top-10%
most-confident accuracy ≈ 0.45–0.52. The model's confidence is uncorrelated with being right.

### The leakage demo (run `python -m src.leakage_demo`)

| Method | Accuracy |
|---|---|
| **Honest** (walk-forward, causal) | **0.49** ← the truth |
| Leaky: shuffled split + global scaling | 0.52 |
| Leaky: **+ one future-peeking feature** | **0.89** ← the fake "high accuracy" |

The leap to ~0.90 comes entirely from a single feature that knows tomorrow — not skill. This
is exactly how the 80–95% papers are produced. Now you can spot and reproduce the trick.

## 3. The one genuinely defensible edge: event-conditional

Instead of predicting every day, predict the drift *after* an anomaly fires:

| Horizon after a flag | Up-rate after flag | Baseline up-rate | Mean return |
|---|---|---|---|
| 1 day | 0.50 | 0.51 | −0.02% |
| **5 days** | **0.41** | 0.53 | −0.34% |
| 10 days | 0.54 | 0.54 | +0.13% |
| 20 days | 0.45 | 0.54 | +1.26% |

After an anomaly, BA is meaningfully **more likely to keep falling over the next 5 days**
(41% up vs 53% baseline — a ~12-point tilt). This is small-sample (n=80) and partly mechanical
(most anomalies are bad-news crashes), but it's the most real directional signal in the project
and worth developing carefully.

## 4. How to push direction as far as honestly possible

1. **News / FinBERT sentiment** (`src/news.py`, needs a free Finnhub key). The one input with
   repeated literature support for a few points of lift — and it can add *lead time* on events
   the price hasn't reacted to yet. Expected realistic ceiling with it: **~55–60%**.
2. **Develop the event-conditional bet** — more events, proper significance testing, separate
   bad-news vs good-news anomalies, add costs.
3. **Longer horizons / trend** — smoother, but beware: "60%+" over 20 days is mostly the *base
   rate* (BA drifted up), not skill. We report skill vs the majority baseline, not raw accuracy.

## 5. What I recommend

Pursue the **news layer + event-conditional** combo and judge it honestly against baselines and
costs. If it reaches ~55–60% with real lead time on events, that's a genuinely useful,
defensible result — far more valuable than a fake 80% that evaporates live. To start the news
layer: get a free key at finnhub.io and set `FINNHUB_API_KEY`, then run `python -m src.news`.

*Not investment advice. No real trades.*
</content>
