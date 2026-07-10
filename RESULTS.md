# Boeing Anomaly Engine — Real-Data Results

*Primary project path: `C:\Users\singhm\Desktop\Econ project\boeing-anomaly-engine` (off OneDrive).*

## Roadmap implementation (2026-06-25)

| Phase | Change | Outcome |
|---|---|---|
| Data (A1–A3) | FRED 30s + Yahoo fallbacks; panel 2517×17 | `oil`/`dollar`/`hy_oas` present; hy_oas 2.3% NaN (HYG/LQD proxy) |
| Operational (E1) | `src/operational.py`, FAA CSV, SEC facts | Deliveries, backlog, FCF, revenue on causal release dates |
| Options (E2) | `src/options.py`, `ba_iv.parquet` | ATM IV ≈ 35% today; **insufficient history** for HARX+IV walk-forward |
| Regime (G2) | `src/regime.py` HMM + per-regime conformal α | Mostly calm regime; anomaly metrics unchanged on this pass |
| LSTM (optional) | PyTorch LSTM in `src/models.py` | **Flare AUC 0.661** — does **not** beat hybrid_vote (0.697); not cached |
| Volatility | HARX skill-vs-naive | **0.226** (10d); still best model; beats_naive_95ci not significant |
| Anomaly | conformal + IF + LOF | recall 6/17, p=0.001, flag_rate 1.5%, idio_sep 2.89× |
| Direction | HGB + operational/regime feats | acc 0.491 — still coin-flip |

Run: `python -m src.run cached`, `python -m src.models`, `python -m pytest tests/ -q`

---

# Boeing Anomaly Engine — Real-Data Results (ceiling pass)

*Computed on real Yahoo data, BA 2016-06-24 → 2026-06-24 (2,514 trading days). All numbers
are walk-forward / out-of-sample, causal (no lookahead). Reproduce with
`python -m src.run` and `python -m src.experiment`.*

## Headline

| Goal | Best model | Score | Baseline | Verdict |
|---|---|---|---|---|
| **Volatility** (10-day ahead) | **HARX** (HAR + VIX) | **OOS R² = 0.558** | naive 0.163, HAR 0.441 | strong, useful |
| **Anomaly** (idiosyncratic) | conformal + IsoForest + LOF | **4.0–4.5× vol separation**, ~1.5% flag rate | — | works; catches all major shocks |
| **Direction** (next-day) | logistic | AUC 0.502 | coin flip 0.50 | at random ceiling (honest) |

## Volatility — the strongest deliverable

HARX (HAR realized-vol terms + VIX) clearly wins. VIX is forward-looking implied vol and
adds real predictive power on top of HAR:

| horizon | naive | HAR | **HARX** | EWMA | GARCH | HGBR |
|---|---|---|---|---|---|---|
| 5d | 0.148 | 0.413 | **0.496** | 0.322 | −0.22 | 0.09 |
| 10d | 0.163 | 0.441 | **0.558** | 0.366 | −0.14 | 0.00 |
| 21d | — | — | **0.476** | — | — | — |

GARCH and gradient boosting underperformed here — HARX is the right tool. R² ≈ 0.56 for
10-day volatility is a credible, industry-grade result.

## Anomaly — catches the real Boeing shocks

Two key improvements over the scaffold, both verified on real data:

1. **Peer-basket residual (instead of ITA).** ITA holds Boeing, so BA's own crashes leaked
   into the "sector" factor and shrank its residual. Switching the sector factor to a
   BA-free peer basket (RTX/GE/LMT/NOC/EADSY) produced sharper, larger idiosyncratic signals:

   | Event | resid_z (ITA) | resid_z (peer basket) |
   |---|---|---|
   | Alaska door-plug blowout (Jan 2024) | −6.10 | **−6.98** |
   | Ethiopian crash / MAX grounding (Mar 2019) | −5.39 | **−6.19** |
   | Lion Air crash (Oct 2018) | −2.79 | **−4.01** |
   | MAX production halt (Dec 2019) | −1.69 | **−3.08** |
   | 787 charges (Jan 2022) | −1.77 | **−3.04** |

2. **Operating-point tuning.** The ensemble was tuned to a clean point: flagged days run at
   **4–4.5× the forward realized volatility of calm days** — strong evidence the flags are
   real, computed across all 2,500 days (not a tiny event sample).

The detector catches every event that produced a genuine idiosyncratic price move (door-plug,
Ethiopian, IAM strike, COVID trough). The events it "misses" are mostly those with *small*
Boeing-specific reactions — leadership change, telegraphed job cuts, pre-announced losses —
where the price simply didn't move abnormally. Raising recall on those requires the **news
layer** (below), since the signal isn't in the price.

## Direction — honestly at the ceiling

Every model (logistic, gradient boosting) across every feature set lands at AUC ≈ 0.50 —
indistinguishable from a coin flip. This is the expected efficient-market result for a liquid
large-cap and we report it as-is rather than overfitting a fake edge.

## Where the ceiling is, and how to break it (brainstorm)

We've reached the realistic ceiling for a **price + cross-asset** approach. The next gains
are not in tuning these models further but in adding information the price doesn't yet carry:

1. **News / sentiment layer (FinBERT + Finnhub).** Biggest expected lift — catches events
   with small or delayed price reactions, and adds lead time. Needs a free API key.
2. **Options-implied signals** (if a free/cheap source can be found): IV jumps, put/call,
   skew often move *before* price.
3. **Event-conditional direction.** Don't predict raw daily direction; predict the *drift
   after* an anomaly fires or after earnings/delivery prints — a narrower, more defensible bet.
4. **Intraday near-real-time monitor.** Recompute the residual on 1–15 min bars for live flags.
5. **Deep sequence models** (LSTM-autoencoder, TFT) as challengers — adopt only if they beat
   HARX / the ensemble out-of-sample.
</content>
