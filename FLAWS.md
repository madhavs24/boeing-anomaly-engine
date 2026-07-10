# Boeing Anomaly Engine — Complete Flaw Audit

*An honest, ruthless inventory of every problem in the project, from tiny to foundational.
Grounded in the actual code. Severity: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low.
This is a self-audit — finding these is what makes the project trustworthy.*

---

## A. Data integrity (the most damaging, because everything downstream inherits it)

🔴 **A1. Unadjusted prices.** `data._yahoo` reads `quote["close"]` — the *raw* close, not
adjusted close. Dividends and (especially) splits create fake jumps in returns:
- **GE** did a 1-for-8 reverse split (2021) and spun off GE Aerospace (2024) → huge artificial
  return spikes in a peer used to build the residual factor.
- **RTX** was United Technologies until the 2020 merger → ticker/price discontinuity.
- ETFs/peers pay dividends → small fake negative returns on ex-dates.
These corrupt the peer-basket residual (the spine of the anomaly detector) around those dates.
**Fix:** use `adjclose` (`indicators.adjclose`) for returns; keep raw close only for display.

🔴 **A2. Structural breaks / survivorship in the peer set.** The peer basket (RTX, GE, LMT,
NOC, EADSY) is chosen as of *today* and assumed stable over 2016–2026. GE's spinoff and RTX's
merger mean the "peer" series aren't the same company through time. The factor model silently
regresses BA on discontinuous series.

🟠 **A3. Missing macro series, silently.** Your fetch lost three FRED series (WTI spot, broad
USD, HY credit spread) to timeouts. The code degrades silently: `dollar_ret`, `hy_chg` are
simply never created, and `oil_ret` falls back to **USO**, an ETF with heavy contango/roll
decay that tracks oil poorly. So "oil exposure" is partly an artifact of USO's decay.

🟠 **A4. Short history.** ~10 years (2016–2026, 2,514 rows). No 2008 GFC, limited regime
variety. Walk-forward burns the first ~750 rows on training, leaving fewer test years than it
looks. Small effective sample for everything.

🟡 **A5. EADSY (Airbus ADR)** trades on US hours as an ADR with currency effects, thin volume,
and stale prints vs the primary Paris listing → adds noise and timing mismatch to the basket.

🟡 **A6. Calendar misalignment + forward-fill.** Series with different holiday calendars
(ADR vs US) are outer-joined; `ffill(limit=5)` can carry a stale price up to 5 days, dampening
or fabricating returns around gaps.

⚪ **A7. Possibly-partial last bar.** If fetched intraday, the final row is an incomplete day,
so "today's" return/feature can be wrong until the close.

---

## B. Methodology & validation

🔴 **B1. Selection bias from the experiment sweep.** `experiment.py` tries many configs and
reports the *best* (by R², AUC, or composite score). With no final held-out period or nested
cross-validation, the "winner" is optimistically biased — you tuned and reported on the same
data. The true out-of-sample number is lower than the leaderboard's best.

🟠 **B2. Volatility models scored on different samples.** `naive/har/harx/ewma/garch/hgbr`
each do their own `dropna()`, so their R²s are computed on *slightly different test sets*. The
leaderboard comparison is therefore not strictly apples-to-apples.

🟠 **B3. OOS R² uses the full-period mean.** `_oos_r2` benchmarks against `np.nanmean(actual)`
over the whole out-of-sample set — a quantity not knowable in real time. This mildly inflates
R². The honest benchmark is the naive forecast (which we do report separately, so cross-check
against *that*, not the R²-vs-mean).

🟡 **B4. Generous event-recall tolerance.** `event_windows(pre=1, post=3)` counts a flag up to
3 days *after* an event as a "catch," which flatters recall and lead-time.

🟡 **B5. No transaction costs / slippage anywhere.** Direction accuracy and any implied trading
value ignore costs; daily signals frequently don't survive them.

🟡 **B6. `dow` ordinal-encoded for logistic.** Day-of-week is fed as an integer 0–4 to the
logistic model, imposing a false ordering (Mon<...<Fri). Fine for trees, wrong for linear.

---

## C. Statistical rigor

🟠 **C1. No confidence intervals or significance tests anywhere.** Event recall (n=17, ~6
"big"), the event-conditional edge (n=80), R², AUC — all reported as point estimates. The
event-conditional "0.41 vs 0.53" could plausibly be noise; we never test it.

🟠 **C2. Conformal calibration assumes exchangeability.** The "calibrated 1% false-alarm rate"
relies on past scores being exchangeable with today's. Financial data has volatility clustering
and regime shifts → the guarantee doesn't strictly hold, and realized flag rates drift from the
nominal alpha (we observed this).

🟡 **C3. Coin-flip baseline is a single random draw.** `direction._score` compares to one
`rng.integers` sample instead of the analytic 0.5, adding noise to the "beats baseline" test.

⚪ **C4. Tiny curated event list, hand-picked by me** — selection bias in *which* events count,
plus date-precision errors (I guessed the first reaction day).

---

## D. Anomaly detector specifics

🔴 **D1. The vol-separation metric is partly circular.** `ANOMALY_FEATS` includes `rv5, rv21,
gap` (contemporaneous volatility), and the validation metric is "flagged days have higher
*forward* volatility." Because volatility clusters, flagging high-vol days almost guarantees
high forward vol — so the headline "4–4.5× separation" overstates genuine skill. It's not
meaningless (the residual term adds real info), but it is inflated by construction.

🟠 **D2. Detectors aren't independent, so "votes" overstate confidence.** The conformal
detector keys on `resid_z`; Isolation Forest and LOF *also* receive `resid_z`/`resid_abs` in
their feature vector. When they agree, it's often the same signal counted twice — a 2/3 "vote"
is weaker evidence than it appears.

🟠 **D3. Isolation Forest / LOF flag rate is mechanical.** `contamination` forces the model to
treat ~that fraction as anomalous regardless of whether real anomalies exist. So part of the
flag rate is a fixed quota, not detection.

🟡 **D4. Infrequent refit (252 days).** IF/LOF retrain only yearly; within a year the model is
stale to regime changes.

🟡 **D5. `gap` is mislabeled.** It's `|close-to-close return|`, not a true overnight gap (that
needs the open). It's also largely redundant with `rv`/`ret`.

---

## E. Modeling choices

🟠 **E1. None of the actual Boeing drivers are in the model.** The research report says
deliveries, free cash flow, FAA caps, certifications, the order book and defense backlog are
what move BA — yet the model uses only price/cross-asset/macro. The biggest information sources
identified in research are absent in implementation. This is the largest conceptual gap.

🟠 **E2. HARX uses market VIX, not BA-implied vol.** VIX is S&P 500 implied vol; we have no
Boeing options IV. HARX works via market comovement, but it isn't a BA-specific forward signal.

🟡 **E3. GARCH fits fail silently.** Convergence warnings are suppressed and failed fits leave
NaNs; the GARCH R² is computed on whatever survived, on yet another sample (see B2).

🟡 **E4. Direction target ignores magnitude/costs.** `ret.shift(-1) > 0` treats a +0.01% day
and a +5% day identically, and counts moves too small to trade.

⚪ **E5. News layer is unbuilt/untested.** `news.py` has only a crude lexicon fallback and has
never run against the real API (no key). Its sentiment quality is unknown.

---

## F. Implementation / engineering

🟠 **F1. Silent degradation everywhere.** Failed data fetches, missing columns, and failed
model fits are logged or swallowed but never *block* a run, so the engine happily reports
numbers computed on a quietly-incomplete dataset (this already happened with the FRED series).

🟡 **F2. `auto` mode can overwrite real data with synthetic.** `get_panel("auto")` tries live
first and, if the network fails, silently builds a *synthetic* panel and saves it over
`panel.parquet`. A failed refresh could clobber your real cached data.

🟡 **F3. No tests.** There is no unit/regression test suite; correctness rests on manual runs.

✅ **F4. OneDrive sync hazard — resolved.** Project moved to local Desktop path
`C:\Users\singhm\Desktop\Econ project\boeing-anomaly-engine` (no cloud sync during runs).

⚪ **F5. Reproducibility gaps.** Results depend on refit cadence, library versions, and the
fetch timing/window; no environment lockfile or seed manifest is captured per run.

---

## G. Scope & framing

🟠 **G1. "Real-time" is aspirational.** Everything runs on daily, ~15-min-delayed free data.
The intraday monitor is scaffolded but not built or validated. The product is end-of-day.

🟡 **G2. No regime/structural-break handling.** One model is assumed to hold across COVID, the
MAX crisis, rate hikes, etc. No explicit regime conditioning.

🟡 **G3. Anomaly detection has no ground truth.** We validate against a proxy (curated events +
vol separation), so "performance" is inherently soft and partly defined by our own choices.

⚪ **G4. Single name, no cross-validation across stocks.** Methods tuned on BA alone; we don't
know if the residual/HARX choices generalize or are BA-overfit.

---

## Severity tally

- 🔴 Critical: A1 (unadjusted prices), A2 (peer structural breaks), B1 (selection bias),
  D1 (circular anomaly metric).
- 🟠 High: A3, A4, B2, B3, C1, C2, D2, D3, E1, E2, F1.
- The single most important fixes, in order: **A1 → A2 → E1 → B1 → D1**.

## What this does NOT have (verified clean)

- ✅ The idiosyncratic residual is genuinely **causal** (fits on the window ending at *t−1*,
  applies to *t*); no lookahead in `rolling_residual`.
- ✅ Direction/volatility use **walk-forward** with train-only scaler fits; no shuffled splits.
- ✅ Forward labels are correctly shifted; signals act on prior-day info.
- ✅ Honest baselines (naive, coin flip, majority) are reported alongside every model.

The leakage we *demonstrated* (`leakage_demo.py`) is deliberately injected for teaching, not
present in the real pipeline.
</content>
