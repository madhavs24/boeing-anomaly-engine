# Flaw Fixes — What Changed, Why, and the Effect on the Numbers

*Companion to FLAWS.md. Every fix was verified end-to-end on the real cached panel. The honest
theme: fixing flaws made the headline numbers **lower but trustworthy** — exactly the point.*

## Verified effect on results (real BA data, before → after fixes)

| Metric | Before (flawed) | After (fixed) | Why it changed |
|---|---|---|---|
| Volatility HARX R² (10d) | 0.558 | **0.34** (R²-vs-mean) / **0.23 skill-vs-naive** | embargo + aligned common sample + honest benchmark |
| Vol "beats naive" | "yes" | **not significant at 95% CI** | block-bootstrap CI on error reduction |
| Anomaly separation | "4.5×" | **2.88× idiosyncratic** (4.1× total, labeled mechanical) | de-circularized metric (residual, not total vol) |
| Anomaly recall significance | not tested | **p = 0.0055 vs random** | permutation test added |
| Direction | "~0.50, no edge" | **0.49–0.51, p≈0.3–0.9, −5 to −24 bps after costs** | significance test + transaction costs |

## Fixes applied (code, verified)

| Flaw | Fix | File |
|---|---|---|
| 🔴 A1 unadjusted prices | use Yahoo **adjusted close** for returns; keep raw for display | data.py `_yahoo` |
| 🔴 A2 peer structural breaks | **median** peer basket (robust to one broken series) + winsorize daily returns at ±25% | features.py |
| 🟠 A3 silent missing series | fetch fails **loud** on missing critical series; warns on non-critical | data.py `build_live` |
| 🔴 B1 selection bias | **embargo** (purged walk-forward) on all forward-label models + trial-count + deflated-Sharpe helper + explicit caveat in leaderboard | validation.py, volatility.py, experiment.py |
| 🟠 B2 different test samples | all vol models scored on the **same common index** | volatility.py `evaluate` |
| 🟠 B3 R² vs global mean | added **skill-vs-naive** as the primary metric | volatility.py |
| 🟡 B5 no costs | direction now reports **net P&L after 10 bps round-trip** + tradeable flag | direction_pro.py |
| 🟡 B6 ordinal day-of-week | **one-hot** dow_0..dow_4 | direction_pro.py |
| 🟠 C1 no significance | **permutation test** (anomaly recall), **binomial test** (direction), **block-bootstrap CIs** (vol) | validation.py + callers |
| 🟠 C2 conformal exchangeability | documented caveat + pointer to adaptive conformal inference | anomaly.py docstring |
| 🟡 C3 noisy coinflip baseline | replaced with **analytic** binomial test vs 0.5 and vs base rate | direction_pro.py |
| 🔴 D1 circular anomaly metric | primary metric now **idiosyncratic forward |residual|** separation; total-vol kept but labeled "partly mechanical" | anomaly.py `evaluate` |
| 🟠 D2 correlated detectors | Isolation Forest / LOF now use **non-residual** features only (disjoint from conformal) | anomaly.py `IF_FEATS` |
| 🟡 D4 yearly refit | refit **twice a year** (126d) | anomaly.py |
| 🟡 E4 magnitude/costs | direction P&L is magnitude- and cost-aware | direction_pro.py |
| 🟠 F1 silent degradation | critical-series check aborts a bad build | data.py |
| 🟡 F2 synthetic clobber | synthetic saved to a **separate file**; never overwrites real cache; new `cached` mode | data.py `get_panel` |

## Still open — needs a re-fetch (data) or is an accepted limitation

**Needs one re-fetch with the improved loader** — ✅ DONE (Desktop copy):
- Panel now has adjusted prices + `oil`/`dollar`/`hy_oas` (FRED with Yahoo/HYG-LQD fallbacks).
- Re-run baseline: `python -m src.run cached` from `C:\Users\singhm\Desktop\Econ project\boeing-anomaly-engine`.

**Accepted limitations (documented, not "bugs"):**
- ✅ E1 Boeing operational data added (`src/operational.py`, SEC EDGAR + O&D CSV + FAA events)
- ✅ E2 BA options IV added (`src/options.py`); free yfinance history is shallow (~1 snapshot/day)
- 🟡 A4 short history (~10y), A5 EADSY ADR noise (mitigated by median basket), A6 ffill, A7 partial
  last bar — data-availability limits, partially mitigated.
- ✅ D3 Isolation-Forest score-threshold detector implemented
- 🟡 E3 GARCH still underperforms (it's a poor fit for BA here) — now counted/handled cleanly.
- ✅ F3 pytest suite in `tests/`; ✅ F4 moved off OneDrive to local Desktop; 🟡 G1 end-of-day;
  ✅ G2 regime conditioning via `src/regime.py`; G3 soft anomaly ground truth; G4 single-name.

## The honest takeaway

The fixes did not raise accuracy — they **revealed the true, lower accuracy** and attached
significance tests and costs to it. Volatility (HARX) is the one component with a real, if
modest and not-yet-95%-significant, edge. Anomaly detection genuinely beats random (p=0.006)
on a de-circularized metric. Direction remains unprofitable after costs. That is the correct,
trustworthy state of the project.
</content>
