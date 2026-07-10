# Boeing (BA) Anomaly Detection & Prediction — Deep Research Report

*Prepared June 24, 2026. This is the research foundation for a single-name, near-real-time
anomaly-detection and forecasting engine focused exclusively on The Boeing Company (NYSE: BA).
It documents what actually moves the stock, which features to build, what data is obtainable
for free, which models work and which only look like they work, and the time frame we can
honestly operate at. A runnable scaffold built from these conclusions ships alongside it.*

> **Not investment advice.** Nothing here trades real money. The goal is an honest,
> industry-grade research and monitoring tool, not a money machine.

---

## 0. Executive summary (read this first)

Three honest conclusions shape the entire design:

1. **The achievable target is anomaly/event detection, then volatility, then — distantly —
   direction.** This mirrors what the prior market-wide project proved and what the 2024–2025
   literature confirms: for a liquid large-cap, *direction* prediction rarely beats a coin
   flip after costs, but *abnormality* and *volatility* are genuinely forecastable. We pursue
   all three (as you asked) but rank effort and trust accordingly.

2. **"Real time" on free data means minute-bars, not microseconds.** True sub-second / tick
   prediction needs co-located servers and paid exchange feeds costing five to six figures a
   year. With free sources the honest operating frequency is **daily for modeling** and
   **1-to-15-minute bars for a near-real-time monitoring layer** (delayed ~15 min). We design
   for exactly that and don't pretend otherwise.

3. **The single most powerful idea for Boeing specifically is residual (idiosyncratic)
   decomposition.** Boeing moves for two reasons: the whole market/sector moves (oil, rates,
   VIX, the aerospace-defense ETF ITA), or *something happened to Boeing* (a grounding, an FAA
   action, a crash, a strike, an order, an earnings surprise). If we statistically strip out
   the market/sector/peer component, what's left — the **idiosyncratic residual** — is the
   purest signal of a Boeing-specific event. Anomalies in that residual are what you actually
   want to catch. This is the spine of the system.

---

## 1. What actually moves Boeing stock

Boeing is unusual: it is simultaneously a cyclical commercial-aircraft manufacturer, a
defense contractor with a record backlog, and a serial headline-risk company. Its drivers
split into three layers.

### 1.1 Idiosyncratic (Boeing-specific) — the highest-information, hardest-to-get layer

These are the events that cause the big, sudden, *stock-specific* moves — the ones a
general market model never sees coming:

- **Production rate & FAA caps.** The FAA capped 737 MAX output at 38/month after the Jan-2024
  Alaska Airlines door-plug blowout, later lifting it to 42 and then 47/month, with a new
  Everett line targeting 52/month by early 2027. Every cap change is a material catalyst.
- **Monthly deliveries & orders.** Boeing reports deliveries monthly; deliveries (not orders)
  drive cash. The stock reacts sharply to the monthly delivery print and to large order
  announcements (e.g., the discussed China purchase of 500+ aircraft).
- **Free cash flow trajectory.** This is the central valuation story. Management guided 2026
  FCF to **+$1B to +$3B**, a swing from a **$1.9B cash burn in 2025**. A miss raises questions
  about servicing **~$54B of debt** — a miss/beat here is among the largest single catalysts.
- **Certification milestones.** 777-9 and 737-7/737-10 certifications are pending, binary,
  high-impact events.
- **Safety events & regulatory actions.** Crashes, groundings, whistleblower reports, NTSB/FAA
  findings — these produce the largest idiosyncratic gaps (the MAX groundings, the door plug).
- **Labor relations.** The 2024 IAM machinists' strike halted production; strikes and their
  resolutions are major.
- **Defense segment.** Defense, Space & Security backlog is at a record (~$86B), revenue +37%
  YoY in a recent quarter; defense contract wins/losses and program charges (fixed-price
  losses on KC-46, T-7, etc.) move the stock and partially offset commercial cyclicality.
- **Supply chain.** Spirit AeroSystems reintegration, engine and fuselage supply constraints.

**Implication:** the highest-value features are *event features* — and most live in news,
filings (8-K), and monthly operational data, not in the price feed. Free price data alone
will miss the *cause*; it only sees the *effect* (the residual jump). So we build both a
price-based residual detector (always works, free) and a news/event layer (better, needs a
free API key).

### 1.2 Sector & peer layer — what to "subtract out"

Boeing co-moves with the aerospace-defense complex. We use these to compute beta and the
residual:

- **ITA** (iShares U.S. Aerospace & Defense ETF) — the cleanest sector proxy. Returned ~+49%
  NAV in 2025. Boeing is a top holding, so use with care (it partly contains BA itself).
- **Peers:** **RTX**, **GE Aerospace (GE)**, **Airbus (EADSY/AIR.PA)**, **LMT**, **NOC**, **GD**.
  Note Boeing–Airbus daily correlation is only ~0.32 (duopoly, but often *inversely* affected
  by each other's stumbles), whereas BA–RTX and BA–GE are higher and better hedges for the
  sector/engine cycle.
- **Broad market:** **SPY** (market beta) and the airlines (JETS ETF) as a demand proxy.

### 1.3 Macro layer

- **Oil / jet fuel** (WTI via USO or FRED `DCOILWTICO`). Jet fuel tracks crude at ~0.98
  correlation; high fuel pressures airline profitability and, indirectly, aircraft demand.
- **Interest rates** (10Y, `^TNX`/`DGS10`) — Boeing is capital-intensive and heavily indebted.
- **US dollar** (DXY) — Boeing is a major exporter; a strong dollar hurts competitiveness vs
  Airbus.
- **VIX** (`^VIX`) — overall risk appetite; high-beta names like BA amplify market stress.
- **Credit spreads** (HY OAS, `BAMLH0A0HYM2`) — BA's leverage makes it credit-sensitive.

---

## 2. Feature engineering — the inputs the models actually learn from

Raw prices are noise. The design computes a compact, causal feature vector per timestamp.
Everything is computed using **only past information** (no peeking), the cardinal rule.

### 2.1 The core idea: idiosyncratic residual

Estimate, on a rolling window, a simple factor model:

```
r_BA,t  =  alpha  +  beta_mkt * r_SPY,t  +  beta_sec * r_ITA_resid,t  +  e_t
```

where `r_ITA_resid` is ITA's return orthogonalized to SPY (so we don't double-count market),
and **`e_t` is the idiosyncratic residual** — the part of Boeing's move *not* explained by the
market or the sector. A large `|e_t|` (in standard-deviation units) is, by construction, a
Boeing-specific event. The **standardized residual** `z_t = e_t / rolling_std(e)` is the
single most important feature in the whole system and the primary anomaly score.

### 2.2 Full feature list

| Group | Features | Why it matters for Boeing |
|---|---|---|
| **Idiosyncratic** | standardized residual `z_t`, |z_t|, residual realized vol (5/21d), rolling residual skew | Isolates Boeing-specific shocks from market/sector |
| **Price/return** | log returns, overnight gap, intraday range (High−Low)/Close, realized vol (5/21/63d), return skew/kurtosis | Boeing gaps hard on overnight news; range captures stress |
| **Volume/liquidity** | volume z-score, Amihud illiquidity (|ret|/$vol), turnover | Event days show volume spikes; illiquidity precedes vol |
| **Relative strength** | BA−ITA, BA−RTX, BA−GE, BA−EADSY spreads; rolling beta to ITA | Underperformance vs peers = company-specific trouble |
| **Macro** | WTI/USO return, 10Y level & change, DXY return, VIX level & change, HY OAS change | Cost, financing, FX, risk-appetite channels |
| **Term-structure / regime** | VIX term slope (VIX3M−VIX), drawdown from 252d high, distance from 50/200d MA | Context: is the whole market also stressed? |
| **Calendar/event** | days-to-earnings, days-since-earnings, monthly-delivery-report flag, options-expiry flag | Boeing moves predictably *around* known events |
| **News/sentiment (optional, API key)** | FinBERT sentiment score, news volume z-score, negative-headline count | Captures the *cause* of idiosyncratic jumps in near-real-time |

The news layer is the single biggest accuracy upgrade and the one piece that needs a free
API key (Finnhub free tier gives news + a sentiment score, 60 calls/min). The scaffold ships
with a working price/macro pipeline and a clean, optional plug-in point for the news layer.

---

## 3. Data sources — what's actually obtainable for free

You confirmed **free sources only**. Here is the realistic menu and what each gives you.

| Source | What you get free | Latency / limit | Role in this project |
|---|---|---|---|
| **Yahoo chart API** (direct, browser UA — *not* yfinance, which throttles/breaks) | BA + peers + ETFs OHLCV; **daily** (10y+) and **1-min/5-min/15-min** (~last 30–60 days) | ~15-min delayed; generous | **Primary** price feed (proven in your prior project) |
| **FRED** (CSV, no key) | WTI oil, 10Y/2Y yields, dollar index, HY OAS, VIX | Daily, end-of-day | Macro layer |
| **Finnhub** (free key) | News + per-article sentiment, basic fundamentals, 20-min delayed quotes | 60 calls/min | **Optional** news/sentiment layer (big upgrade) |
| **Alpha Vantage** (free key) | News sentiment, daily bars | 25 calls/**day** | Backup news only (too rate-limited for prices) |
| **Boeing IR / SEC EDGAR** | 8-K filings, monthly deliveries, earnings | On release | Event calendar & filing-trigger layer |
| **Options / implied vol** | — | Effectively **not free** at quality | Documented as a paid upgrade; we proxy with realized vol |

**Honest limitation:** intraday minute bars on free feeds only go back ~30–60 days and are
delayed. That is enough for a *near-real-time monitor* and for studying recent events, but
**not** enough to train deep intraday models on years of history. So we train on **daily**
history (10+ years) and run the **intraday monitor** on the rolling recent window.

---

## 4. Time frame — the honest decision

You asked which frequency to operate at (nanoseconds … days). The answer is forced by data
and by signal-to-noise, not by preference.

| Horizon | Feasible on free data? | Signal-to-noise | Verdict |
|---|---|---|---|
| Nanosecond / microsecond (HFT) | **No** — needs colocation + paid tick feeds | n/a for us | **Out of scope.** Don't pretend. |
| Seconds | No (no free tick data) | Pure microstructure noise | Out of scope |
| **1–15 minutes** | Partially (delayed, ~30–60d history) | Low but usable for *monitoring* | **Near-real-time anomaly monitor** |
| **Daily** | **Yes** (10y+ history) | **Best** for ML/DL | **Primary modeling horizon** |
| Weekly/monthly | Yes | Good but fewer samples | Context / regime layer |

**Decision:** two-speed design.
- **Modeling & prediction → daily.** This is where the literature finds real, if modest,
  edges, where we have enough history to validate honestly, and where overfitting is
  controllable. Volatility and anomaly models train here.
- **Live anomaly monitoring → 1–15 min bars.** A lightweight layer that recomputes the
  idiosyncratic residual and a few liquidity features on the most recent delayed bars and
  raises a flag when Boeing is moving abnormally *right now*. This is the "real-time" feel,
  done honestly (it's monitoring/detection, not minute-ahead price prediction).

---

## 5. Model survey — what works, what only looks like it works

Organized by the three goals, ranked by how much we trust them for Boeing.

### 5.1 Goal A — Anomaly / event detection *(highest trust)*

The literature (2024–2025) converges on **hybrids**: tree-based + reconstruction-based +
statistical, combined. Our design uses three detectors that vote:

1. **Conformal residual alarm (statistical, primary).** Convert the standardized
   idiosyncratic residual into a **conformal p-value** with a *calibrated* false-alarm rate
   you set (e.g., 1%). Formula:
   `p_t = (1 + #{past scores ≥ today's}) / (calibration_size + 1)`; flag if `p_t ≤ α`.
   This is the same rigorously-calibrated method that led real crises in your prior project —
   here pointed at Boeing's *idiosyncratic* score. Transparent, leakage-free, tunable.
2. **Isolation Forest (multivariate, unsupervised).** Operates on the full feature vector to
   catch *multi-dimensional* anomalies (e.g., normal price move but abnormal volume + peer
   divergence + IV). Cheap, robust, the standard baseline in recent papers.
3. **LSTM autoencoder (sequence reconstruction).** Learns Boeing's "normal" multivariate
   dynamics; a high **reconstruction error** flags regime breaks. Recent hybrid frameworks
   (Isolation Forest + LSTM-AE) report >90% detection accuracy on time series. This is the
   "deep learning" component, used for *detection* (its honest strength) not price prediction.

**Ensemble:** a day is flagged at a severity level by how many detectors agree. This is the
core deliverable and the most defensible.

### 5.2 Goal B — Volatility forecasting *(medium-high trust)*

Predicting how *turbulent* BA will be over the next N days is genuinely doable.

- **Baseline: HAR-RV** (Heterogeneous AutoRegressive). A weighted blend of daily/weekly/
  monthly realized vol. Famously hard to beat; out-of-sample R² typically 0.3–0.5. **This is
  the benchmark every fancier model must beat.**
- **GARCH(1,1)** as a classical challenger (volatility clustering).
- **LSTM / TCN / Temporal Fusion Transformer (TFT)** as the deep challengers. TFT is the
  strongest in recent benchmarks (40–50% MAE reduction over LSTM on some equity sets) **and**
  gives interpretable feature-importance — but evidence is *mixed*: several studies find it
  does **not** reliably beat a well-tuned LSTM or even HAR. So: include it, but only adopt it
  if it beats HAR out-of-sample on Boeing specifically.

### 5.3 Goal C — Direction prediction *(low trust — included honestly)*

You asked for it; here is the honest treatment. For a liquid large-cap, next-day direction is
near-random and notoriously overfit in papers (which often leak future data). Plan:

- **Baselines first:** logistic regression and gradient boosting (XGBoost/LightGBM) on the
  feature vector, evaluated with **walk-forward** validation and compared to the only honest
  benchmarks: a coin flip and "predict the majority/last move."
- **Deep challengers:** CNN-LSTM hybrid and TFT, *only* as challengers to those baselines.
- **Where an edge plausibly exists:** not raw daily direction, but **conditional, event-driven
  direction** — e.g., the sign/size of the post-earnings or post-delivery-report drift, or
  direction *conditional on a fired anomaly*. We test these narrow, defensible hypotheses
  rather than the hopeless "predict tomorrow's close."
- **Sentiment-augmented:** FinBERT news sentiment is the one feature with repeated evidence of
  *small* directional lift for single stocks. It's wired in as the optional upgrade.

**We will report direction results with brutal honesty** — including when (likely) nothing
beats the baseline — because a tool you can trust is worth more than a backtest that lies.

### 5.4 Recommended architecture (hybrid)

```
            ┌────────────── DATA (daily 10y + intraday rolling) ──────────────┐
            │  Yahoo (BA, peers, ETFs) · FRED (macro) · [Finnhub news opt.]   │
            └───────────────────────────┬─────────────────────────────────────┘
                                         ▼
                       FEATURES (causal): residual z, vol, liquidity,
                       relative strength, macro, regime, [sentiment]
                                         ▼
        ┌────────────────┬──────────────────────────┬───────────────────────┐
        ▼                ▼                          ▼                       ▼
  A. ANOMALY        B. VOLATILITY              C. DIRECTION            LIVE MONITOR
  conformal +       HAR (baseline) →           baselines →            (1–15 min):
  IsolationForest + GARCH / LSTM / TFT         XGB / CNN-LSTM /       recompute residual
  LSTM-AE  (vote)   (adopt only if > HAR)      TFT (vs coinflip)      → flag if abnormal
        └──────────────────────────────┬───────────────────────────────────┘
                                        ▼
              OUTPUTS: anomaly timeline · vol forecast band · honest
              direction scorecard · live "is Boeing weird right now?" flag
```

---

## 6. How we avoid fooling ourselves (the part that makes it industry-ready)

The prior project's discipline carries over and is non-negotiable:

- **No lookahead / leakage.** Every feature and label uses only past data; expanding/rolling
  windows; `shift(1)` before any signal acts. Leakage is the #1 way finance ML lies.
- **Walk-forward validation only.** No random train/test splits on time series. Refit on the
  past, test on the untouched future, roll forward.
- **Honest baselines always shown.** HAR for vol; coin flip / last-move for direction;
  VIX/realized-vol for stress. A fancy model that doesn't beat these is reported as a failure.
- **Calibrated false-alarm rate** for anomalies, so "1% alarm rate" actually means 1%.
- **Out-of-sample everything; no tuning until a number "looks good."**
- **Every output ships with its real, modest score and a "not advice" note.**

---

## 7. Build plan (what the scaffold delivers now vs. next)

**Shipped now (runnable today):**
- Project skeleton with config, the proven free-data loader (Yahoo+FRED) plus a synthetic
  fallback so it runs offline, the causal feature pipeline centered on the idiosyncratic
  residual, a working **conformal + Isolation Forest anomaly ensemble**, a **HAR volatility
  baseline**, and an **honest logistic direction baseline vs. coin flip** — all wired into one
  `run.py` that prints a Boeing status report and saves results.

**Next iterations (in priority order):**
1. LSTM-autoencoder detector + ensemble voting.
2. Finnhub news/FinBERT sentiment layer (needs your free API key).
3. Intraday near-real-time monitor (1–15 min loop).
4. Deep challengers (TFT) for vol and direction — adopt only if they beat the baselines.
5. Dashboard/website once the engine is trusted.

---

## 8. Sources

- Boeing FCF / backlog / deliveries / FAA caps: [Seeking Alpha](https://seekingalpha.com/news/4578159-boeing-targets-1b-3b-2026-free-cash-flow-as-737-output-plans-rise-to-47-per-month-this-summer), [IndexBox](https://www.indexbox.io/blog/boeings-682-billion-backlog-signals-turnaround-amid-cash-flow-challenges/), [Boeing IR Q1 2026](https://investors.boeing.com/investors/news/press-release-details/2026/Boeing-Reports-First-Quarter-Results/default.aspx), [Capital.com](https://capital.com/en-int/market-updates/boeing-stock-forecast-13-03-2026), [CoinCentral](https://coincentral.com/boeing-ba-stock-china-deal-faa-approval-and-new-orders-lift-outlook-time-to-buy/), [defence-blog](https://defence-blog.com/boeings-defense-arm-surges-while-company-stays-unprofitable/)
- Peers / sector / fuel correlations: [PortfoliosLab BA vs EADSY](https://portfolioslab.com/tools/stock-comparison/EADSY/BA), [Nasdaq BA vs RTX](https://www.nasdaq.com/articles/ba-vs-rtx-which-aerospace-defense-stock-smarter-option), [iShares ITA](https://www.ishares.com/us/products/239502/ishares-us-aerospace-defense-etf), [IATA Jet Fuel Monitor](https://www.iata.org/en/publications/economics/fuel-monitor/), [Oil shocks & airline returns (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2212012225000048)
- Free data APIs: [Best free stock APIs 2026](https://thenextgennexus.com/2026/05/15/10-best-free-stock-market-apis-2026/), [Finnhub](https://finnhub.io/), [Alpha Vantage docs](https://www.alphavantage.co/documentation/)
- Models — anomaly: [AI anomaly detection / market efficiency (Springer)](https://link.springer.com/article/10.1007/s10614-025-11274-8), [IF + Autoencoder + ConvLSTM hybrid (Springer)](https://link.springer.com/article/10.1007/s10115-025-02580-6)
- Models — forecasting: [Hybrid CNN-LSTM + TFT (IIETA)](https://www.iieta.org/journals/isi/paper/10.18280/isi.301122), [TFT GNN (MDPI)](https://www.mdpi.com/2673-9909/5/4/176), [Transformers vs LSTMs for trading (arXiv)](https://arxiv.org/pdf/2309.11400)
- Intraday feasibility / noise: [Deep learning intraday price (proceedings)](https://proceedings.stis.ac.id/icdsos/article/download/278/90/3334), [Predicting daily direction with DL (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2666827025001276)
- Sentiment: [FinBERT-LSTM (ACM)](https://dl.acm.org/doi/fullHtml/10.1145/3694860.3694870), [FinBERT + SHAP (MDPI)](https://www.mdpi.com/2227-7390/13/17/2747)
</content>
</invoke>
