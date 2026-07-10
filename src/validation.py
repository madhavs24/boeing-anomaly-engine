"""Validation & statistics utilities — the rigor layer (fixes B1, B3, C1, C3, and supports
multiple-testing awareness from B1).

Provides:
  - embargoed_blocks()      walk-forward blocks with an EMBARGO gap so a label that looks h
                            days ahead can't leak into the test block (purged/embargoed CV,
                            Lopez de Prado).
  - moving_block_bootstrap_ci()  block bootstrap CIs that respect autocorrelation.
  - binom_p()               exact one-sided binomial test for accuracy vs a base rate.
  - permutation_recall_p()  null distribution for anomaly event-recall (random flags).
  - probabilistic_sharpe(), deflated_sharpe()  selection-bias-aware skill tests.

References: Lopez de Prado (purged CV); Bailey & Lopez de Prado (deflated Sharpe);
Gibbs & Candes (adaptive conformal, used as a caveat in anomaly.py).
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def embargoed_blocks(n, min_train=750, test_size=63, embargo=0):
    """Yield (train_end, test_slice). Train on rows [0:train_end-embargo], test [i:i+test_size].
    `embargo` should be >= the label horizon h so forward-looking labels don't leak."""
    i = min_train
    while i < n:
        train_end = max(1, i - embargo)
        yield train_end, slice(i, min(i + test_size, n))
        i += test_size


def moving_block_bootstrap_ci(values, stat=np.mean, n_boot=1000, block=21, alpha=0.05, seed=0):
    """CI for a statistic of a (time-ordered) array using the moving-block bootstrap."""
    x = np.asarray(values, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < block * 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, n - block + 1)
    out = []
    for _ in range(n_boot):
        starts = rng.choice(starts_pool, size=nblocks, replace=True)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        out.append(stat(sample))
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def binom_p(successes, n, p0=0.5, alternative="greater"):
    """Exact binomial test that accuracy > p0 (e.g., beats a coin flip / base rate)."""
    if n == 0:
        return float("nan")
    return float(stats.binomtest(int(successes), int(n), p0, alternative=alternative).pvalue)


def permutation_recall_p(flag, event_idx, tol_post=3, pre=1, n_perm=2000, seed=0):
    """p-value that the detector's event recall exceeds what RANDOM flags at the same rate
    would achieve. Preserves the number of flags; shuffles their positions."""
    flag = np.asarray(flag, bool); ev = np.asarray(event_idx, bool)
    n = len(flag); k = int(flag.sum()); rng = np.random.default_rng(seed)
    ev_positions = np.where(ev)[0]
    if k == 0 or len(ev_positions) == 0:
        return float("nan"), float("nan")

    def recall(fl):
        caught = 0
        fpos = np.where(fl)[0]
        for e in ev_positions:
            if np.any((fpos >= e - pre) & (fpos <= e + tol_post)):
                caught += 1
        return caught / len(ev_positions)

    obs = recall(flag)
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = np.zeros(n, bool); perm[rng.choice(n, size=k, replace=False)] = True
        null[b] = recall(perm)
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return round(obs, 3), p


def probabilistic_sharpe(sharpe, n, skew=0.0, kurt=3.0, sr_benchmark=0.0):
    """PSR: probability the true Sharpe exceeds sr_benchmark given sample noise (annualized
    inputs not required if sharpe & benchmark are per-period and n is the number of periods)."""
    if n < 2:
        return float("nan")
    denom = np.sqrt(1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2)
    z = (sharpe - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_threshold(n_trials, sr_std_across_trials, n_obs, euler=0.5772):
    """Expected MAX Sharpe under ZERO skill across `n_trials` (the bar a backtest must clear).
    Bailey & Lopez de Prado. Compare your best trial's Sharpe to this threshold."""
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sr_std_across_trials * ((1 - euler) * z1 + euler * z2))
