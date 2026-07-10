"""Curated real Boeing-specific event dates, used as ground truth to VALIDATE the
unsupervised anomaly detector (recall / lead time). Anomaly detection has no natural
accuracy label, so we check whether the detector fires around days when something
genuinely material happened to Boeing.

Dates are the first U.S. trading day on/after the event (when the stock reacted).
Sources: public reporting on each event. Extend this list as needed.
"""
from __future__ import annotations
import pandas as pd

# (date, short description)
BOEING_EVENTS = [
    ("2018-10-29", "Lion Air 610 crash (first MAX crash)"),
    ("2019-03-11", "Ethiopian 302 crash -> MAX grounding days later"),
    ("2019-03-13", "FAA grounds 737 MAX"),
    ("2019-04-05", "MAX production cut to 42/mo"),
    ("2019-12-16", "MAX production halt announced"),
    ("2020-02-24", "COVID crash begins"),
    ("2020-03-16", "COVID crash trough region"),
    ("2020-11-18", "FAA ungrounds 737 MAX"),
    ("2021-07-14", "787 deliveries halt / production issues"),
    ("2022-01-26", "Q4 results, 787 charges"),
    ("2024-01-08", "Alaska 1282 door-plug blowout (Jan 5 Fri)"),
    ("2024-01-24", "FAA caps 737 MAX production at 38/mo"),
    ("2024-03-25", "CEO Calhoun to step down / leadership shakeup"),
    ("2024-09-13", "IAM machinists strike begins"),
    ("2024-10-11", "Boeing to cut 17,000 jobs / capital raise"),
    ("2024-11-04", "Machinists approve deal, strike ends"),
    ("2025-01-28", "FY2024 results, large losses"),
]


def event_index(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean Series over `index`: True on the first trading day on/after each event."""
    flags = pd.Series(False, index=index)
    for d, _ in BOEING_EVENTS:
        ts = pd.Timestamp(d)
        future = index[index >= ts]
        if len(future):
            flags.loc[future[0]] = True
    return flags


def event_windows(index: pd.DatetimeIndex, pre: int = 0, post: int = 3) -> pd.Series:
    """Boolean Series: True within [-pre, +post] trading days of any event.
    Used for recall with tolerance (a flag near an event counts as a catch)."""
    base = event_index(index)
    win = base.copy()
    pos = list(index)
    for i, hit in enumerate(base.values):
        if hit:
            lo, hi = max(0, i - pre), min(len(pos), i + post + 1)
            win.iloc[lo:hi] = True
    return win
