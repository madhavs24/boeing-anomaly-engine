"""News + sentiment features (Idea #9) — 100% FREE, no API key required.

Sources (tried in order, all free):
  1. Yahoo Finance RSS headlines for BA         (no key)
  2. Google News RSS for "Boeing"               (no key)
  3. Finnhub company-news (only if FINNHUB_API_KEY is set — optional)
Scoring: FinBERT (ProsusAI/finbert via transformers) if installed, else a finance lexicon.

Daily CAUSAL features (yesterday's news predicts today -> shift(1)):
  news_sent        mean headline sentiment [-1,1]
  news_vol_z       z-score of daily headline count (attention spike)
  news_neg_share   fraction of clearly-negative headlines

RSS only returns RECENT headlines, so history is shallow; features are NaN before coverage
starts (handled downstream). For deep history, add an archived-news source later.

    python -m src.news
"""
from __future__ import annotations
import os, re, datetime as dt
import numpy as np, pandas as pd, requests
from xml.etree import ElementTree as ET
from .util import PROC, log

UA = {"User-Agent": "Mozilla/5.0"}
CACHE = PROC / "news.parquet"
CACHE_DAILY = PROC / "news_daily.parquet"
CACHE_MAX_AGE_DAYS = 1
RSS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BA&region=US&lang=en-US",
    "https://news.google.com/rss/search?q=Boeing%20when:400d&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Boeing+737%20when:400d&hl=en-US&gl=US&ceid=US:en",
]
POS = {"beat", "surge", "win", "approval", "record", "profit", "upgrade", "deal", "order",
       "delivery", "rebound", "recovery", "certified", "resume"}
NEG = {"crash", "grounded", "halt", "probe", "strike", "loss", "cut", "delay", "fine", "fall",
       "plunge", "defect", "charge", "lawsuit", "fraud", "recall", "whistleblower", "fault"}
_FINBERT = None


def _rss(url):
    try:
        r = requests.get(url, headers=UA, timeout=15); r.raise_for_status()
        root = ET.fromstring(r.content); rows = []
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            pub = it.findtext("pubDate") or ""
            try:
                ts = pd.to_datetime(pub, utc=True).tz_localize(None)
            except Exception:
                ts = pd.Timestamp.today().normalize()
            if title:
                rows.append((ts, title))
        return rows
    except Exception as e:
        log(f"RSS failed ({url[:40]}...): {e}"); return []


def _finnhub():
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return []
    rows = []; end = dt.date.today()
    for back in range(0, 365, 30):
        to = end - dt.timedelta(days=back); frm = to - dt.timedelta(days=30)
        try:
            r = requests.get("https://finnhub.io/api/v1/company-news",
                             params={"symbol": "BA", "from": frm.isoformat(), "to": to.isoformat(),
                                     "token": key}, timeout=15); r.raise_for_status()
            for a in r.json():
                rows.append((pd.to_datetime(a["datetime"], unit="s"), a.get("headline", "")))
        except Exception:
            pass
    return rows


def _finbert():
    global _FINBERT
    if _FINBERT is None:
        from transformers import pipeline
        _FINBERT = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)
    return _FINBERT


def _score(texts):
    try:
        clf = _finbert()
        out = []
        for t in texts:
            sc = {d["label"].lower(): d["score"] for d in clf(t[:512])[0]}
            out.append(sc.get("positive", 0) - sc.get("negative", 0))
        log("news scored with FinBERT")
        return np.array(out)
    except Exception:
        log("FinBERT not installed — using finance lexicon")
        out = []
        for t in texts:
            w = set(re.findall(r"[a-z]+", t.lower()))
            p, n = len(w & POS), len(w & NEG)
            out.append((p - n) / max(1, p + n))
        return np.array(out)


def _fetch_daily() -> pd.DataFrame | None:
    rows = []
    for u in RSS:
        rows += _rss(u)
    rows += _finnhub()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "headline"]).dropna()
    df["date"] = pd.to_datetime(df["ts"]).dt.normalize()
    df = df.drop_duplicates(subset=["date", "headline"])
    df["sent"] = _score(df["headline"].tolist())
    g = df.groupby("date")
    f = pd.DataFrame({"news_sent": g["sent"].mean(), "news_vol": g.size(),
                      "news_neg_share": g["sent"].apply(lambda s: float((s < -0.1).mean()))})
    f["news_vol_z"] = (f["news_vol"] - f["news_vol"].rolling(20, min_periods=5).mean()) \
        / (f["news_vol"].rolling(20, min_periods=5).std() + 1e-9)
    return f[["news_sent", "news_vol_z", "news_neg_share"]].shift(1)


def _load_daily(max_age_days=CACHE_MAX_AGE_DAYS) -> pd.DataFrame | None:
    if not CACHE_DAILY.exists():
        return None
    age = (pd.Timestamp.today() - pd.Timestamp(CACHE_DAILY.stat().st_mtime, unit="s")).days
    if age > max_age_days:
        return None
    f = pd.read_parquet(CACHE_DAILY)
    f.index = pd.to_datetime(f.index)
    log(f"news loaded from cache ({len(f)} days, age {age}d)")
    return f


def news_features(index: pd.DatetimeIndex, force_refresh=False) -> pd.DataFrame | None:
    daily = None if force_refresh else _load_daily()
    if daily is None:
        daily = _fetch_daily()
        if daily is None:
            log("no news fetched (offline?) — news features unavailable")
            return _load_daily(max_age_days=10_000)
        daily.to_parquet(CACHE_DAILY)
    out = daily.reindex(index.union(daily.index)).sort_index().ffill(limit=5).reindex(index)
    out.to_parquet(CACHE)
    return out


NEWS_FEATS = ["news_sent", "news_vol_z", "news_neg_share"]


if __name__ == "__main__":
    idx = pd.bdate_range("2024-01-01", pd.Timestamp.today())
    d = news_features(idx, force_refresh=True)
    print(d.dropna().tail() if d is not None else "no news (need network)")
