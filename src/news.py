"""News + sentiment layer (the one input with real literature support for a few points of
directional lift). Pulls Boeing headlines from Finnhub (free tier) and scores them with
FinBERT, producing daily features: mean sentiment, news volume, negative-headline share.

Setup (one time):
  1. Get a free key at https://finnhub.io  ->  set env var:  FINNHUB_API_KEY=xxxx
  2. pip install transformers torch    (FinBERT)   -- optional but recommended
  3. python -m src.news               # caches results/news_features.parquet

Without a key/transformers this module no-ops (returns None) so the rest of the engine still
runs. With them, pass the features into direction_pro.honest_direction(news=...).

NOTE: Finnhub free tier returns ~1 year of company news, so the sentiment features cover the
recent window only; older history will be NaN (handled downstream).
"""
from __future__ import annotations
import os, time, datetime as dt
import numpy as np
import pandas as pd
import requests
from .util import RESULTS, load_config, log

FINNHUB = "https://finnhub.io/api/v1/company-news"
CACHE = RESULTS / "news_features.parquet"


def _fetch_headlines(symbol, days=365, key=None):
    key = key or os.environ.get("FINNHUB_API_KEY")
    if not key:
        log("no FINNHUB_API_KEY set — news layer disabled"); return None
    end = dt.date.today(); rows = []
    # Finnhub caps each call's range; walk back in ~monthly chunks
    for back in range(0, days, 30):
        to = end - dt.timedelta(days=back); frm = to - dt.timedelta(days=30)
        try:
            r = requests.get(FINNHUB, params={"symbol": symbol, "from": frm.isoformat(),
                                              "to": to.isoformat(), "token": key}, timeout=15)
            r.raise_for_status()
            for a in r.json():
                rows.append({"ts": pd.to_datetime(a["datetime"], unit="s"),
                             "headline": a.get("headline", "")})
            time.sleep(0.3)
        except Exception as e:
            log(f"finnhub chunk failed: {e}")
    return pd.DataFrame(rows) if rows else None


def _finbert_scores(texts):
    """Return P(positive)-P(negative) per text in [-1,1]. Falls back to a lexicon if no model."""
    try:
        from transformers import pipeline
        clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)
        out = []
        for t in texts:
            scores = {d["label"].lower(): d["score"] for d in clf(t[:512])[0]}
            out.append(scores.get("positive", 0) - scores.get("negative", 0))
        return np.array(out)
    except Exception as e:
        log(f"FinBERT unavailable ({e}); using simple lexicon fallback")
        POS = {"beat", "surge", "win", "approval", "record", "profit", "upgrade", "deal", "order"}
        NEG = {"crash", "grounded", "halt", "probe", "strike", "loss", "cut", "delay", "fine", "fall", "plunge"}
        out = []
        for t in texts:
            w = set(t.lower().split())
            out.append((len(w & POS) - len(w & NEG)) / max(1, len(w & (POS | NEG))))
        return np.array(out)


def build_news_features(symbol=None, days=365) -> pd.DataFrame | None:
    cfg = load_config(); symbol = symbol or cfg["target"]
    df = _fetch_headlines(symbol, days)
    if df is None or df.empty:
        return None
    df["sent"] = _finbert_scores(df["headline"].tolist())
    df["date"] = df["ts"].dt.normalize()
    g = df.groupby("date")
    feat = pd.DataFrame({
        "news_sent": g["sent"].mean(),
        "news_vol": g.size(),
        "news_neg_share": g["sent"].apply(lambda s: float((s < -0.1).mean())),
    })
    feat["news_vol_z"] = (feat["news_vol"] - feat["news_vol"].rolling(20, min_periods=5).mean()) \
        / (feat["news_vol"].rolling(20, min_periods=5).std() + 1e-9)
    feat = feat.shift(1)                       # causal: yesterday's news predicts today
    feat.to_parquet(CACHE); log(f"news features cached -> {CACHE} ({len(feat)} days)")
    return feat


def load_news_features() -> pd.DataFrame | None:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    return build_news_features()


if __name__ == "__main__":
    f = build_news_features()
    print("no news features (set FINNHUB_API_KEY)" if f is None else f.tail())
