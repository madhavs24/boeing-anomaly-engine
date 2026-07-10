from __future__ import annotations
import io, time, os
import concurrent.futures as cf
import numpy as np
import pandas as pd
import requests
from .util import PROC, load_config, log

UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
PANEL=PROC/"panel.parquet"; SYNTH=PROC/"panel_synth.parquet"
TIMEOUT=int(os.environ.get("DATA_FETCH_TIMEOUT","12"))
FRED_TIMEOUT=int(os.environ.get("FRED_FETCH_TIMEOUT","30"))
LAST_SOURCE={"value":None}

def _yahoo(ticker, years, timeout=TIMEOUT):
    sym=ticker.replace("^","%5E")
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={years}y&interval=1d&events=div%2Csplit"
    r=requests.get(url,headers=UA,timeout=timeout); r.raise_for_status(); js=r.json()["chart"]["result"][0]
    ts=pd.to_datetime(js["timestamp"],unit="s").normalize(); q=js["indicators"]["quote"][0]
    adj=js["indicators"].get("adjclose",[{}])[0].get("adjclose"); raw=q["close"]; close=adj if adj is not None else raw
    return pd.DataFrame({"close":close,"close_raw":raw,"high":q["high"],"low":q["low"],"volume":q["volume"]},index=ts).dropna(how="all")

def _fred(series_id, timeout=FRED_TIMEOUT):
    url=f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r=requests.get(url,headers=UA,timeout=timeout); r.raise_for_status(); df=pd.read_csv(io.StringIO(r.text)); df.columns=["date","value"]
    df["date"]=pd.to_datetime(df["date"]); df["value"]=pd.to_numeric(df["value"],errors="coerce"); return df.set_index("date")["value"].dropna()

def _retry(fn,*a,tries=3,backoff=2.0,**k):
    last=None
    for i in range(tries):
        try: return fn(*a,**k)
        except Exception as e:
            last=e
            if i<tries-1: time.sleep(backoff*(2**i))
    raise last

def _hy_oas_proxy(yrs):
    """Imperfect HY OAS proxy from HYG/LQD price ratio when FRED BAMLH0A0HYM2 times out."""
    hyg=_retry(_yahoo,"HYG",yrs)["close"]; lqd=_retry(_yahoo,"LQD",yrs)["close"]
    ratio=(hyg/lqd).replace(0,np.nan)
    z=(ratio-ratio.rolling(252,min_periods=60).mean())/ratio.rolling(252,min_periods=60).std()
    return (4.0+z*1.5).clip(2.0,15.0)

def _fetch_fred_series(sid, name, yrs, fb):
    if fb == "__hyg_lqd__" or name == "hy_oas":
        s=_hy_oas_proxy(yrs); log(f"HYG/LQD proxy -> hy_oas ({len(s)})"); return s,"hyg_lqd_proxy"
    if fb:
        s=_retry(_yahoo,fb,yrs)["close"]; log(f"Yahoo fallback {fb} -> {name} ({len(s)})"); return s,"yahoo_fallback"
    raise RuntimeError(f"no fallback for {sid}/{name}")

def build_live():
    cfg=load_config(); yrs=int(cfg.get("history_years",10)); target,sec,mkt=cfg["target"],cfg["sector_etf"],cfg["market"]
    equities=[target,sec,mkt]+list(cfg.get("peers",[])); macro_y=cfg.get("macro_yahoo",{}); cols={}; sources={}
    fb_yahoo=dict(cfg.get("fred_fallback_yahoo",{})); fb_yahoo.setdefault("BAMLH0A0HYM2","__hyg_lqd__")
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(_retry,_yahoo,t,yrs):t for t in equities+list(macro_y)}
        for fut in cf.as_completed(futs):
            t=futs[fut]
            try:
                df=fut.result(); cols[t]=df["close"]
                if t==target: cols["BA_high"]=df["high"]; cols["BA_low"]=df["low"]; cols["BA_volume"]=df["volume"]
                log(f"Yahoo {t} ok ({len(df)})")
            except Exception as e: log(f"Yahoo {t} FAILED: {e}")
    for sym,name in macro_y.items():
        if sym in cols: cols[name]=cols.pop(sym)
    if "y10x10" in cols: cols["y10"]=cols.pop("y10x10")/10.0
    for sid,name in cfg.get("fred",{}).items():
        if name in cols and not cols[name].dropna().empty: continue
        try:
            cols[name]=_retry(_fred,sid,tries=3,timeout=FRED_TIMEOUT); sources[name]="fred"; log(f"FRED {sid} ok")
        except Exception as e:
            log(f"FRED {sid} skipped: {e}")
            try:
                cols[name],sources[name]=_fetch_fred_series(sid,name,yrs,fb_yahoo.get(sid))
            except Exception as e2: log(f"fallback for {name} FAILED: {e2}")
    critical=[target,sec,mkt,"vix"]; missing_crit=[c for c in critical if c not in cols or cols[c].dropna().empty]
    if missing_crit: raise RuntimeError(f"missing critical series {missing_crit} — aborting")
    want_peers=[p for p in cfg.get("peers",[]) if p in cols and not cols[p].dropna().empty]
    if len(want_peers)<3: raise RuntimeError(f"only {len(want_peers)} peers — need >=3")
    expected=set(equities)|set(macro_y.values())|set(cfg.get("fred",{}).values()); got={k for k in cols if not str(k).startswith("BA_")}
    panel_missing=sorted(expected-got-{"y10x10"})
    if panel_missing: log(f"WARNING non-critical missing: {panel_missing}")
    panel=pd.DataFrame(cols).sort_index(); panel=panel[panel.index>=pd.Timestamp.today()-pd.DateOffset(years=yrs+1)].ffill(limit=5)
    panel.attrs["missing_series"]=panel_missing; panel.attrs["sources"]=sources
    if len(panel)<500: raise RuntimeError(f"panel too short ({len(panel)})")
    return panel

def build_synthetic(n_days=2600, seed=11):
    rng=np.random.default_rng(seed); idx=pd.bdate_range(end=pd.Timestamp.today().normalize(),periods=n_days)
    mkt_vol=0.01*(1+0.5*np.abs(np.sin(np.linspace(0,12,n_days)))); r_mkt=rng.normal(0.0003,1,n_days)*mkt_vol
    r_sec=0.7*r_mkt+rng.normal(0,1,n_days)*0.011; e=rng.normal(0,1,n_days)*0.014
    events=rng.choice(range(60,n_days-5),size=14,replace=False)
    for c in events: e[c]+=rng.choice([-1,1])*rng.uniform(0.06,0.16)
    r_ba=1.0*r_mkt+0.8*(r_sec-0.7*r_mkt)+e
    def price(r,p0=100.0): return p0*np.cumprod(1+r)
    df=pd.DataFrame(index=idx); df["BA"]=price(r_ba,180); df["ITA"]=price(r_sec,110); df["SPY"]=price(r_mkt,400)
    for pk,b in [("RTX",0.6),("GE",0.7),("EADSY",0.5),("LMT",0.4),("NOC",0.4)]: df[pk]=price(b*r_sec+rng.normal(0,0.012,n_days),120)
    rng2=np.random.default_rng(seed+1); df["BA_high"]=df["BA"]*(1+np.abs(rng2.normal(0,0.008,n_days))); df["BA_low"]=df["BA"]*(1-np.abs(rng2.normal(0,0.008,n_days)))
    df["BA_volume"]=np.abs(rng2.normal(7e6,2e6,n_days))*(1+6*np.isin(np.arange(n_days),events))
    df["vix"]=np.clip(mkt_vol*np.sqrt(252)*100*rng2.uniform(0.8,1.2,n_days),9,80); df["oil"]=np.clip(70+np.cumsum(rng2.normal(0,0.4,n_days)),25,140); df["oil_etf"]=df["oil"]*0.9
    df["y10"]=np.clip(3+np.cumsum(rng2.normal(0,0.01,n_days)),0.5,6); df["dollar"]=100+np.cumsum(rng2.normal(0,0.05,n_days)); df["hy_oas"]=np.clip(3+(df["vix"]-18)*0.1+rng2.normal(0,0.2,n_days),2.5,12)
    return df

def get_panel(mode="auto"):
    if mode=="synthetic": LAST_SOURCE["value"]="synthetic"; p=build_synthetic(); p.to_parquet(SYNTH); return p
    if mode=="cached":
        if not PANEL.exists(): raise FileNotFoundError("no cached panel")
        LAST_SOURCE["value"]="cached"; return pd.read_parquet(PANEL)
    try:
        p=build_live(); LAST_SOURCE["value"]="live"; p.to_parquet(PANEL); log(f"panel saved: {p.shape} -> {PANEL}"); print(p.tail()); print(f"source: live"); print(list(p.columns)); return p
    except Exception as e:
        if mode=="live": raise
        if PANEL.exists(): log(f"live failed ({e}); using cache"); LAST_SOURCE["value"]="cached"; return pd.read_parquet(PANEL)
        log(f"live failed ({e}); synthetic"); LAST_SOURCE["value"]="synthetic"; p=build_synthetic(); p.to_parquet(SYNTH); return p

if __name__=="__main__":
    import sys
    get_panel(sys.argv[1] if len(sys.argv)>1 else "live")
