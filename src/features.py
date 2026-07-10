from __future__ import annotations
import numpy as np
import pandas as pd
from .util import load_config
TRADING = 252

def _ret(s): return np.log(s).diff()

def rolling_residual(r_ba, r_mkt, r_sec, window=252, min_obs=120):
    df = pd.concat({"ba": r_ba, "mkt": r_mkt, "sec": r_sec}, axis=1).dropna()
    z = pd.Series(np.nan, index=df.index)
    X_all = df[["mkt","sec"]].values; y_all = df["ba"].values; n=len(df)
    for i in range(min_obs, n):
        lo=max(0,i-window); Xw,yw=X_all[lo:i],y_all[lo:i]
        m=Xw[:,0]; b=np.polyfit(m,Xw[:,1],1); sec_res_w=Xw[:,1]-(b[0]*m+b[1])
        A=np.column_stack([np.ones_like(m),m,sec_res_w])
        coef,*_=np.linalg.lstsq(A,yw,rcond=None); resid_w=yw-A@coef; sd=resid_w.std()
        mi,si=X_all[i,0],X_all[i,1]; sec_res_i=si-(b[0]*mi+b[1])
        pred_i=coef[0]+coef[1]*mi+coef[2]*sec_res_i
        z.iloc[i]=(y_all[i]-pred_i)/sd if sd>0 else np.nan
    return z.reindex(r_ba.index)

def build_features(panel):
    cfg=load_config(); tgt,sec,mkt=cfg["target"],cfg["sector_etf"],cfg["market"]
    peers=[p for p in cfg.get("peers",[]) if p in panel.columns]
    px=panel[tgt].ffill(); r=_ret(px)
    f=pd.DataFrame(index=panel.index); f["price"]=px; f["ret"]=r
    f["rv5"]=r.rolling(5).std()*np.sqrt(TRADING); f["rv21"]=r.rolling(21).std()*np.sqrt(TRADING); f["rv63"]=r.rolling(63).std()*np.sqrt(TRADING)
    f["gap"]=(np.log(px)-np.log(px.shift(1))).abs()
    if "BA_high" in panel and "BA_low" in panel: f["range"]=(panel["BA_high"]-panel["BA_low"])/px
    f["ret_skew21"]=r.rolling(21).skew(); f["dd_252"]=px/px.rolling(252,min_periods=60).max()-1; f["dist_ma50"]=px/px.rolling(50).mean()-1
    if "BA_volume" in panel:
        vol=panel["BA_volume"].replace(0,np.nan).ffill()
        f["vol_z"]=(vol-vol.rolling(63).mean())/vol.rolling(63).std()
        f["amihud"]=(r.abs()/(vol*px).replace(0,np.nan)).rolling(21).mean()*1e9
    if peers and mkt in panel:
        peer_ret=pd.concat([_ret(panel[p]) for p in peers],axis=1)
        sec_ret=peer_ret.median(axis=1).clip(-0.25,0.25)
    elif sec in panel: sec_ret=_ret(panel[sec]).clip(-0.25,0.25)
    else: sec_ret=None
    if sec_ret is not None and mkt in panel:
        f["resid_z"]=rolling_residual(r,_ret(panel[mkt]).clip(-0.25,0.25),sec_ret)
        f["resid_abs"]=f["resid_z"].abs(); f["resid_rv21"]=f["resid_z"].rolling(21).std()
    if sec in panel: f["rs_sector"]=r-_ret(panel[sec])
    for p in peers: f[f"rs_{p}"]=(r-_ret(panel[p])).rolling(5).mean()
    if "oil" in panel: f["oil_ret"]=_ret(panel["oil"])
    elif "oil_etf" in panel: f["oil_ret"]=_ret(panel["oil_etf"])
    if "y10" in panel: f["y10"]=panel["y10"]; f["y10_chg"]=panel["y10"].diff(5)
    if "dollar" in panel: f["dollar_ret"]=_ret(panel["dollar"])
    if "vix" in panel: f["vix"]=panel["vix"]; f["vix_chg"]=panel["vix"].diff()
    if "hy_oas" in panel: f["hy_chg"]=panel["hy_oas"].diff(21)
    cfg = load_config()
    if cfg.get("use_operational", False):
        from .operational import build_operational_features
        op = build_operational_features(f.index)
        if not op.empty:
            f = f.join(op, how="left")
    if cfg.get("use_options_iv", True):
        try:
            from .options import build_ba_iv_series
            iv = build_ba_iv_series(f.index)
            if not iv.empty:
                f = f.join(iv, how="left")
        except Exception:
            pass
    if cfg.get("use_regime", True):
        from .regime import fit_expanding_regime, regime_one_hot
        reg = fit_expanding_regime(f)
        f = f.join(regime_one_hot(reg))
        f["_regime"] = reg
    return f

ANOMALY_FEATS=["resid_z","resid_abs","vol_z","rv5","rv21","gap","rs_sector","amihud","vix_chg"]
VOL_FEATS=["rv5","rv21","rv63"]
DIR_FEATS=["resid_z","rv5","rv21","rs_sector","vix","vix_chg","oil_ret","y10_chg","dd_252","vol_z",
           "deliveries_m","orders_m","backlog_q","fcf_q","days_since_od_release",
           "regime_calm","regime_stress","regime_crisis"]
OP_FEATS=["deliveries_m","orders_m","backlog_q","fcf_q","revenue_q","days_since_od_release","days_since_faa_event"]
