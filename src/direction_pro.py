from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from .features import build_features
warnings.filterwarnings("ignore")

RICH=["resid_z","rv5","rv21","rs_sector","vix","vix_chg","oil_ret","y10_chg","dd_252","vol_z",
      "ret_lag1","ret_lag2","ret_lag3","ret_lag5","mom5","mom10","mom20","rsi","ma_gap50","ma_gap200",
      "rs_RTX","rs_GE","gap","dow_0","dow_1","dow_2","dow_3","dow_4",
      "deliveries_m","orders_m","backlog_q","fcf_q","days_since_od_release",
      "regime_calm","regime_stress","regime_crisis"]
ROUND_TRIP_COST=0.001

def rich_features(feats):
    F=feats.copy(); px=F["price"]; r=F["ret"]
    for k in (1,2,3,5): F[f"ret_lag{k}"]=r.shift(k)
    F["mom5"]=px.pct_change(5); F["mom10"]=px.pct_change(10); F["mom20"]=px.pct_change(20)
    up=r.clip(lower=0).rolling(14).mean(); dn=(-r.clip(upper=0)).rolling(14).mean()
    F["rsi"]=100-100/(1+up/(dn+1e-9))
    F["ma_gap50"]=px/px.rolling(50).mean()-1; F["ma_gap200"]=px/px.rolling(200).mean()-1
    dow=F.index.dayofweek
    for k in range(5): F[f"dow_{k}"]=(dow==k).astype(float)
    return F

def honest_direction(feats, model="hgb", cols=None, refit=63, min_train=750, news=None):
    F=rich_features(feats); cols=[c for c in (cols or RICH) if c in F.columns]
    if news is not None: F=F.join(news); cols+=[c for c in news.columns if c not in cols]
    px=F["price"]; y=(px.shift(-1)>px).astype(int)
    df=F[cols].join(y.rename("y")).replace([np.inf,-np.inf],np.nan).dropna()
    X,yv=df[cols].values,df["y"].values; proba=pd.Series(np.nan,index=df.index); i=min_train
    while i<len(df)-1:
        te=max(1,i-1)  # embargo 1 (label looks 1 day ahead)
        if len(np.unique(yv[:te]))<2: i+=refit; continue
        if model=="logit":
            sc=StandardScaler().fit(X[:te]); m=LogisticRegression(max_iter=500,class_weight="balanced").fit(sc.transform(X[:te]),yv[:te])
            proba.iloc[i:i+refit]=m.predict_proba(sc.transform(X[i:i+refit]))[:,1]
        else:
            m=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=300,l2_regularization=1.0).fit(X[:te],yv[:te])
            proba.iloc[i:i+refit]=m.predict_proba(X[i:i+refit])[:,1]
        i+=refit
    o=pd.concat({"y":df["y"],"proba":proba},axis=1).dropna(); o["pred"]=(o["proba"]>0.5).astype(int); o["conf"]=(o["proba"]-0.5).abs()
    o["ret_fwd"]=F["ret"].shift(-1).reindex(o.index)
    acc=float((o["y"]==o["pred"]).mean()); n=len(o)
    auc=float(roc_auc_score(o["y"],o["proba"])) if o["y"].nunique()>1 else None
    curve={}
    for q in (1.0,0.5,0.3,0.1):
        thr=o["conf"].quantile(1-q); sub=o[o["conf"]>=thr]; curve[f"top_{int(q*100)}pct"]=round(float((sub["y"]==sub["pred"]).mean()),3)
    base=float(max(o["y"].mean(),1-o["y"].mean()))
    from .validation import binom_p
    succ=int((o["y"]==o["pred"]).sum())
    signed=np.where(o["pred"]==1,1.0,-1.0)*o["ret_fwd"].values-ROUND_TRIP_COST; net=float(np.nanmean(signed))
    return {"model":model,"n":n,"accuracy":round(acc,3),"auc":round(auc,3) if auc else None,
            "majority_baseline":round(base,3),"p_vs_coinflip":round(binom_p(succ,n,0.5),4),
            "p_vs_majority":round(binom_p(succ,n,base),4),"net_daily_pnl_after_costs_bps":round(net*1e4,2),
            "tradeable_after_costs":bool(net>0),"selective_acc":curve}

def event_conditional(feats, anom, horizons=(1,5,10,20)):
    px=feats["price"]; flag=anom["flag"].reindex(feats.index).fillna(0).astype(bool); after=flag.shift(1).fillna(False); out={}
    for h in horizons:
        fwd=px.shift(-h)/px-1; af=fwd[after].dropna()
        out[f"{h}d"]={"n":int(len(af)),"mean_ret_pct":round(float(af.mean()*100),2),
                      "up_rate_after_flag":round(float((af>0).mean()),2),"up_rate_baseline":round(float((fwd>0).mean()),2)}
    return out

def evaluate(feats, anom=None):
    res={"honest":{m:honest_direction(feats,m) for m in ("logit","hgb")}}
    if anom is not None: res["event_conditional"]=event_conditional(feats,anom)
    return res
