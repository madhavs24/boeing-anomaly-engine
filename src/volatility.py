from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .validation import moving_block_bootstrap_ci
warnings.filterwarnings("ignore")
TRADING = 252

def _targets(ret, horizon):
    rv_d=ret.rolling(5).std()*np.sqrt(TRADING); rv_w=ret.rolling(21).std()*np.sqrt(TRADING); rv_m=ret.rolling(63).std()*np.sqrt(TRADING)
    fwd=(ret.rolling(horizon).std()*np.sqrt(TRADING)).shift(-horizon)
    return rv_d,rv_w,rv_m,fwd

def _linear_wf(Xdf, y, min_train=500, embargo=0):
    X=np.column_stack([np.ones(len(Xdf))]+[Xdf[c].values for c in Xdf.columns]); yv=y.values; pred=pd.Series(np.nan,index=y.index)
    for i in range(min_train,len(y)):
        te=max(1,i-embargo); coef,*_=np.linalg.lstsq(X[:te],yv[:te],rcond=None); pred.iloc[i]=X[i]@coef
    return pred

def _model_preds(feats, horizon, embargo, min_train=500):
    ret=feats["ret"]; rv_d,rv_w,rv_m,fwd=_targets(ret,horizon)
    base=pd.concat({"d":rv_d,"w":rv_w,"m":rv_m,"y":fwd},axis=1).dropna()
    out={"actual":base["y"],"naive":base["d"]}
    out["har"]=_linear_wf(base[["d","w","m"]],base["y"],min_train,embargo)
    if "vix" in feats.columns:
        bx=base.join(feats["vix"]).dropna(); out["actual"]=bx["y"]; out["naive"]=bx["d"]
        out["har"]=out["har"].reindex(bx.index); out["harx"]=_linear_wf(bx[["d","w","m","vix"]],bx["y"],min_train,embargo)
        if "ba_iv" in feats.columns:
            bxi=bx.join(feats["ba_iv"]).dropna()
            if len(bxi)>min_train+50:
                out["harx_iv"]=_linear_wf(bxi[["d","w","m","vix","ba_iv"]],bxi["y"],min_train,embargo)
    var=ret.pow(2).ewm(alpha=1-0.94).mean(); out["ewma"]=(np.sqrt(var)*np.sqrt(TRADING))
    from .features import VOL_FEATS
    cols=[c for c in (VOL_FEATS+["vix","resid_rv21","vol_z"]) if c in feats.columns]
    dfh=feats[cols].join(fwd.rename("y")).replace([np.inf,-np.inf],np.nan).dropna()
    Xh,yh=dfh[cols].values,dfh["y"].values; ph=pd.Series(np.nan,index=dfh.index); i=min_train
    while i<len(dfh):
        te=max(1,i-embargo); m=HistGradientBoostingRegressor(max_depth=3,learning_rate=0.05,max_iter=300,l2_regularization=1.0).fit(Xh[:te],yh[:te])
        ph.iloc[i:i+63]=m.predict(Xh[i:i+63]); i+=63
    out["hgbr"]=ph
    try: out["garch"]=_garch(ret,horizon,embargo)
    except Exception as e: out["garch"]=pd.Series(np.nan,index=base.index); out["_garch_err"]=str(e)[:60]
    return out

def _garch(ret, horizon, embargo, min_train=750, refit_every=63):
    from arch import arch_model
    r=(ret*100).dropna(); pred=pd.Series(np.nan,index=ret.index); ridx=list(r.index); i=min_train; n_ok=0
    while i<len(ridx):
        te=max(10,i-embargo)
        try:
            res=arch_model(r.iloc[:te],p=1,q=1,vol="Garch",dist="normal").fit(disp="off")
            vp=res.forecast(horizon=horizon,reindex=False).variance.values[-1]; ann=np.sqrt(vp.mean())/100*np.sqrt(TRADING)
            for j in range(i,min(i+refit_every,len(ridx))): pred.loc[ridx[j]]=ann
            n_ok+=1
        except Exception: pass
        i+=refit_every
    return pred

def _oos_r2(a,p): return 1-np.nansum((a-p)**2)/np.nansum((a-np.nanmean(a))**2)
def _rmse(a,p): return float(np.sqrt(np.nanmean((a-p)**2)))

def evaluate(feats, horizon=5, with_garch=True):
    embargo=horizon; preds=_model_preds(feats,horizon,embargo); actual=preds["actual"]
    models=[m for m in ("naive","har","harx","harx_iv","ewma","garch","hgbr") if m in preds]
    common=actual.dropna().index
    for m in models:
        if not with_garch and m=="garch": continue
        common=common.intersection(preds[m].dropna().index)
    a=actual.reindex(common).values; naive_err=(a-preds["naive"].reindex(common).values)**2
    res={"n_common":int(len(common)),"horizon":horizon}
    for m in models:
        if not with_garch and m=="garch": continue
        p=preds[m].reindex(common).values; err=(a-p)**2; skill=1-err.sum()/naive_err.sum()
        res[m]={"r2_vs_mean":round(float(_oos_r2(a,p)),3),"skill_vs_naive":round(float(skill),3),"rmse":round(_rmse(a,p),4)}
    scored={m:res[m]["skill_vs_naive"] for m in models if m in res and m!="naive"}
    best=max(scored,key=scored.get) if scored else None
    if best:
        diff=naive_err-(a-preds[best].reindex(common).values)**2; lo,hi=moving_block_bootstrap_ci(diff,stat=np.mean)
        res["best_model"]=best; res["best_skill_vs_naive"]=scored[best]; res["best_beats_naive_95ci"]=bool(lo>0)
        res["latest_forecast"]=round(float(preds[best].dropna().iloc[-1]),3) if preds[best].notna().any() else None
    if "_garch_err" in preds: res["garch_note"]=preds["_garch_err"]
    return res
