from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from . import events as ev
from .validation import permutation_recall_p

IF_FEATS = ["vol_z","rv5","rv21","gap","rs_sector","amihud","vix_chg"]

def conformal_alarm(resid_z, alpha=0.01, min_cal=250, regime=None, alpha_by_regime=None):
    s=resid_z.abs(); p=pd.Series(np.nan,index=s.index); vals=s.values
    for i in range(min_cal,len(vals)):
        if np.isnan(vals[i]): continue
        past=vals[:i]; past=past[~np.isnan(past)]
        if len(past)<min_cal: continue
        p.iloc[i]=(1+np.sum(past>=vals[i]))/(len(past)+1)
    out=pd.DataFrame({"resid_z":resid_z,"p_value":p})
    if regime is not None and alpha_by_regime:
        alphas=regime.reindex(s.index).map(lambda r: float(alpha_by_regime.get(str(r), alpha)))
        out["flag_conformal"]=(out["p_value"]<=alphas).astype(int)
    else:
        out["flag_conformal"]=(out["p_value"]<=alpha).astype(int)
    return out

def _wf_flag(X, fit_predict, refit_every=126, min_train=500):
    flag=pd.Series(0,index=X.index); i=min_train
    while i<len(X):
        tr,te=X.iloc[:i].values,X.iloc[i:i+refit_every].values
        try: flag.iloc[i:i+refit_every]=(fit_predict(tr,te)==-1).astype(int)
        except Exception: pass
        i+=refit_every
    return flag

def isoforest_flag(X, contamination=0.01):
    # FIX D3: threshold on the anomaly SCORE at a train-set quantile, instead of forcing a
    # fixed contamination quota. Flag rate is then data-driven (≈0 in calm regimes, higher in
    # turbulent ones) rather than a constant fraction by construction.
    def fp(tr,te):
        m=IsolationForest(n_estimators=200,contamination="auto",random_state=0).fit(tr)
        thr=np.quantile(m.score_samples(tr),contamination)   # lower score = more anomalous
        return np.where(m.score_samples(te)<thr,-1,1)
    return _wf_flag(X,fp)

def lof_flag(X, contamination=0.01, n_neighbors=20):
    def fp(tr,te):
        sc=StandardScaler().fit(tr); m=LocalOutlierFactor(n_neighbors=n_neighbors,contamination=contamination,novelty=True)
        m.fit(sc.transform(tr)); return m.predict(sc.transform(te))
    return _wf_flag(X,fp)

def detect(feats, alpha=0.01, contamination=0.01, detectors=("conformal","isoforest","lof"), min_votes=2, regime=None, alpha_by_regime=None):
    from .util import load_config
    cfg=load_config()
    if regime is None and "_regime" in feats.columns:
        regime=feats["_regime"]
    if alpha_by_regime is None:
        alpha_by_regime=cfg.get("anomaly_alpha_by_regime")
    cols=[c for c in IF_FEATS if c in feats.columns]
    X=feats[cols].replace([np.inf,-np.inf],np.nan).ffill().dropna()
    out=pd.DataFrame(index=feats.index); out["price"]=feats["price"]
    if "resid_z" in feats: out["resid_z"]=feats["resid_z"]
    votes=pd.Series(0,index=feats.index)
    if "conformal" in detectors and "resid_z" in feats:
        c=conformal_alarm(feats["resid_z"],alpha=alpha,regime=regime,alpha_by_regime=alpha_by_regime); out["p_value"]=c["p_value"]
        out["flag_conformal"]=c["flag_conformal"].fillna(0).astype(int); votes=votes.add(out["flag_conformal"],fill_value=0)
    if "isoforest" in detectors:
        out["flag_isoforest"]=isoforest_flag(X,contamination).reindex(feats.index).fillna(0).astype(int); votes=votes.add(out["flag_isoforest"],fill_value=0)
    if "lof" in detectors:
        out["flag_lof"]=lof_flag(X,contamination).reindex(feats.index).fillna(0).astype(int); votes=votes.add(out["flag_lof"],fill_value=0)
    out["votes"]=votes.fillna(0).astype(int); out["severity"]=out["votes"]; out["flag"]=(out["votes"]>=min_votes).astype(int)
    return out

def evaluate(feats, anom, tol_post=3):
    idx=anom.index; flag=anom["flag"].reindex(idx).fillna(0).astype(bool)
    ev_day=ev.event_index(idx); n_events=int(ev_day.sum()); pos=list(idx); caught=0
    for i,is_ev in enumerate(ev_day.values):
        if is_ev:
            lo,hi=max(0,i-1),min(len(pos),i+tol_post+1)
            if flag.iloc[lo:hi].any(): caught+=1
    recall=caught/n_events if n_events else float("nan")
    flagged=int(flag.sum()); flag_rate=flagged/len(flag) if len(flag) else float("nan")
    obs_recall,perm_p=permutation_recall_p(flag.values,ev_day.values,tol_post=tol_post)
    if "resid_z" in feats:
        fwd_absres=feats["resid_z"].abs().rolling(5).mean().shift(-5).reindex(idx)
        idio_sep=float(fwd_absres[flag].mean()/fwd_absres[~flag].mean()) if flag.any() and (~flag).any() else float("nan")
    else: idio_sep=float("nan")
    fwd_vol=(feats["ret"].rolling(5).std().shift(-5)*np.sqrt(252)).reindex(idx)
    vol_sep=float(fwd_vol[flag].mean()/fwd_vol[~flag].mean()) if flag.any() and (~flag).any() else float("nan")
    return {"n_events_known":n_events,"events_caught":caught,"event_recall":round(recall,3),
            "event_recall_vs_random_p":(round(perm_p,4) if perm_p==perm_p else None),
            "days_flagged":flagged,"flag_rate":round(flag_rate,4),
            "idio_forward_residual_separation":round(idio_sep,2),
            "total_vol_separation_partly_mechanical":round(vol_sep,2)}
