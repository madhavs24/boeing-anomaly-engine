from __future__ import annotations
import sys, json
import pandas as pd
from .util import RESULTS, load_config
from .data import get_panel
from .features import build_features, VOL_FEATS, DIR_FEATS
from . import anomaly, volatility, direction_pro

def sweep_volatility(feats, horizons=(5,10,21)):
    rows=[]
    for h in horizons:
        ev=volatility.evaluate(feats,horizon=h,with_garch=(h==5))
        for model in ("naive","har","harx","ewma","garch","hgbr"):
            if model in ev and isinstance(ev[model],dict) and "skill_vs_naive" in ev[model]:
                rows.append({"horizon":h,"model":model,"skill_vs_naive":ev[model]["skill_vs_naive"],"r2_vs_mean":ev[model]["r2_vs_mean"],"rmse":ev[model]["rmse"]})
    return pd.DataFrame(rows)

def sweep_direction(feats):
    rows=[]; feat_sets={"rich":direction_pro.RICH,"lean":["resid_z","rv21","vix","vix_chg","dd_252"],"macro_only":["vix","vix_chg","oil_ret","y10_chg"]}
    for fs,cols in feat_sets.items():
        for model in ("logit","hgb"):
            s=direction_pro.honest_direction(feats,model=model,cols=cols)
            if not s or s.get("n",0)==0: continue
            rows.append({"feature_set":fs,"model":model,"n":s["n"],"accuracy":s["accuracy"],"auc":s["auc"],"p_vs_coinflip":s["p_vs_coinflip"],"net_bps_after_costs":s["net_daily_pnl_after_costs_bps"],"tradeable":s["tradeable_after_costs"]})
    return pd.DataFrame(rows)

def sweep_anomaly(feats):
    rows=[]; combos=[("conformal",),("isoforest",),("lof",),("conformal","isoforest"),("conformal","isoforest","lof")]
    for alpha in (0.005,0.01,0.02):
        for dets in combos:
            for mv in (1, max(1,len(dets)-1) if len(dets)>1 else 1):
                d=anomaly.detect(feats,alpha=alpha,contamination=alpha,detectors=dets,min_votes=mv); e=anomaly.evaluate(feats,d)
                rows.append({"alpha":alpha,"detectors":"+".join(dets),"min_votes":mv,"recall":e["event_recall"],"recall_p":e["event_recall_vs_random_p"],"flag_rate":e["flag_rate"],"idio_sep":e["idio_forward_residual_separation"],"vol_sep":e["total_vol_separation_partly_mechanical"]})
    df=pd.DataFrame(rows).drop_duplicates(subset=["alpha","detectors","min_votes"])
    df["score"]=((df["idio_sep"].fillna(1)-1).clip(lower=0)*0.4+df["recall"].fillna(0)*0.4-(df["flag_rate"].fillna(0)*2).clip(upper=0.3))
    return df.sort_values("score",ascending=False)

def main(mode="auto"):
    feats=build_features(get_panel(mode))
    vol=sweep_volatility(feats); dir_=sweep_direction(feats); anom=sweep_anomaly(feats)
    n_trials=int(len(vol)+len(dir_)+len(anom))
    print("VOL top:"); print(vol.sort_values("skill_vs_naive",ascending=False).head(4).to_string(index=False))
    print("\nDIR:"); print(dir_.sort_values("auc",ascending=False).to_string(index=False))
    print("\nANOM top:"); print(anom.head(4).to_string(index=False))
    print(f"\n(selection-bias caveat: best-of {n_trials} trials on same data; true OOS lower)")
    return {"n_trials":n_trials}

if __name__=="__main__": main(sys.argv[1] if len(sys.argv)>1 else "auto")
