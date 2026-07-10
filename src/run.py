from __future__ import annotations
import sys, json
from .util import RESULTS, load_config
from .data import get_panel, LAST_SOURCE
from .features import build_features
from . import anomaly, volatility, direction_pro

def main(mode="auto"):
    cfg=load_config(); panel=get_panel(mode); feats=build_features(panel)
    feats.to_parquet(RESULTS/"features.parquet")
    anom=anomaly.detect(feats,alpha=float(cfg.get("anomaly_alpha",0.01)),contamination=float(cfg.get("anomaly_alpha",0.01)),min_votes=int(cfg.get("anomaly_min_votes",2)))
    anom.to_csv(RESULTS/"anomalies.csv"); anom_eval=anomaly.evaluate(feats,anom)
    vol=volatility.evaluate(feats,horizon=int(cfg.get("vol_horizon",10))); dir_=direction_pro.evaluate(feats,anom)
    last=anom.dropna(subset=["price"]).iloc[-1]
    summary={"data_source":LAST_SOURCE["value"],"as_of":str(feats.index[-1].date()),"rows":int(len(feats)),
             "missing_series":panel.attrs.get("missing_series",[]),
             "anomaly":{**anom_eval,"today_votes":int(last.get("votes",0)),"today_resid_z":round(float(last.get("resid_z",float("nan"))),2)},
             "volatility":vol,"direction":dir_}
    json.dump(summary,open(RESULTS/"summary.json","w"),indent=2,default=str)
    print("\n[A] ANOMALY recall=%s/%s p=%s flag_rate=%s idio_sep=%s vol_sep=%s"%(anom_eval["events_caught"],anom_eval["n_events_known"],anom_eval["event_recall_vs_random_p"],anom_eval["flag_rate"],anom_eval["idio_forward_residual_separation"],anom_eval["total_vol_separation_partly_mechanical"]))
    print("[B] VOL best=%s skill=%s beats_naive_95ci=%s"%(vol.get("best_model"),vol.get("best_skill_vs_naive"),vol.get("best_beats_naive_95ci")))
    print("[C] DIR hgb acc=%s tradeable=%s | event5d up=%s base=%s"%(dir_["honest"]["hgb"]["accuracy"],dir_["honest"]["hgb"]["tradeable_after_costs"],dir_["event_conditional"]["5d"]["up_rate_after_flag"],dir_["event_conditional"]["5d"]["up_rate_baseline"]))
    return summary

if __name__=="__main__": main(sys.argv[1] if len(sys.argv)>1 else "auto")
