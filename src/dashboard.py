"""Stage 5 — Web app. Generate a self-contained interactive dashboard.html from the engine's
results (results/anomalies.csv, summary.json, experiments_*.csv). Open the file in any browser
(no server needed). Re-run after `python -m src.run` + `python -m src.experiment` to refresh.

    python -m src.dashboard
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .util import ROOT, RESULTS
from . import events as EV


def _load():
    anom = pd.read_csv(RESULTS / "anomalies.csv", index_col=0, parse_dates=True)
    summary = json.load(open(RESULTS / "summary.json"))
    def _csv(name):
        p = RESULTS / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    return anom, summary, _csv("experiments_vol.csv"), _csv("experiments_dir.csv"), _csv("experiments_anom.csv")


def build(out_path=None) -> str:
    anom, summary, vol, dirdf, anomdf = _load()
    anom = anom.dropna(subset=["price"])
    dates = [d.strftime("%Y-%m-%d") for d in anom.index]
    price = [round(float(x), 2) for x in anom["price"]]
    flag = anom["flag"].fillna(0).astype(int).tolist() if "flag" in anom else [0] * len(anom)
    resid = anom["resid_z"].round(2).where(anom["resid_z"].notna(), None).tolist() if "resid_z" in anom else [None] * len(anom)
    votes = anom["votes"].fillna(0).astype(int).tolist() if "votes" in anom else [0] * len(anom)

    # flagged points (date,price,resid,votes)
    flagged = [{"x": dates[i], "y": price[i], "z": resid[i], "v": votes[i]}
               for i in range(len(dates)) if flag[i] == 1]

    # event markers mapped to nearest available date
    ev_points = []
    idx = anom.index
    for dstr, desc in EV.BOEING_EVENTS:
        fut = idx[idx >= pd.Timestamp(dstr)]
        if len(fut):
            ed = fut[0]
            ev_points.append({"x": ed.strftime("%Y-%m-%d"),
                              "y": round(float(anom.loc[ed, "price"]), 2), "desc": desc})

    # per-event detection table
    ev_rows = []
    pos = list(idx)
    for dstr, desc in EV.BOEING_EVENTS:
        fut = idx[idx >= pd.Timestamp(dstr)]
        if not len(fut):
            continue
        ed = fut[0]; i = pos.index(ed); lo, hi = max(0, i - 1), min(len(pos), i + 4)
        caught = bool(anom["flag"].iloc[lo:hi].any()) if "flag" in anom else False
        rz = anom.loc[ed, "resid_z"] if "resid_z" in anom else np.nan
        ev_rows.append({"date": ed.strftime("%Y-%m-%d"), "desc": desc,
                        "resid_z": None if pd.isna(rz) else round(float(rz), 2), "caught": caught})

    vol_rows = vol.sort_values("r2", ascending=False).to_dict("records") if len(vol) else []
    dir_rows = dirdf.sort_values("auc", ascending=False).to_dict("records") if len(dirdf) else []

    payload = {
        "summary": summary, "dates": dates, "price": price,
        "flagged": flagged, "events": ev_points, "ev_rows": ev_rows,
        "vol_rows": vol_rows, "dir_rows": dir_rows,
        "recent_flags": flagged[-12:][::-1],
    }

    html = _TEMPLATE.replace("/*DATA*/", json.dumps(payload, default=str))
    out_path = out_path or (ROOT / "dashboard.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"dashboard written -> {out_path}")
    return str(out_path)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Boeing (BA) Anomaly Engine</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#0b1020;--card:#151c33;--ink:#e8ecf6;--mut:#94a0bd;--acc:#4da3ff;--red:#ff5c6c;--grn:#37d39b;--amb:#ffb454;--line:#26304f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{font-size:24px;margin:0 0 2px}.sub{color:var(--mut);font-size:13px;margin-bottom:20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin-bottom:22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.card .lab{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.card .val{font-size:24px;font-weight:650;margin-top:4px}
.card .note{font-size:12px;color:var(--mut);margin-top:2px}
.sec{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:20px}
.sec h2{font-size:16px;margin:0 0 4px}.sec p.d{color:var(--mut);font-size:13px;margin:0 0 14px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600}.yes{color:var(--grn)}.no{color:var(--red)}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600}
.pill.ok{background:rgba(55,211,155,.15);color:var(--grn)}.pill.warn{background:rgba(255,92,108,.15);color:var(--red)}
.foot{color:var(--mut);font-size:12px;margin-top:8px}
canvas{max-width:100%}
.bar{height:9px;background:var(--line);border-radius:6px;overflow:hidden;display:inline-block;width:120px;vertical-align:middle}
.bar>i{display:block;height:100%;background:var(--acc)}
</style></head><body><div class="wrap">
<h1>Boeing (BA) Anomaly &amp; Prediction Engine</h1>
<div class="sub" id="sub"></div>
<div class="cards" id="cards"></div>

<div class="sec">
  <h2>Anomaly timeline</h2>
  <p class="d">Boeing price with detected idiosyncratic anomalies (red) and known real events (amber). Hover for details.</p>
  <canvas id="ts" height="320"></canvas>
</div>

<div class="sec">
  <h2>Volatility forecast — model leaderboard</h2>
  <p class="d">Out-of-sample R² (walk-forward). Higher is better; naive is the bar to beat.</p>
  <table id="voltab"></table>
</div>

<div class="sec">
  <h2>Known-event detection</h2>
  <p class="d">Did the detector fire within ±3 trading days of each curated Boeing event? "resid_z" is the idiosyncratic move (in std devs).</p>
  <table id="evtab"></table>
</div>

<div class="sec">
  <h2>Direction — honesty check</h2>
  <p class="d">Next-day up/down. For a liquid large-cap this is near-random; we report it plainly rather than overfit.</p>
  <table id="dirtab"></table>
</div>

<div class="sec">
  <h2>Recent flags</h2>
  <table id="recent"></table>
  <div class="foot">Research / education tool. Not investment advice. No real trades.</div>
</div>
</div>
<script>
const D = /*DATA*/;
const S = D.summary;
document.getElementById('sub').textContent =
  `As of ${S.as_of} · data: ${S.data_source} · ${S.rows} trading days · BA 2016–2026`;

// cards
const a=S.anomaly||{}, v=S.volatility||{}, dr=S.direction||{};
const best=v.best_model, bestr2=v.ceiling_r2;
const todayWarn=(a.today_votes||0)>=2;
const cards=[
 {lab:'Today status', val: todayWarn?'⚠ Anomaly':'Normal', note:`votes ${a.today_votes||0}/3 · resid_z ${a.today_resid_z}`, warn:todayWarn},
 {lab:'Vol forecast ('+(S.volatility&&S.volatility.har?'10d':'')+')', val:(v.harx&&v.harx.latest!=null)?(v.harx.latest):(v.har&&v.har.latest!=null?v.har.latest:'—'), note:'annualized, HARX'},
 {lab:'Vol model R²', val: bestr2!=null?bestr2:'—', note:`best: ${best||'—'}`},
 {lab:'Events caught', val:`${a.events_caught||0}/${a.n_events_known||0}`, note:`flag rate ${(a.flag_rate*100||0).toFixed(1)}%`},
 {lab:'Vol separation', val:(a.fwd_vol_ratio_flag_vs_calm||'—')+'×', note:'flagged vs calm fwd-vol'},
 {lab:'Direction AUC', val: dr.ceiling_auc!=null?dr.ceiling_auc:'—', note: dr.any_beats_baseline?'beats baseline':'≈ coin flip'},
];
document.getElementById('cards').innerHTML=cards.map(c=>
 `<div class="card"><div class="lab">${c.lab}</div><div class="val" style="color:${c.warn?'var(--red)':'var(--ink)'}">${c.val}</div><div class="note">${c.note}</div></div>`).join('');

// timeline
const ctx=document.getElementById('ts');
new Chart(ctx,{type:'line',
 data:{labels:D.dates,datasets:[
  {label:'BA price',data:D.price,borderColor:'#4da3ff',borderWidth:1.2,pointRadius:0,tension:.1},
  {label:'Anomaly',type:'scatter',data:D.flagged.map(p=>({x:p.x,y:p.y,z:p.z,v:p.v})),
   backgroundColor:'#ff5c6c',pointRadius:4,pointHoverRadius:6,parsing:false},
  {label:'Known event',type:'scatter',data:D.events.map(p=>({x:p.x,y:p.y,desc:p.desc})),
   backgroundColor:'#ffb454',pointStyle:'triangle',pointRadius:6,parsing:false},
 ]},
 options:{interaction:{mode:'nearest',intersect:true},plugins:{legend:{labels:{color:'#94a0bd'}},
  tooltip:{callbacks:{label:c=>{const r=c.raw;
    if(r.desc!==undefined)return '★ '+r.desc;
    if(r.z!==undefined)return `Anomaly ${r.x} · resid_z ${r.z} · votes ${r.v}`;
    return 'BA $'+c.formattedValue;}}}},
  scales:{x:{type:'category',ticks:{color:'#94a0bd',maxTicksLimit:10},grid:{color:'#26304f'}},
          y:{ticks:{color:'#94a0bd'},grid:{color:'#26304f'}}}}});

// vol table
const vt=document.getElementById('voltab');
const vmax=Math.max(0.001,...D.vol_rows.map(r=>r.r2||0));
vt.innerHTML='<tr><th>Model</th><th>Horizon</th><th>OOS R²</th><th></th><th>RMSE</th></tr>'+
 D.vol_rows.map(r=>`<tr><td>${r.model}</td><td>${r.horizon}d</td><td>${(r.r2!=null?r.r2:'—')}</td>
   <td><span class="bar"><i style="width:${Math.max(0,(r.r2||0)/vmax*100)}%"></i></span></td>
   <td>${r.rmse!=null?r.rmse:'—'}</td></tr>`).join('');

// event table
const et=document.getElementById('evtab');
et.innerHTML='<tr><th>Date</th><th>Event</th><th>resid_z</th><th>Detected</th></tr>'+
 D.ev_rows.map(r=>`<tr><td>${r.date}</td><td>${r.desc}</td><td>${r.resid_z!=null?r.resid_z:'—'}</td>
   <td>${r.caught?'<span class="pill ok">caught</span>':'<span class="pill warn">missed</span>'}</td></tr>`).join('');

// dir table
const dt=document.getElementById('dirtab');
dt.innerHTML='<tr><th>Model</th><th>Features</th><th>Accuracy</th><th>AUC</th><th>Coinflip</th><th>Beats?</th></tr>'+
 D.dir_rows.map(r=>`<tr><td>${r.model}</td><td>${r.feature_set}</td><td>${r.accuracy}</td><td>${r.auc}</td>
   <td>${r.coinflip}</td><td>${r.beats_baseline?'<span class="yes">yes</span>':'<span class="no">no</span>'}</td></tr>`).join('');

// recent flags
const rt=document.getElementById('recent');
rt.innerHTML='<tr><th>Date</th><th>Price</th><th>resid_z</th><th>Votes</th></tr>'+
 (D.recent_flags.length?D.recent_flags.map(r=>`<tr><td>${r.x}</td><td>$${r.y}</td><td>${r.z}</td><td>${r.v}/3</td></tr>`).join('')
  :'<tr><td colspan="4" style="color:var(--mut)">no flags in range</td></tr>');
</script></body></html>"""


if __name__ == "__main__":
    build()
