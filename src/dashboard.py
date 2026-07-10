"""Web app — self-contained interactive dashboard focused on DETECTION + RISK MANAGEMENT.
Generates dashboard.html (open in any browser, no server). Re-run after data refresh.

    python -m src.dashboard
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from .util import ROOT, RESULTS
from .features import build_features
from .data import get_panel
from . import anomaly as A, strategy as S, detection as DET, events as EV


def _equity(ret, start=10000): return (start * (1 + ret.fillna(0)).cumprod())


def build(mode="cached", out_path=None):
    feats = build_features(get_panel(mode))
    anom = A.detect(feats, min_votes=2)
    aev = A.evaluate(feats, anom)
    # detection extras
    pcal, fmetrics = DET.calibrated_flare(feats, refit=252)
    palerts = DET.precision_alerts(pcal, feats)
    _, _, aci_rate = DET.adaptive_conformal(feats["resid_z"], target=0.01)
    # risk strategies
    bt = S.backtest(feats, anom)
    r = feats["ret"]
    flag = anom["flag"].reindex(feats.index).fillna(0)
    derisk = flag.rolling(5, min_periods=1).max().shift(1).fillna(0)
    pos = {"buy_hold": pd.Series(1.0, index=r.index), "derisk_anomaly": (1 - derisk)}
    curves = {k: _equity((v.shift(1).fillna(1 if k == "buy_hold" else 0) * r)) for k, v in pos.items()}
    dd = {k: (c / c.cummax() - 1) for k, c in curves.items()}

    idx = anom.dropna(subset=["price"]).index
    dates = [d.strftime("%Y-%m-%d") for d in idx]
    flagged = [{"x": d.strftime("%Y-%m-%d"), "y": round(float(anom.loc[d, "price"]), 2),
                "z": round(float(anom.loc[d, "resid_z"]), 2)}
               for d in idx if anom.loc[d, "flag"] == 1]
    ev_pts = []
    for ds, desc in EV.BOEING_EVENTS:
        fut = idx[idx >= pd.Timestamp(ds)]
        if len(fut): ev_pts.append({"x": fut[0].strftime("%Y-%m-%d"),
                                    "y": round(float(anom.loc[fut[0], "price"]), 2), "desc": desc})
    last = anom.dropna(subset=["price"]).iloc[-1]

    # ---- industry signals (context only — excluded from the model per A/B results) ----
    demand_pl = op_pl = news_pl = None
    try:
        from .demand import demand_features
        dm = demand_features(feats.index, allow_synthetic=False)
        if dm is not None and dm["demand_yoy"].notna().any():
            dmi = dm.reindex(idx)
            demand_pl = {"yoy": [None if pd.isna(v) else round(float(v), 3) for v in dmi["demand_yoy"]]}
    except Exception:
        pass
    try:
        try:
            from .edgar import operational_features, OP_FEATS
        except ImportError:
            from .operational import operational_features, OP_FEATS
        op = operational_features(feats.index)
        if op is not None and op[OP_FEATS].notna().any().any():
            ol = op.dropna(subset=["op_revenue_yoy"]).iloc[-1]
            op_pl = {"revenue_yoy": round(float(ol["op_revenue_yoy"]), 3),
                     "ocf_bn": (round(float(ol["op_ocf"]), 2) if pd.notna(ol.get("op_ocf")) else None),
                     "days_since": (int(ol["op_days_since"]) if pd.notna(ol.get("op_days_since")) else None)}
    except Exception:
        pass
    try:
        from .news import news_features
        nw = news_features(feats.index)
        if nw is not None and nw["news_sent"].notna().any():
            nwi = nw.reindex(idx)
            nl = nw.dropna(subset=["news_sent"]).iloc[-1]
            news_pl = {"sent": [None if pd.isna(v) else round(float(v), 3) for v in nwi["news_sent"]],
                       "latest": round(float(nl["news_sent"]), 3)}
    except Exception:
        pass

    payload = {
        "as_of": dates[-1], "rows": len(feats),
        "today": {"votes": int(last["votes"]), "resid_z": round(float(last["resid_z"]), 2),
                  "flare_prob": (round(float(pcal.dropna().iloc[-1]), 3) if pcal.notna().any() else None)},
        "anom_eval": aev, "flare_metrics": fmetrics, "precision_alerts": palerts,
        "aci_rate": round(float(aci_rate), 4),
        "dates": dates, "price": [round(float(x), 2) for x in anom.loc[idx, "price"]],
        "flare_series": [None if pd.isna(v) else round(float(v), 3) for v in pcal.reindex(idx)],
        "flagged": flagged, "events": ev_pts,
        "eq_bh": [round(float(x)) for x in curves["buy_hold"].reindex(idx)],
        "eq_dr": [round(float(x)) for x in curves["derisk_anomaly"].reindex(idx)],
        "dd_bh": [round(float(x) * 100, 1) for x in dd["buy_hold"].reindex(idx)],
        "dd_dr": [round(float(x) * 100, 1) for x in dd["derisk_anomaly"].reindex(idx)],
        "stats": bt.reset_index().rename(columns={"index": "strategy"}).to_dict("records"),
        "demand": demand_pl, "operational": op_pl, "news": news_pl,
    }
    html = _TEMPLATE.replace("/*DATA*/", json.dumps(payload, default=str))
    out_path = out_path or (ROOT / "dashboard.html")
    open(out_path, "w", encoding="utf-8").write(html)
    print(f"dashboard -> {out_path}")
    return str(out_path)


_TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Boeing Risk &amp; Anomaly Monitor</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.2.0/dist/chartjs-plugin-zoom.min.js"></script>
<style>
:root{--bg:#0b1020;--card:#151c33;--ink:#e8ecf6;--mut:#94a0bd;--acc:#4da3ff;--red:#ff5c6c;--grn:#37d39b;--amb:#ffb454;--line:#26304f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:24px}h1{font-size:23px;margin:0 0 2px}.sub{color:var(--mut);font-size:13px;margin-bottom:18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:13px 15px}
.card .lab{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em}.card .val{font-size:22px;font-weight:650;margin-top:3px}.card .note{font-size:12px;color:var(--mut)}
.sec{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:18px}
.sec h2{font-size:16px;margin:0 0 3px}.sec p.d{color:var(--mut);font-size:13px;margin:0 0 12px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}th{color:var(--mut)}
.good{color:var(--grn)}.bad{color:var(--red)}.foot{color:var(--mut);font-size:12px}
canvas{max-width:100%}
.chart-wrap{position:relative}.chart-tools{display:flex;align-items:center;gap:6px;margin-bottom:10px;flex-wrap:wrap}
.chart-tools button{background:var(--line);color:var(--ink);border:1px solid #334066;border-radius:6px;padding:4px 11px;font-size:13px;font-weight:600;cursor:pointer;line-height:1.2}
.chart-tools button:hover{background:#334066}.chart-hint{color:var(--mut);font-size:11px;margin-left:2px}
.q{border-bottom:1px dotted var(--mut);cursor:help}
.guide{display:grid;gap:10px}.guide .g{display:flex;gap:10px;align-items:flex-start}.guide .ic{font-size:18px;line-height:1.3;flex:0 0 24px;text-align:center}
.guide b{color:var(--ink)}.guide .g div{color:var(--mut);font-size:13px}
details.sec summary{cursor:pointer;font-size:16px;font-weight:650;list-style:none;display:flex;justify-content:space-between;align-items:center}
details.sec summary::after{content:'▸';color:var(--mut);transition:transform .15s}details.sec[open] summary::after{transform:rotate(90deg)}
</style></head><body><div class="wrap">
<h1>Boeing (BA) — Risk &amp; Anomaly Monitor</h1><div class="sub" id="sub"></div>
<div class="sec" id="livebar" style="display:flex;justify-content:space-between;align-items:center;gap:12px"><div id="liveinfo" style="font-size:14px">Live panel…</div><button id="refbtn" style="background:var(--acc);color:#06122b;border:0;border-radius:8px;padding:7px 14px;font-weight:600;cursor:pointer">Refresh</button></div>
<details class="sec" open><summary>How to read this dashboard</summary>
<div class="guide" style="margin-top:12px">
<div class="g"><span class="ic">🟥</span><div><b>Anomalies (the math)</b> — days where Boeing's stock moved drastically differently from the rest of the aerospace sector — an "idiosyncratic" move that market-wide news can't explain. Three independent detectors vote; 2+ votes raises a flag.</div></div>
<div class="g"><span class="ic">🔶</span><div><b>Events (the reality)</b> — curated dates of real, known Boeing incidents (737 MAX groundings, the door-plug blowout, DOJ charges…). We use these to grade the detector: did the math catch the real event?</div></div>
<div class="g"><span class="ic">📈</span><div><b>Flare probability (the forecast)</b> — the model's estimate that Boeing will have an abnormal move <i>within the next 5 days</i>. This is the "predict the market" part: it forecasts risk (how bumpy), not direction (up vs down) — direction is honestly near a coin flip.</div></div>
<div class="g"><span class="ic">💵</span><div><b>Risk management (the payoff)</b> — what happens to $10k if you simply step to cash for 5 days after each anomaly flag, vs holding through everything. Fewer deep losses is the goal.</div></div>
</div></details>
<div class="cards" id="cards"></div>
<div class="sec"><h2>Anomaly timeline — what we flagged vs what really happened</h2><p class="d">Boeing's price with days the math flagged as abnormal (red dots) and real, documented Boeing incidents (amber triangles). When a red dot sits near an amber triangle, the detector caught a real event. Hover any point for a plain-English explanation.</p><div class="chart-wrap"><div class="chart-tools"><button type="button" data-chart="ts" data-zoom="in" title="Zoom in">+</button><button type="button" data-chart="ts" data-zoom="out" title="Zoom out">−</button><button type="button" data-chart="ts" data-zoom="reset" title="Reset zoom">Reset</button><span class="chart-hint">Scroll or pinch to zoom · drag to pan</span></div><canvas id="ts" height="300"></canvas></div></div>
<div class="sec"><h2>Flare probability — "will Boeing behave abnormally in the next 5 days?"</h2><p class="d">The model's forward-looking risk forecast. Higher = elevated chance of a big move soon. It predicts <b>risk</b> (how bumpy), not direction (up vs down) — daily direction is famously near a coin flip, and we don't pretend otherwise. Tested only on data the model had never seen (<span class="q" title="Walk-forward: the model is trained on the past only, predicts forward, then retrains — no peeking at the future. Calibrated: a 30% forecast really happens about 30% of the time.">how?</span>).</p><div class="chart-wrap"><div class="chart-tools"><button type="button" data-chart="flare" data-zoom="in" title="Zoom in">+</button><button type="button" data-chart="flare" data-zoom="out" title="Zoom out">−</button><button type="button" data-chart="flare" data-zoom="reset" title="Reset zoom">Reset</button><span class="chart-hint">Scroll or pinch to zoom · drag to pan</span></div><canvas id="flare" height="170"></canvas></div></div>
<div class="sec"><h2>Risk management — did the alarm actually save money?</h2><p class="d">Imagine you invested <b>$10,000 in Boeing in 2016</b>. The <b style="color:var(--grn)">green line</b> sells to cash for 5 days every time the alarm rings, then buys back in. The <b style="color:#94a0bd">grey line</b> ignores the alarm and just holds. If green stays above grey, the alarm is earning its keep.</p><div class="chart-wrap"><div class="chart-tools"><button type="button" data-chart="eq" data-zoom="in" title="Zoom in">+</button><button type="button" data-chart="eq" data-zoom="out" title="Zoom out">−</button><button type="button" data-chart="eq" data-zoom="reset" title="Reset zoom">Reset</button><span class="chart-hint">Scroll or pinch to zoom · drag to pan</span></div><canvas id="eq" height="220"></canvas></div>
<p class="d" style="margin-top:14px">This second chart shows the <b>pain meter</b>: how far below its own record high each approach fell at every moment. Closer to 0% = less pain. Deep dips are the crashes; the green strategy's dips should be shallower.</p><div class="chart-wrap"><div class="chart-tools"><button type="button" data-chart="ddc" data-zoom="in" title="Zoom in">+</button><button type="button" data-chart="ddc" data-zoom="out" title="Zoom out">−</button><button type="button" data-chart="ddc" data-zoom="reset" title="Reset zoom">Reset</button><span class="chart-hint">Scroll or pinch to zoom · drag to pan</span></div><canvas id="ddc" height="150"></canvas></div>
<details style="margin-top:12px"><summary style="cursor:pointer;color:var(--acc);font-size:13px">How is this graph made?</summary><div style="font-size:13px;color:var(--mut);margin-top:8px;padding:10px 12px;background:rgba(0,0,0,.2);border-radius:8px;line-height:1.7">
1. Every day since 2016 we re-run the anomaly detector using <b>only information available up to that day</b> (no hindsight).<br>
2. When 2 of 3 detectors flag a day as abnormal, the strategy sells Boeing at the <b>next day's</b> price and sits in cash for 5 trading days, then buys back.<br>
3. We subtract realistic trading costs (0.05% per trade) from every buy/sell.<br>
4. Both lines compound the same $10,000 forward day by day; the drawdown chart is each line's distance below its own all-time high.<br>
Source data: daily Boeing prices from Yahoo Finance. Code: <b>src/strategy.py</b> in the GitHub repo.
</div></details></div>
<div class="sec" id="industry" style="display:none"><h2>Industry health check — how is Boeing's world doing?</h2><p class="d">Three real-world gauges of Boeing's business environment: are people flying? Is Boeing making money? Is the news good or bad? These give context to the alarms above (they're shown for information — our tests found they don't improve the math model itself).</p>
<div class="cards" id="opcards" style="display:none"></div>
<p class="d" id="demandlabel" style="display:none;margin-top:6px"><b>Are people flying?</b> Airport traffic vs the same day last year. Above 0% = more people flying than a year ago (good for airplane orders); below 0% = travel is shrinking (bad for Boeing's customers).</p>
<div class="chart-wrap" id="demandwrap" style="display:none"><div class="chart-tools"><button type="button" data-chart="demand" data-zoom="in" title="Zoom in">+</button><button type="button" data-chart="demand" data-zoom="out" title="Zoom out">−</button><button type="button" data-chart="demand" data-zoom="reset" title="Reset zoom">Reset</button><span class="chart-hint">Scroll or pinch to zoom · drag to pan</span></div><canvas id="demand" height="170"></canvas></div>
<p class="d" id="newslabel" style="display:none;margin-top:14px"><b>Is the news good or bad?</b> Each day's Boeing headlines are read by an AI and scored from −1 (very negative: crashes, probes, lawsuits) to +1 (very positive: orders, deliveries, upgrades). Sustained dips below zero mean a rough news cycle.</p>
<div class="chart-wrap" id="newswrap" style="display:none"><div class="chart-tools"><button type="button" data-chart="newsc" data-zoom="in" title="Zoom in">+</button><button type="button" data-chart="newsc" data-zoom="out" title="Zoom out">−</button><button type="button" data-chart="newsc" data-zoom="reset" title="Reset zoom">Reset</button><span class="chart-hint">Scroll or pinch to zoom · drag to pan</span></div><canvas id="newsc" height="150"></canvas></div>
<details style="margin-top:12px"><summary style="cursor:pointer;color:var(--acc);font-size:13px">How are these graphs made?</summary><div style="font-size:13px;color:var(--mut);margin-top:8px;padding:10px 12px;background:rgba(0,0,0,.2);border-radius:8px;line-height:1.7">
<b>Are people flying? (TSA demand)</b> — every day the TSA publishes how many passengers went through US airport security. We fetch that table (tsa.gov + a maintained mirror), then compare each day to the same day last year. Code: <b>src/demand.py</b>.<br>
<b>Boeing's finances (SEC cards above)</b> — revenue and operating cash flow come straight from Boeing's official quarterly filings via the SEC EDGAR API. Each number only appears on the dashboard from the date it was actually filed — no peeking at the future. Code: <b>src/edgar.py</b>.<br>
<b>News sentiment</b> — we pull Boeing headlines from free Yahoo Finance and Google News RSS feeds, then a finance-tuned AI language model (FinBERT) reads each headline and scores it from −1 (negative) to +1 (positive). Daily average is plotted. Code: <b>src/news.py</b>.
</div></details>
</div>
<div class="sec"><h2>Recent flags — why did the alarm ring?</h2><p class="d">The most recent anomaly flags and the primary reason each one fired.</p><table id="recent"></table></div>
<div class="sec"><h2>Strategy stats — the numbers behind the chart above</h2><table id="stats"></table><div class="foot" style="margin-top:8px">Research/education only. Not investment advice. No real trades.</div></div>
</div>
<script>
const D=/*DATA*/;
document.getElementById('sub').textContent=`As of ${D.as_of} · ${D.rows} trading days · detection + risk management`;
const t=D.today, ae=D.anom_eval, fm=D.flare_metrics, pa=D.precision_alerts;
const warn=t.votes>=2;
const cards=[
 {l:'Today',v:warn?'⚠ Anomaly':'Normal',n:warn?`${t.votes} of 3 detectors agree`:'behaving like its sector',w:warn,
  q:'Three independent detectors each vote on whether today is abnormal. 2+ votes = anomaly.'},
 {l:'Chance of abnormal move (5d)',v:t.flare_prob!=null?(Math.round(t.flare_prob*100)+'%'):'—',n:'model forecast, calibrated',
  q:'Probability Boeing has an unusually large move within the next 5 trading days. Calibrated = 30% means it happens ~30% of the time.'},
 {l:'Real events caught',v:`${ae.events_caught}/${ae.n_events_known}`,n:`better than luck (p=${ae.event_recall_vs_random_p})`,
  q:'Of the known real Boeing incidents in our test set, how many did the detector flag within a few days? The p-value shows this beats random flagging.'},
 {l:'Forecast skill (AUC)',v:fm.auc,n:'0.5 = guessing, 1.0 = perfect',
  q:'AUC measures how well the flare forecast separates calm periods from abnormal ones. Anything reliably above 0.5 has real signal.'},
 {l:'Alert precision (unseen data)',v:pa.oos_precision!=null?Math.round(pa.oos_precision*100)+'%':'—',n:`${pa.alerts_per_year} alerts/yr`,
  q:'Of the highest-confidence alerts on data the model never trained on, how many were followed by a real abnormal move?'},
 {l:'False-alarm control',v:D.aci_rate,n:'target ≈ 0.01 (1%)',
  q:'The alarm threshold self-adjusts so that only ~1% of normal days trigger a false alarm, even when volatility regimes change.'},
];

// ---- LIVE 'today' panel: poll /api/now (works under `python -m src.serve`); snapshot fallback ----
function fmtPct(x){return x==null?'—':Math.round(x*100)+'%';}
function renderLive(o,live){
  const sev=o.anomaly_votes!=null?o.anomaly_votes:(D.today.votes||0);
  const fp=o.flare_prob_5d!=null?o.flare_prob_5d:D.today.flare_prob;
  const vf=o.vol_forecast_10d_annualized!=null?o.vol_forecast_10d_annualized:null;
  const rz=o.resid_z_today!=null?o.resid_z_today:D.today.resid_z;
  const tag=live?`<span style="color:var(--grn)">● LIVE</span>`:`<span style="color:var(--mut)">snapshot</span>`;
  const warn=sev>=2;
  const story=warn
    ? `<b style="color:var(--red)">⚠ Anomaly detected:</b> Boeing is behaving highly unusually vs its peers today — it moved <b>${Math.abs(rz)}</b> standard deviations away from what the sector and market predict. <b>${sev} of 3</b> independent detectors agree.`
    : `<b style="color:var(--grn)">All clear:</b> Boeing is trading in line with the aerospace sector and the market (deviation ${rz}σ, below alarm level). <b>${sev} of 3</b> detectors see anything unusual.`;
  document.getElementById('liveinfo').innerHTML=
    `${tag} &nbsp; <b>${o.as_of||D.as_of}</b><br>${story}`+
    `<br><span style="color:var(--mut)">Chance of an abnormal move in the next 5 days: <b style="color:var(--ink)">${fmtPct(fp)}</b>`+
    (vf!=null?` · expected volatility (10d): <b style="color:var(--ink)">${(vf*100).toFixed(1)}%</b> annualized`:'')+`</span>`+
    (o.latency_ms!=null?` <span style="color:var(--mut)">(${o.latency_ms} ms)</span>`:'');
}
function refreshLive(){
  document.getElementById('liveinfo').innerHTML='<span style="color:var(--mut)">Loading live status…</span>';
  fetch('/api/now',{cache:'no-store'}).then(r=>r.json()).then(o=>{
    if(o.error){renderLive({},false);}else{renderLive(o,true);}
  }).catch(()=>renderLive({},false));
}
document.getElementById('refbtn').onclick=refreshLive;
refreshLive(); setInterval(refreshLive,30000);

document.getElementById('cards').innerHTML=cards.map(c=>`<div class="card" title="${c.q||''}"><div class="lab">${c.l} <span class="q">?</span></div><div class="val" style="color:${c.w?'var(--red)':'var(--ink)'}">${c.v}</div><div class="note">${c.n}</div></div>`).join('');

const CHARTS={};
const ZOOM_OPTS={zoom:{wheel:{enabled:true,speed:.1},pinch:{enabled:true},mode:'x'},pan:{enabled:true,mode:'x'},limits:{x:{min:'original',max:'original'}}};
const AX_X={ticks:{color:'#94a0bd',maxTicksLimit:14},grid:{color:'#26304f'}};
const AX_Y={ticks:{color:'#94a0bd'},grid:{color:'#26304f'}};

CHARTS.ts=new Chart(document.getElementById('ts'),{type:'line',data:{labels:D.dates,datasets:[
 {label:'BA',data:D.price,borderColor:'#4da3ff',borderWidth:1.1,pointRadius:0,tension:.1},
 {label:'Anomaly',type:'scatter',data:D.flagged,backgroundColor:'#ff5c6c',pointRadius:4,parsing:false},
 {label:'Event',type:'scatter',data:D.events,backgroundColor:'#ffb454',pointStyle:'triangle',pointRadius:6,parsing:false}]},
 options:{interaction:{mode:'nearest',intersect:true},plugins:{legend:{labels:{color:'#94a0bd'}},tooltip:{callbacks:{label:c=>{const r=c.raw;
  if(r.desc!==undefined)return ['📌 Real Boeing event: '+r.desc,'(documented incident, used to grade the detector)'];
  if(r.z!==undefined)return ['🚨 Anomaly flagged on '+r.x,`Boeing moved ${Math.abs(r.z)} standard deviations away from`, 'what the sector + market predicted that day', (Math.abs(r.z)>=3?'— an extreme idiosyncratic move.':'— unusual enough that 2+ detectors agreed.')];
  return 'BA $'+c.formattedValue;}}},zoom:ZOOM_OPTS},scales:{x:AX_X,y:AX_Y}}});

CHARTS.flare=new Chart(document.getElementById('flare'),{type:'line',data:{labels:D.dates,datasets:[
 {label:'Flare prob',data:D.flare_series,borderColor:'#ffb454',borderWidth:1,pointRadius:0,fill:true,backgroundColor:'rgba(255,180,84,.12)'}]},
 options:{plugins:{legend:{display:false},zoom:ZOOM_OPTS},scales:{x:AX_X,y:{...AX_Y,min:0,max:1}}}});

CHARTS.eq=new Chart(document.getElementById('eq'),{type:'line',data:{labels:D.dates,datasets:[
 {label:'buy & hold',data:D.eq_bh,borderColor:'#94a0bd',borderWidth:1.2,pointRadius:0},
 {label:'derisk_anomaly',data:D.eq_dr,borderColor:'#37d39b',borderWidth:1.5,pointRadius:0}]},
 options:{plugins:{legend:{labels:{color:'#94a0bd'}},zoom:ZOOM_OPTS},scales:{x:AX_X,y:AX_Y}}});

CHARTS.ddc=new Chart(document.getElementById('ddc'),{type:'line',data:{labels:D.dates,datasets:[
 {label:'DD buy&hold',data:D.dd_bh,borderColor:'#94a0bd',borderWidth:1,pointRadius:0,fill:true,backgroundColor:'rgba(148,160,189,.10)'},
 {label:'DD derisk',data:D.dd_dr,borderColor:'#37d39b',borderWidth:1,pointRadius:0,fill:true,backgroundColor:'rgba(55,211,155,.12)'}]},
 options:{plugins:{legend:{labels:{color:'#94a0bd'}},zoom:ZOOM_OPTS},scales:{x:AX_X,y:AX_Y}}});

// ---- industry signals (context only) ----
if(D.demand||D.operational||D.news){
 document.getElementById('industry').style.display='block';
 if(D.operational){
  const o=D.operational;
  const rvGood=o.revenue_yoy>=0;
  document.getElementById('opcards').style.display='grid';
  document.getElementById('opcards').innerHTML=[
   {l:'Is revenue growing?',v:(rvGood?'▲ +':'▼ ')+(o.revenue_yoy*100).toFixed(1)+'%',n:'vs same quarter last year',g:rvGood,
    q:'Boeing\'s sales compared to the same quarter a year ago, straight from its official SEC filing.'},
   {l:'Cash from operations',v:o.ocf_bn!=null?('$'+o.ocf_bn+'B'):'—',n:o.ocf_bn!=null&&o.ocf_bn<0?'burning cash this quarter':'generating cash',g:o.ocf_bn!=null&&o.ocf_bn>=0,
    q:'Cash Boeing\'s core business produced last quarter. Negative = spending more than it takes in.'},
   {l:'How fresh is this?',v:o.days_since!=null?o.days_since+' days':'—',n:'since Boeing\'s last SEC report',g:null,
    q:'Financials only update when Boeing files a quarterly report, roughly every ~90 days.'},
  ].map(c=>`<div class="card" title="${c.q||''}"><div class="lab">${c.l} <span class="q">?</span></div><div class="val" style="color:${c.g==null?'var(--ink)':(c.g?'var(--grn)':'var(--red)')}">${c.v}</div><div class="note">${c.n}</div></div>`).join('');
 }
 if(D.demand){
  document.getElementById('demandlabel').style.display='block';
  document.getElementById('demandwrap').style.display='block';
  CHARTS.demand=new Chart(document.getElementById('demand'),{type:'line',data:{labels:D.dates,datasets:[
   {label:'Air travel vs last year',data:D.demand.yoy,borderColor:'#4da3ff',borderWidth:1.1,pointRadius:0,fill:true,backgroundColor:'rgba(77,163,255,.10)'}]},
   options:{plugins:{legend:{display:false},zoom:ZOOM_OPTS,tooltip:{callbacks:{label:c=>{const v=c.raw;if(v==null)return '';return (v>=0?'+':'')+(v*100).toFixed(1)+'% passengers vs same day last year';}}}},scales:{x:AX_X,y:{...AX_Y,ticks:{...AX_Y.ticks,callback:v=>(v*100).toFixed(0)+'%'}}}}});
 }
 if(D.news){
  document.getElementById('newslabel').style.display='block';
  document.getElementById('newswrap').style.display='block';
  CHARTS.newsc=new Chart(document.getElementById('newsc'),{type:'line',data:{labels:D.dates,datasets:[
   {label:'News sentiment',data:D.news.sent,borderColor:'#ffb454',borderWidth:1.2,pointRadius:0,spanGaps:false}]},
   options:{plugins:{legend:{display:false},zoom:ZOOM_OPTS,tooltip:{callbacks:{label:c=>{const v=c.raw;if(v==null)return '';return 'Headline mood: '+(v>0.1?'positive':(v<-0.1?'negative':'neutral'))+' ('+v+')';}}}},scales:{x:AX_X,y:{...AX_Y,min:-1,max:1}}}});
 }
}

document.querySelectorAll('[data-zoom]').forEach(btn=>btn.addEventListener('click',()=>{
 const ch=CHARTS[btn.dataset.chart]; if(!ch) return;
 const z=btn.dataset.zoom;
 if(z==='in') ch.zoom(1.25); else if(z==='out') ch.zoom(0.8); else ch.resetZoom();
}));

// recent flags with plain-English driver
function driver(z){
 if(z==null)return 'multiple detectors agreed (volume/volatility pattern)';
 const a=Math.abs(z);
 const dir=z<0?'underperformed':'outperformed';
 if(a>=3)return `extreme move: ${dir} the aerospace sector by ${a}σ`;
 if(a>=2)return `sharp divergence: ${dir} the sector by ${a}σ`;
 return `unusual volume/volatility pattern (deviation ${z}σ)`;
}
const recent=(D.flagged||[]).slice(-10).reverse();
document.getElementById('recent').innerHTML='<tr><th>Date</th><th>BA price</th><th style="text-align:left">Primary driver</th></tr>'+
 (recent.length?recent.map(r=>`<tr><td>${r.x}</td><td>$${r.y}</td><td style="text-align:left">${driver(r.z)}</td></tr>`).join('')
  :'<tr><td colspan="3" style="color:var(--mut)">no flags in range</td></tr>');

const cols=["strategy","CAGR_%","sharpe","sortino","max_drawdown_%","calmar","final_$","pct_invested"];
const colNames={strategy:'Strategy','CAGR_%':'Yearly return %',sharpe:'Sharpe',sortino:'Sortino','max_drawdown_%':'Worst loss %',calmar:'Calmar','final_$':'Final $','pct_invested':'% time invested'};
const colHelp={strategy:'buy_hold = hold through everything. derisk_anomaly = step to cash after flags.','CAGR_%':'Average yearly growth rate.',sharpe:'Return per unit of risk. Higher is better; >1 is good.',sortino:'Like Sharpe but only penalizes downside swings.','max_drawdown_%':'Deepest fall from a previous peak — the pain metric.',calmar:'Yearly return divided by worst loss. Higher = better risk/reward.','final_$':'What $10k in 2016 became.','pct_invested':'Share of days the strategy held the stock.'};
document.getElementById('stats').innerHTML='<tr>'+cols.map(c=>`<th title="${colHelp[c]||''}">${colNames[c]||c} <span class="q">?</span></th>`).join('')+'</tr>'+
 D.stats.map(r=>'<tr>'+cols.map(c=>`<td>${r[c]}</td>`).join('')+'</tr>').join('');
</script></body></html>"""


if __name__ == "__main__":
    build()
