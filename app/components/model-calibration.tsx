"use client";
import {useRef,useState} from "react";
import {SiteHeader,SiteFooter,useSavedCount} from "./site-chrome";
import calibration from "@/lib/model-calibration/qwen35-4b-fast10.json";
import "./model-calibration.css";

type Curve={name:string;auc:number;n_ai:number;n_human:number;tpr_at_1pct:number;fpr:number[];tpr:number[]};
const SERIES=["var(--cal-s1)","var(--cal-s2)","var(--cal-s3)"];
const pct=(v:number,d=1)=>`${(v*100).toFixed(v>0&&v<0.001?3:v<0.01?2:d)}%`;
const at=(c:Curve,f:number)=>{let j=0;while(j+1<c.fpr.length&&c.fpr[j+1]<=f)j++;return c.tpr[j];};

/** ROC curves on a log false-positive axis, with markers at the published thresholds. */
function RocChart({curves,label,marks}:{curves:Curve[];label:string;marks:number[]}){
 const W=760,H=380,m={l:56,r:16,t:12,b:44},xmin=1e-4;
 const x=(f:number)=>m.l+(Math.log10(Math.max(f,xmin))+4)/4*(W-m.l-m.r), y=(t:number)=>m.t+(1-t)*(H-m.t-m.b);
 const svg=useRef<SVGSVGElement>(null);
 const [hover,setHover]=useState<{px:number;f:number}|null>(null);
 const move=(e:React.PointerEvent)=>{const r=svg.current!.getBoundingClientRect(),px=(e.clientX-r.left)*W/r.width;
  if(px<m.l||px>W-m.r){setHover(null);return;}setHover({px,f:Math.pow(10,(px-m.l)/(W-m.l-m.r)*4-4)});};
 return <figure className="cal-figure">
  <div className="cal-legend">{curves.map((c,i)=><span key={c.name}><i style={{background:SERIES[i]}}/>{c.name} <span className="site-num">AUROC {c.auc.toFixed(3)}</span></span>)}</div>
  <div className="cal-plot">
   <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} onPointerMove={move} onPointerLeave={()=>setHover(null)}>
    {[1e-4,1e-3,1e-2,1e-1,1].map(f=><g key={f}><line x1={x(f)} x2={x(f)} y1={m.t} y2={H-m.b} className="cal-grid"/><text x={x(f)} y={H-m.b+18} textAnchor="middle">{pct(f,0)}</text></g>)}
    {[0,.25,.5,.75,1].map(t=><g key={t}><line x1={m.l} x2={W-m.r} y1={y(t)} y2={y(t)} className="cal-grid"/><text x={m.l-8} y={y(t)+4} textAnchor="end">{t*100}%</text></g>)}
    <line x1={m.l} x2={W-m.r} y1={y(0)} y2={y(0)} className="cal-axis"/>
    {marks.map(f=><g key={f}><line x1={x(f)} x2={x(f)} y1={m.t} y2={H-m.b} className="cal-mark"/><text x={x(f)+5} y={m.t+12} className="cal-mark-label">{pct(f)} FPR</text></g>)}
    <text x={(W+m.l)/2} y={H-6} textAnchor="middle">False-positive rate on human papers (log scale)</text>
    <text x={14} y={(H-m.b+m.t)/2} textAnchor="middle" transform={`rotate(-90 14 ${(H-m.b+m.t)/2})`}>AI text caught</text>
    {curves.map((c,i)=><path key={c.name} fill="none" stroke={SERIES[i]} strokeWidth={2} strokeLinejoin="round"
      d={c.fpr.map((f,j)=>[f,c.tpr[j]] as const).filter(([f])=>f>=xmin).map(([f,t],j)=>`${j?"L":"M"}${x(f).toFixed(1)} ${y(t).toFixed(1)}`).join(" ")}/>)}
    {curves.map((c,i)=><circle key={c.name} cx={x(.01)} cy={y(c.tpr_at_1pct)} r={4.5} fill={SERIES[i]} className="cal-dot"/>)}
    {hover&&<g><line x1={hover.px} x2={hover.px} y1={m.t} y2={H-m.b} className="cal-axis"/>
     {curves.map((c,i)=><circle key={c.name} cx={x(Math.max(c.fpr[c.fpr.findLastIndex(f=>f<=hover.f)]??xmin,xmin))} cy={y(at(c,hover.f))} r={4} fill={SERIES[i]} className="cal-dot"/>)}</g>}
    <rect x={m.l} y={m.t} width={W-m.l-m.r} height={H-m.t-m.b} fill="transparent"/>
   </svg>
   {hover&&<div className="cal-tip" style={{left:`${Math.min(hover.px/W*100,70)}%`}}>
    <b className="site-num">FPR ≤ {pct(hover.f,2)}</b>
    {curves.map((c,i)=><span key={c.name}><i style={{background:SERIES[i]}}/>{c.name}: <span className="site-num">{pct(at(c,hover.f))}</span></span>)}
   </div>}
  </div>
  <div className="cal-table-wrap"><table className="cal-table">
   <thead><tr><th>Set</th><th>AI items</th><th>Human items</th><th>AUROC</th><th>Caught at 0.1%</th><th>at 1%</th><th>at 5%</th></tr></thead>
   <tbody>{curves.map(c=><tr key={c.name}><td>{c.name}</td><td>{c.n_ai.toLocaleString()}</td><td>{c.n_human.toLocaleString()}</td><td>{c.auc.toFixed(4)}</td><td>{pct(at(c,.001))}</td><td>{pct(c.tpr_at_1pct)}</td><td>{pct(at(c,.05))}</td></tr>)}</tbody>
  </table></div>
 </figure>;
}

/** Median share of sentences flagged per paper (bar) with the interquartile range (whisker). */
function YearChart(){
 const rows=calibration.years;
 const W=760,rowH=30,m={l:250,r:60,t:8,b:36},H=m.t+m.b+rows.length*rowH;
 const x=(v:number)=>m.l+v*(W-m.l-m.r);
 return <figure className="cal-figure">
  <div className="cal-plot"><svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Share of sentences flagged per paper, by year">
   {[0,.25,.5,.75,1].map(v=><g key={v}><line x1={x(v)} x2={x(v)} y1={m.t} y2={H-m.b} className="cal-grid"/><text x={x(v)} y={H-m.b+18} textAnchor="middle">{v*100}%</text></g>)}
   <text x={(W+m.l)/2} y={H-4} textAnchor="middle">Share of the paper’s sentences flagged at 1% FPR (median, with middle half of papers)</text>
   {rows.map((r,i)=>{const cy=m.t+i*rowH+rowH/2,cls=r.kind==="human"?"cal-bar human":r.kind==="iclr"?"cal-bar iclr":"cal-bar";
    return <g key={r.label}>
     <text x={m.l-10} y={cy+4} textAnchor="end" className="cal-row-label">{r.label}</text>
     <rect x={x(0)} y={cy-7} width={Math.max(x(r.flagged_share_median)-x(0),2)} height={14} rx={3} className={cls}/>
     <line x1={x(r.flagged_share_p25)} x2={x(r.flagged_share_p75)} y1={cy} y2={cy} className="cal-whisker"/>
     <text x={x(Math.max(r.flagged_share_p75,r.flagged_share_median))+6} y={cy+4} className="cal-value">{pct(r.flagged_share_median)}</text>
    </g>;})}
  </svg></div>
  <div className="cal-table-wrap"><table className="cal-table">
   <thead><tr><th>Group</th><th>Papers</th><th>Median flagged</th><th>Middle half</th><th>Above human 99th percentile</th><th>Median document score</th></tr></thead>
   <tbody>{rows.map(r=><tr key={r.label}><td>{r.label}</td><td>{r.papers.toLocaleString()}</td><td>{pct(r.flagged_share_median)}</td><td>{pct(r.flagged_share_p25)}–{pct(r.flagged_share_p75)}</td><td>{pct(r.over_human_p99)}</td><td>{r.document_score_median.toFixed(3)}</td></tr>)}</tbody>
  </table></div>
 </figure>;
}

export default function ModelCalibration(){
 const savedCount=useSavedCount();
 const c=calibration,one=c.sentence_curves[0];
 return <div className="site cal">
  <SiteHeader page="calibration" savedCount={savedCount}/>
  <main className="cal-main">
   <header className="cal-head">
    <p className="site-kicker">Model calibration</p>
    <h1>{c.model.name}</h1>
    <p>{c.model.detail} Sentence scores are the model’s raw probabilities. A sentence is flagged when its score is above the threshold that flags 1% of sentences in human-written papers.</p>
   </header>

   <section className="cal-facts" aria-label="Key figures">
    <div><b className="site-num">{c.thresholds.find(t=>t.target_fpr===0.01)!.threshold.toFixed(4)}</b><span>Sentence threshold at 1% FPR</span></div>
    <div><b className="site-num">{pct(c.thresholds.find(t=>t.target_fpr===0.01)!.heldout_fpr,2)}</b><span>False-positive rate on held-out human papers</span></div>
    <div><b className="site-num">{pct(one.tpr_at_1pct)}</b><span>Single-sentence AI edits caught at 1% FPR</span></div>
    <div><b className="site-num">{c.calibration_set.papers}</b><span>Human papers behind the thresholds</span></div>
   </section>

   <section className="cal-section">
    <h2>Thresholds</h2>
    <p>{c.calibration_set.description} {c.calibration_set.method}</p>
    <div className="cal-table-wrap"><table className="cal-table">
     <thead><tr><th>Target false-positive rate</th><th>Sentence threshold</th><th>Held-out false-positive rate</th><th>95% interval</th><th>Status</th></tr></thead>
     <tbody>{c.thresholds.map(t=><tr key={t.target_fpr}><td>{pct(t.target_fpr)}</td><td>{t.threshold.toFixed(4)}</td><td>{pct(t.heldout_fpr,2)}</td><td>{pct(t.heldout_ci95[0],2)}–{pct(t.heldout_ci95[1],2)}</td>
      <td><span className={t.stable?"cal-pill ok":"cal-pill warn"}>{t.stable?"Confirmed":"Unstable"}</span></td></tr>)}</tbody>
    </table></div>
   </section>

   <section className="cal-section">
    <h2>Sentence level: AI edits inside human papers</h2>
    <p>Edited sentences from the frozen evaluation suite, scored in their paper context, against every sentence of the human calibration papers. Dashed lines mark the published thresholds.</p>
    <RocChart curves={c.sentence_curves as Curve[]} label="Sentence-level ROC curves" marks={[.001,.01,.05]}/>
   </section>

   <section className="cal-section">
    <h2>Document level: AI manuscripts against human papers</h2>
    <p>120 AI-involved manuscripts against the {c.calibration_set.papers} human papers. {c.document_threshold.note} At this size the document-level false-positive rate is known only to lie between {pct(c.document_threshold.human_fpr_ci95[0])} and {pct(c.document_threshold.human_fpr_ci95[1])}.</p>
    <RocChart curves={c.document_curves as Curve[]} label="Document-level ROC curves" marks={[.01]}/>
   </section>

   <section className="cal-section">
    <h2>Flagged share by year</h2>
    <p>The same threshold applied to papers from each year. Papers that share text with the model’s training data are excluded. Scores stay at the human baseline through 2024 and rise from 2025.</p>
    <YearChart/>
   </section>

   <section className="cal-section">
    <h2>Limits</h2>
    <ul className="cal-limits">{c.caveats.map(t=><li key={t}>{t}</li>)}</ul>
   </section>
  </main>
  <SiteFooter/>
 </div>;
}
