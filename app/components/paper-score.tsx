import {paperRate, summaryFor, modelLabel, type ScoredPaper} from '@/lib/paper-scores';
export default function PaperScore({paper,model,threshold}:{paper:ScoredPaper & {classified:boolean};model:string;threshold:number}) {
 const rate=paperRate(summaryFor(paper,model),threshold);
 const version=model==='all'?(paper.score_summaries?.v8?'v8':'v5'):model;
 if(rate===null)return <span className="atlas-score-empty">{paper.classified?'Score unavailable':'Not classified'}</span>;
 const band=rate<=.1?'low':rate<=.5?'medium':'high';
 return <span className={`atlas-score-card ${band}`} aria-label={`${(rate*100).toFixed(1)}% of passages above ${threshold.toFixed(2)}, ${modelLabel(version)}`}>
   <span className="atlas-score-top"><strong>{(rate*100).toFixed(1)}<span>%</span></strong><span className="atlas-score-model">{modelLabel(version)}</span></span>
   <span className="atlas-score-track" aria-hidden="true"><span style={{width:`${rate*100}%`}}/></span>
   <span className="atlas-score-caption">passages above {threshold.toFixed(2)}</span>
 </span>;
}
