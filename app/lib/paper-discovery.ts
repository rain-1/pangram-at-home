import type {AtlasPaper} from './atlas-collection.ts';
export type BrowseFilters={q:string;collection:string;area:string;tag:string;saved:boolean;sort:string};
export function searchText(p:AtlasPaper){return [p.title,p.filename,p.forum_id,p.number,p.primary_area,p.tldr,p.abstract_preview,...(p.keywords||[])].join(' ').toLowerCase();}
/** Compares titles exactly as String#localeCompare does, without per-call locale setup. */
export const titleOrder=new Intl.Collator().compare;
/** `titleSorted` means `papers` is already in title order, so a title sort can be skipped. */
export function filterDiscovery(papers:AtlasPaper[],filters:BrowseFilters,saved:Set<string>,index?:Map<string,string>,titleSorted=false){
 const terms=filters.q.toLowerCase().trim().split(/\s+/).filter(Boolean);
 const matches=papers.filter(p=>(!filters.collection||p.collection===filters.collection)&&(!filters.area||p.primary_area===filters.area)&&(!filters.tag||p.keywords?.some(k=>k.toLowerCase()===filters.tag.toLowerCase()))&&(!filters.saved||saved.has(p.id))&&terms.every(t=>(index?.get(p.id)||searchText(p)).includes(t)));
 if(filters.sort==='title'&&titleSorted)return matches;
 return matches.sort((a,b)=>filters.sort==='title'?titleOrder(a.title,b.title)||a.id.localeCompare(b.id):filters.sort==='number'?(a.number??Infinity)-(b.number??Infinity)||a.id.localeCompare(b.id):(b.updated||0)-(a.updated||0)||a.title.localeCompare(b.title));
}
export function relatedPapers(paper:AtlasPaper,papers:AtlasPaper[]){
 const keys=new Set(paper.keywords?.map(k=>k.toLowerCase()));
 return papers.filter(p=>p.id!==paper.id&&p.collection===paper.collection).map(p=>({p,score:(p.primary_area&&p.primary_area===paper.primary_area?1:0)+(p.keywords||[]).filter(k=>keys.has(k.toLowerCase())).length*3})).filter(r=>r.score>0).sort((a,b)=>b.score-a.score||a.p.title.localeCompare(b.p.title)).slice(0,5).map(r=>r.p);
}
