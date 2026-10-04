import type {AtlasPaper} from './atlas-collection.ts';
import {filterDiscovery,relatedPapers,searchText,titleOrder,type BrowseFilters} from './paper-discovery.ts';
export type BrowsePage={items:AtlasPaper[];total:number;page:number;pages:number;scopeTotal:number;areas:[string,number][];topics:[string,number][];collections:string[];collectionCounts:[string,number][];selected?:AtlasPaper;position:number;previous?:string;next?:string;related:AtlasPaper[];revision?:string};
/** Per-publication work done once, so each request only filters and counts. */
export type PreparedPapers={items:AtlasPaper[];text:Map<string,string>;keys:Map<string,string[]>;byId:Map<string,AtlasPaper>};
export function preparePapers(papers:AtlasPaper[]):PreparedPapers{
 const items=[...papers].sort((a,b)=>titleOrder(a.title,b.title)||a.id.localeCompare(b.id));
 const text=new Map<string,string>(),keys=new Map<string,string[]>(),byId=new Map<string,AtlasPaper>();
 for(const p of items){text.set(p.id,searchText(p));keys.set(p.id,[...new Set((p.keywords||[]).map(k=>k.trim().toLowerCase()).filter(Boolean))]);if(!byId.has(p.id))byId.set(p.id,p);}
 return {items,text,keys,byId};
}
export function browsePage(source:AtlasPaper[]|PreparedPapers,params:URLSearchParams):BrowsePage{
 const prepared=Array.isArray(source)?undefined:source,papers=prepared?prepared.items:source as AtlasPaper[];
 const f:BrowseFilters={q:(params.get('q')||'').slice(0,300),collection:params.get('collection')??'iclr/2027',area:params.get('area')||'',tag:params.get('tag')||'',sort:params.get('sort')||'title',saved:params.get('saved')==='1'};
 const saved=new Set((params.get('ids')||'').split(',').filter(id=>/^[a-f0-9]{24}$/.test(id)));
 const results=filterDiscovery(papers,f,saved,prepared?.text,!!prepared),scope=papers.filter(p=>!f.collection||p.collection===f.collection);
 const collectionCounts=new Map<string,number>();for(const p of papers)collectionCounts.set(p.collection,(collectionCounts.get(p.collection)||0)+1);
 const areas=new Map<string,number>(),topics=new Map<string,number>();
 for(const p of scope){if(p.primary_area)areas.set(p.primary_area,(areas.get(p.primary_area)||0)+1);if(!f.area||p.primary_area===f.area)for(const k of prepared?.keys.get(p.id)??new Set((p.keywords||[]).map(k=>k.trim().toLowerCase()).filter(Boolean)))topics.set(k,(topics.get(k)||0)+1);}
 const pages=Math.max(1,Math.ceil(results.length/20)),raw=Number(params.get('page'));
 const page=Math.min(pages-1,Number.isSafeInteger(raw)&&raw>0?raw:0),id=params.get('paper'),selected= id?(prepared?prepared.byId.get(id):papers.find(p=>p.id===id)):undefined,position=results.findIndex(p=>p.id===id);
 return {items:results.slice(page*20,page*20+20),total:results.length,page,pages,scopeTotal:scope.length,areas:[...areas].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])),topics:[...topics].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).slice(0,12),collections:[...collectionCounts.keys()].sort().reverse(),collectionCounts:[...collectionCounts].sort((a,b)=>a[0].localeCompare(b[0])),selected,position,previous:position>0?results[position-1].id:undefined,next:position>=0&&position<results.length-1?results[position+1].id:undefined,related:selected?relatedPapers(selected,papers):[]};
}
