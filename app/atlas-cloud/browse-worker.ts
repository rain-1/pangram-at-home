import {browsePage,preparePapers,type BrowsePage,type PreparedPapers} from '../lib/browse-page.ts';
import type {AtlasPaper} from '../lib/atlas-collection.ts';
type BrowseManifest={revision:string;index_key:string;home_key:string};
/** Immutable publications let the default page bypass index parsing entirely. */
export function createBrowseHandler(){
 let manifest:{data:BrowseManifest;until:number}|undefined;
 let manifestLoading:Promise<BrowseManifest>|undefined;
 let parsed:{revision:string;papers:PreparedPapers}|undefined;
 let loading:Promise<typeof parsed>|undefined;
 async function current(bucket:R2Bucket){
  if(manifest&&manifest.until>Date.now())return manifest.data;
  if(manifestLoading)return manifestLoading;
  manifestLoading=(async()=>{const o=await bucket.get('indexes/browse/current.json');if(!o)throw new Error('Browse publication unavailable');const data=await o.json<BrowseManifest>();if(!/^[a-f0-9]{64}$/.test(data.revision)||data.index_key!==`indexes/browse/${data.revision}/papers.json`||data.home_key!==`indexes/browse/${data.revision}/home.json`)throw new Error('Invalid browse publication');manifest={data,until:Date.now()+15000};return data;})().finally(()=>{manifestLoading=undefined;});return manifestLoading;
 }
 return async function(request:Request,bucket:R2Bucket,ctx?:ExecutionContext){
  const url=new URL(request.url),params=url.searchParams,started=Date.now();
  const m=await current(bucket);
  const canonical=new URLSearchParams();
  for(const key of ['collection','q','area','tag','sort','page','paper','saved','ids']){const value=params.get(key);if(value)canonical.set(key,value);}
  // Empty collection means all collections and must remain distinct from the default.
  if(params.has('collection')&&params.get('collection')==='')canonical.set('collection','');
  canonical.sort();
  const isHome=(!params.has('collection')||params.get('collection')==='iclr/2027')&&(!params.get('sort')||params.get('sort')==='title')&&[...params.keys()].every(k=>['collection','sort','page'].includes(k))&&(!params.get('page')||params.get('page')==='0');
  const privateList=params.get('saved')==='1'||params.has('ids');
  const key=new Request(`${url.origin}/backend/v1/browse-cache/${m.revision}?${canonical}`);
  const cache=typeof caches!=='undefined'?(caches as CacheStorage & {default:Cache}).default:undefined;
  if(!privateList&&cache){const hit=await cache.match(key);if(hit){const response=new Response(request.method==='HEAD'?null:hit.body,hit);response.headers.set('Cache-Control','public, max-age=15');response.headers.set('Server-Timing',`edge;desc="edge cache hit";dur=${Date.now()-started}`);return response;}}
  let response:Response,timing='home';
  const headers={'Content-Type':'application/json','Cache-Control':privateList?'private, no-store':'public, max-age=15','X-Content-Type-Options':'nosniff','X-Browse-Revision':m.revision};
  if(isHome){const o=await bucket.get(m.home_key);if(!o)throw new Error('Missing homepage');response=new Response(o.body,{headers});}
  else{
   const cold=parsed?.revision!==m.revision,loadStart=Date.now();
   if(cold){
    // Share an index load rather than buffering a copy for every concurrent search.
    if(!loading)loading=(async()=>{const o=await bucket.get(m.index_key);if(!o)throw new Error('Missing browse index');const data=await o.json<{items:AtlasPaper[]}>();parsed={revision:m.revision,papers:preparePapers(data.items)};return parsed;})().finally(()=>{loading=undefined;});
    await loading;
    if(parsed?.revision!==m.revision)return new Response('Publication changed; retry',{status:503});
   }
   const computeStart=Date.now(),page:BrowsePage=browsePage(parsed!.papers,params);page.revision=m.revision;response=Response.json(page,{headers});
   timing=`${cold?`index;desc="index load";dur=${computeStart-loadStart}, `:''}compute;dur=${Date.now()-computeStart}`;
  }
  response.headers.set('Server-Timing',`${timing==='home'?'home;desc="home snapshot"':timing}, total;dur=${Date.now()-started}`);
  if(!privateList&&cache&&ctx){const copy=response.clone();copy.headers.delete('Server-Timing');copy.headers.set('Cache-Control','public, max-age=86400');ctx.waitUntil(cache.put(key,copy));}
  return request.method==='HEAD'?new Response(null,response):response;
 };
}
