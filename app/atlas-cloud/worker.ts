import legacyAliases from "../lib/legacy-paper-aliases.json" with {type:"json"};
import {mergeArchive, type PublishedPaper} from "../lib/atlas-archive.ts";
import {createBrowseHandler} from "./browse-worker.ts";
interface Env { ASSETS: Fetcher; BUCKET: R2Bucket }
type Catalogue = {items:PublishedPaper[];models:unknown[];published_at:number};
function publicPaper(p:PublishedPaper){return {id:p.id,title:p.title,filename:p.filename,collection:p.collection,bytes:p.bytes,classified:p.classified,models:p.models,score_summaries:p.score_summaries,forum_id:p.forum_id,number:p.number,keywords:p.keywords,primary_area:p.primary_area,tldr:p.tldr,abstract_preview:p.abstract_preview,published:p.published,updated:p.updated};}
const security = {
  "X-Content-Type-Options":"nosniff", "X-Frame-Options":"DENY",
  "Referrer-Policy":"no-referrer", "Cross-Origin-Resource-Policy":"same-origin",
  "Permissions-Policy":"camera=(), microphone=(), geolocation=()",
  "Content-Security-Policy":"default-src 'none'; frame-ancestors 'none'; sandbox",
};
function headers(extra:HeadersInit={}) {const result=new Headers(security);new Headers(extra).forEach((v,k)=>result.set(k,v));return result;}
function json(value:unknown,status=200) {return Response.json(value,{status,headers:headers({"Cache-Control":"no-store"})});}
function matchesEtag(value:string|null,etag:string) {return !!value && value.split(',').some(v=>v.trim()==='*'||v.trim().replace(/^W\//,'')===etag.replace(/^W\//,''));}
/** Single byte ranges only. Reject unsafe integers and empty suffix ranges. */
export function byteRange(value:string,size:number):{offset:number;length:number}|null {
  const parts=/^bytes=(\d*)-(\d*)$/.exec(value);
  if(!parts||(!parts[1]&&!parts[2])||size<=0)return null;
  const first=parts[1]?Number(parts[1]):undefined, last=parts[2]?Number(parts[2]):undefined;
  if((first!==undefined&&!Number.isSafeInteger(first))||(last!==undefined&&!Number.isSafeInteger(last)))return null;
  if(first===undefined){if(!last)return null;const length=Math.min(last,size);return {offset:size-length,length};}
  const end=last===undefined?size-1:Math.min(last,size-1);
  return first>=size||first>end?null:{offset:first,length:end-first+1};
}
/** Factory keeps caches isolated in tests; production has one instance per isolate. */
export function createAtlasWorker() {
  const browse=createBrowseHandler();
  let cached: {value:Catalogue;byId:Map<string,PublishedPaper>;body:string;models:unknown[];etag:string;sourceEtag:string;until:number;builtAt:number;stored?:boolean}|undefined;
  let refreshing=false;
  let loading:Promise<typeof cached>|undefined;
  async function catalogue(env:Env) {
    if(cached&&cached.until>Date.now())return cached;
    if(loading)return loading;
    loading=(async()=>{
      const object=await env.BUCKET.get('atlas-public/catalogue.json',cached?{onlyIf:{etagDoesNotMatch:cached.sourceEtag.replace(/^"|"$/g,'')}}:undefined);
      if(!object)return undefined;
      const value:Catalogue="body" in object?await object.json<Catalogue>():cached!.value;
      if(!Array.isArray(value.items))throw new Error('Invalid catalogue');
      // Discover uploaded PDFs independently of extraction or classification publication.
      const objects:{key:string;size:number;customMetadata?:Record<string,string>}[]=[];
      const prefixes=Array.from('0123456789abcdef',c=>'papers/'+c);
      // Bound concurrent storage operations while avoiding a long serial full-bucket scan.
      await Promise.all(Array.from({length:4},async()=>{
        while(prefixes.length){
          const prefix=prefixes.shift()!;
          let cursor:string|undefined;
          do {
            const page=await env.BUCKET.list({prefix,limit:1000,cursor,include:['customMetadata']});
            objects.push(...page.objects.map(o=>({key:o.key,size:o.size,customMetadata:o.customMetadata})));
            cursor=page.truncated?page.cursor:undefined;
          } while(cursor);
        }
      }));
      objects.sort((a,b)=>a.key.localeCompare(b.key));
      const index=await env.BUCKET.get('indexes/downloaded-papers.json');
      const archive=index?await index.json<{papers:Parameters<typeof mergeArchive>[2]}>():{papers:[]};
      const submissions=await env.BUCKET.get('indexes/iclr2027-pdfs.json');
      const submissionIndex=submissions?await submissions.json<{papers:Parameters<typeof mergeArchive>[2]}>():{papers:[]};
      const merged={...value,items:mergeArchive(value.items,objects,[...archive.papers,...submissionIndex.papers])};
      merged.models=(value.models as {total:number;available:number;complete:boolean}[]).map(m=>({...m,total:merged.items.length,complete:m.available===merged.items.length}));
      const byId=new Map<string,PublishedPaper>();
      const publicItems=merged.items.map(p=>{
        if(!/^[a-f0-9]{24}$/.test(p.id)||byId.has(p.id)||!/^papers\/[a-f0-9]{64}\.pdf$/.test(p.pdf_key)||(p.detail_key!==undefined&&!/^atlas-public\/objects\/[a-f0-9]{64}\.json\.gz$/.test(p.detail_key))||(p.classified&&!p.detail_key))throw new Error('Invalid published object');
        byId.set(p.id,p);
        return publicPaper(p);
      });
      const body=JSON.stringify({items:publicItems,models:merged.models,published_at:value.published_at});
      const signature=[object.httpEtag,index?.httpEtag,submissions?.httpEtag,...objects.map(o=>`${o.key}:${o.size}`)].join('|');
      const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(signature));
      const etag='"'+Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('')+'"';
      // Keep only the source manifest for revalidation; inventory metadata can change independently.
      cached={value,byId,body,models:merged.models,etag,sourceEtag:object.httpEtag,until:Date.now()+60000,builtAt:Date.now()};
      return cached;
    })().finally(()=>{loading=undefined;});
    return loading;
  }
  let readerSource:{items:Map<string,PublishedPaper>;until:number}|undefined;
  let readerLoading:Promise<Map<string,PublishedPaper>>|undefined;
  async function readerPaper(env:Env,id:string){
    const known=cached&&cached.until>Date.now()?cached.byId.get(id):undefined;
    if(known)return known;
    if(!readerSource||readerSource.until<=Date.now()){
      if(!readerLoading)readerLoading=(async()=>{
        const object=await env.BUCKET.get('atlas-public/catalogue.json');
        if(!object)throw new Error('Missing publication');
        const value=await object.json<Catalogue>(),items=new Map<string,PublishedPaper>();
        if(!Array.isArray(value.items))throw new Error('Invalid publication');
        for(const p of value.items){
          if(!/^[a-f0-9]{24}$/.test(p.id)||items.has(p.id)||!/^papers\/[a-f0-9]{64}\.pdf$/.test(p.pdf_key)||(p.detail_key!==undefined&&!/^atlas-public\/objects\/[a-f0-9]{64}\.json\.gz$/.test(p.detail_key))||(p.classified&&!p.detail_key))throw new Error('Invalid published object');
          items.set(p.id,p);
        }
        readerSource={items,until:Date.now()+60000};return items;
      })().finally(()=>{readerLoading=undefined;});
      await readerLoading;
    }
    const result=readerSource!.items.get(id);if(result)return result;
    // Unclassified IDs are public PDF hash prefixes; never scan the full archive.
    const found:{key:string;size:number}[]=[];let cursor:string|undefined;
    do{const page=await env.BUCKET.list({prefix:`papers/${id}`,limit:1000,cursor});
      found.push(...page.objects.filter(o=>/^papers\/[a-f0-9]{64}\.pdf$/.test(o.key)&&o.key.slice(7,31)===id));
      cursor=page.truncated?page.cursor:undefined;
    }while(cursor);
    if(found.length>1)throw new Error('PDF identifier collision');
    const item=found[0];return item?{id,pdf_key:item.key,classified:false,models:[],title:id,filename:id+'.pdf',collection:'Other uploads',bytes:item.size} as PublishedPaper:undefined;
  }
  async function serve(request:Request,env:Env,ctx?:ExecutionContext):Promise<Response> {
    const url=new URL(request.url),head=request.method==='HEAD';
    if(url.pathname==='/backend/v1/browse'&&['GET','HEAD'].includes(request.method))return browse(request,env.BUCKET,ctx);
    if(url.pathname==='/private'||url.pathname.startsWith('/private/'))return json({detail:'Not found'},404);
    if(!['GET','HEAD'].includes(request.method))return new Response(null,{status:405,headers:headers({Allow:'GET, HEAD','Cache-Control':'no-store'})});
    if(url.pathname==='/backend/v1/discovery'){
      const object=await env.BUCKET.get('indexes/iclr2027-discovery.json');
      if(!object)return json({papers:[]});
      return new Response(head?null:object.body,{headers:headers({'Content-Type':'application/json','Cache-Control':'public, max-age=60','ETag':object.httpEtag})});
    }
    const list=url.pathname==='/backend/v1/pdf-reader';
    const match=url.pathname.match(/^\/backend\/v1\/pdf-reader\/([a-f0-9]{24})(\/file|\/metadata)?$/);
    if(!list&&!match){
      if(url.pathname.startsWith('/backend/')||url.pathname==='/workbench')return json({detail:'Not found'},404);
      return env.ASSETS.fetch(request);
    }
    // Building the catalogue lists the whole archive (~20s cold), so every colo keeps the last
    // build in the edge cache, serves it immediately, and rebuilds in the background when stale.
    const unfiltered=list&&!['q','collection','model'].some(k=>url.searchParams.get(k));
    const edge=typeof caches!=='undefined'?(caches as CacheStorage&{default:Cache}).default:undefined;
    const edgeKey=new Request(`${url.origin}/backend/v1/catalogue-cache/v1`);
    const storeEdge=(c:NonNullable<typeof cached>)=>edge!.put(edgeKey,new Response(c.body,{headers:{'Content-Type':'application/json','ETag':c.etag,'X-Built-At':String(c.builtAt),'Cache-Control':'public, max-age=604800'}}));
    if(unfiltered&&edge&&ctx&&!(cached&&cached.until>Date.now())){
      const started=Date.now(),hit=await edge.match(edgeKey);
      if(hit){
        const etag=hit.headers.get('ETag')||'';
        if(Date.now()-Number(hit.headers.get('X-Built-At')||0)>120000&&!refreshing){refreshing=true;ctx.waitUntil(catalogue(env).then(c=>{if(c){c.stored=true;return storeEdge(c);}}).catch(()=>{}).finally(()=>{refreshing=false;}));}
        const h=headers({'Content-Type':'application/json','Cache-Control':'public, max-age=60','ETag':etag,'Server-Timing':`edge;desc="catalogue edge copy";dur=${Date.now()-started}`});
        if(matchesEtag(request.headers.get('if-none-match'),etag))return new Response(null,{status:304,headers:h});
        return new Response(head?null:hit.body,{headers:h});
      }
    }
    const buildStart=Date.now();
    const published=list?await catalogue(env):undefined;
    if(unfiltered&&published&&edge&&ctx&&!published.stored){published.stored=true;ctx.waitUntil(storeEdge(published).catch(()=>{published.stored=false;}));}
    // Only published results and the dedicated public PDF archive are exposed.
    if(list&&!published)return json({detail:'The collection is temporarily unavailable.'},503);
    if(list&&published){
      const q=(url.searchParams.get('q')||'').slice(0,200).toLowerCase(),collection=url.searchParams.get('collection'),model=url.searchParams.get('model');
      const filtered=!!(q||collection||model);
      const h=headers({'Content-Type':'application/json','Cache-Control':'public, max-age=60','Server-Timing':`catalogue;dur=${Date.now()-buildStart}`});
      if(!filtered){h.set('ETag',published.etag);if(matchesEtag(request.headers.get('if-none-match'),published.etag))return new Response(null,{status:304,headers:h});}
      if(head)return new Response(null,{headers:h});
      const body=filtered?JSON.stringify({models:published.models,published_at:published.value.published_at,items:[...published.byId.values()].filter(p=>(!q||`${p.id} ${p.title} ${p.filename}`.toLowerCase().includes(q))&&(!collection||p.collection===collection)&&(!model||p.models.includes(model))).map(publicPaper)}):published.body;
      return new Response(body,{headers:h});
    }
    const id=(legacyAliases as Record<string,string>)[match![1]]||match![1];
    const paper=await readerPaper(env,id);
    if(!paper)return json({detail:'Paper not found'},404);
    if(match![2]==='/metadata'){
      const hash=paper.pdf_key.slice(7,-4);
      for(const prefix of ['indexes/paper-details','indexes/iclr2027-details']){
        const shard=await env.BUCKET.get(`${prefix}/${hash.slice(0,2)}.json`);
        const data=shard?await shard.json<Record<string,unknown>>():{};
        if(data[hash.slice(0,24)])return json(data[hash.slice(0,24)]);
      }
      return json({});
    }
    const file=match![2]==='/file',key=file?paper.pdf_key:paper.detail_key;
    if(!file&&!key)return json({version:paper.pdf_key.slice(7,-4),reports:[]});
    const meta=await env.BUCKET.head(key!);
    if(!meta)return json({detail:file?'PDF not found':'Results not found'},404);
    const h=headers({'Content-Type':file?'application/pdf':'application/json','ETag':meta.httpEtag,'Cache-Control':file?'public, max-age=86400':'public, max-age=60'});
    if(file){h.set('Accept-Ranges','bytes');h.set('Content-Disposition',`inline; filename="${id}.pdf"`);}else h.set('Content-Encoding','gzip');
    if(matchesEtag(request.headers.get('if-none-match'),meta.httpEtag))return new Response(null,{status:304,headers:h});
    let range={offset:0,length:meta.size},status=200;
    const requested=request.headers.get('range');
    if(file&&!head&&requested&&(!request.headers.has('if-range')||request.headers.get('if-range')===meta.httpEtag)){
      const parsed=byteRange(requested,meta.size);
      if(!parsed){h.set('Content-Range',`bytes */${meta.size}`);h.set('Cache-Control','no-store');return new Response(null,{status:416,headers:h});}
      range=parsed;status=206;h.set('Content-Range',`bytes ${range.offset}-${range.offset+range.length-1}/${meta.size}`);
    }
    h.set('Content-Length',String(range.length));
    if(head)return new Response(null,{headers:h});
    // Content-addressed keys are immutable, so metadata and this body cannot race versions.
    const object=await env.BUCKET.get(key!,file?{range}:undefined);
    if(!object)return json({detail:'Object not found'},404);
    return new Response(object.body,{status,headers:h,...(!file?{encodeBody:'manual' as const}:{})});
  }
  return {async fetch(request:Request,env:Env,ctx?:ExecutionContext):Promise<Response>{
    try {const response=await serve(request,env,ctx);return request.method==='HEAD'?new Response(null,response):response;}
    catch(error) {console.error("Catalogue failure", error instanceof Error?error.message:"Storage failure");const response=json({detail:'The collection is temporarily unavailable. Please retry.'},503);return request.method==='HEAD'?new Response(null,response):response;}
  }};
}
export default createAtlasWorker() satisfies ExportedHandler<Env>;
