import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createAtlasWorker,byteRange} from '../atlas-cloud/worker.ts';
const id='a'.repeat(24), pdfKey='papers/'+ 'b'.repeat(64)+'.pdf',detailKey='atlas-public/objects/'+'c'.repeat(64)+'.json.gz';
function fixture(mutate=x=>x){
 const calls=[];
 const cat=mutate({items:[{id,title:'A paper',filename:'a.pdf',collection:'iclr/2025',bytes:100,classified:true,models:['v8'],pdf_key:pdfKey,detail_key:detailKey}],models:[],published_at:123});
 const env={ASSETS:{fetch:async()=>new Response('asset')},BUCKET:{
   list:async()=>({objects:[],truncated:false}),
   head:async key=>{calls.push(['head',key]);return {size:100,httpEtag:'"obj"'};},
   get:async(key,options)=>{if(key.startsWith('indexes/'))return null;calls.push(['get',key,options]);if(key==='atlas-public/catalogue.json')return cat===null?null:{body:'catalogue',httpEtag:'"cat"',json:async()=>cat};return {body:'bytes'};}
 }};
 const worker=createAtlasWorker();
 const fetch=(path='',init={})=>worker.fetch(new Request('https://example.test/backend/v1/pdf-reader'+path,init),env);
 return {fetch,calls,env};
}
test('publication allowlist rejects unknown IDs, object traversal and write methods without D1 fallback',async()=>{
 const {fetch,calls}=fixture();
 for(const path of ['/'+ 'd'.repeat(24),'/__proto__','/%2e%2e/secret','/'+id+'/file/extra'])assert.equal((await fetch(path)).status,404);
 for(const method of ['POST','PUT','DELETE','PATCH','OPTIONS'])assert.equal((await fetch('/'+id,{method})).status,405);
 assert.equal(calls.filter(c=>c[0]==='get').length,1);
});
test('missing or unsafe publication manifest fails closed and does not expose keys',async()=>{
 for(const mutate of [()=>null,x=>({...x,items:[{...x.items[0],detail_key:'private/secret'}]}),x=>({...x,items:[x.items[0],x.items[0]]})]){
  const {fetch,calls}=fixture(mutate);const r=await fetch('/'+id);assert.equal(r.status,503);assert.equal(calls.length,1);assert.doesNotMatch(await r.text(),/secret|stack/);
 }
});
test('concurrent catalogue requests share one read; conditional GET and HEAD have no body',async()=>{
 const {fetch,calls}=fixture();
 const responses=await Promise.all(Array.from({length:50},()=>fetch()));
 assert.equal(calls.length,1);const data=await responses[0].json();assert.equal(data.items[0].pdf_key,undefined);assert.equal(data.items[0].detail_key,undefined);
 const cached=await fetch('',{headers:{'If-None-Match':'"other", W/'+responses[0].headers.get('etag')}});assert.equal(cached.status,304);assert.equal(await cached.text(),'');
 const head=await fetch('',{method:'HEAD'});assert.equal(await head.text(),'');assert.equal(head.headers.get('etag'),responses[0].headers.get('etag'));
 const filtered=await fetch('?q='+id);assert.equal((await filtered.json()).items.length,1);assert.equal(filtered.headers.get('etag'),null);
 assert.equal((await (await fetch('?model=v5')).json()).items.length,0);
});
test('PDF ranges, If-Range, validators, HEAD and response security are consistent',async()=>{
 const {fetch,calls}=fixture();
 let r=await fetch('/'+id+'/file',{headers:{Range:'bytes=10-19'}});assert.equal(r.status,206);assert.equal(r.headers.get('content-range'),'bytes 10-19/100');assert.deepEqual(calls.at(-1)[2],{range:{offset:10,length:10}});
 r=await fetch('/'+id+'/file',{headers:{Range:'bytes=10-19','If-Range':'"stale"'}});assert.equal(r.status,200);assert.equal(r.headers.get('content-length'),'100');
 for(const range of ['bytes=100-','bytes=-0','bytes=9-1','bytes=0-1,4-5','bytes=9007199254740993-','bytes=-9007199254740993']){
  r=await fetch('/'+id+'/file',{headers:{Range:range}});assert.equal(r.status,416);assert.equal(r.headers.get('content-range'),'bytes */100');assert.equal(r.headers.get('x-content-type-options'),'nosniff');
 }
 const before=calls.filter(c=>c[0]==='get').length;
 r=await fetch('/'+id+'/file',{method:'HEAD',headers:{Range:'bytes=0-1'}});assert.equal(r.status,200);assert.equal(r.headers.get('content-length'),'100');assert.equal(await r.text(),'');
 r=await fetch('/'+id,{headers:{'If-None-Match':'W/"obj"'}});assert.equal(r.status,304);
 assert.equal(calls.filter(c=>c[0]==='get').length,before);
 r=await fetch('/'+id);assert.equal(r.headers.get('content-encoding'),'gzip');assert.equal(r.headers.get('x-frame-options'),'DENY');assert.match(r.headers.get('content-security-policy'),/sandbox/);assert.equal(r.headers.get('access-control-allow-origin'),null);
});
test('storage failures return sanitized retryable errors, never raw internal exceptions',async()=>{
 const {fetch,env}=fixture();env.BUCKET.get=async()=>{throw new Error('secret internal path');};const r=await fetch();assert.equal(r.status,503);assert.doesNotMatch(await r.text(),/secret/);assert.equal(r.headers.get('cache-control'),'no-store');
});
test('10,000 generated ranges never escape object bounds',()=>{
 let seed=113;const rnd=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed;};
 for(let n=0;n<10000;n++){
  const size=rnd()%100000+1,a=rnd()%200000,b=rnd()%200000;
  const range=byteRange(n%2?`bytes=${a}-${b}`:`bytes=-${a}`,size);
  if(range){assert.ok(Number.isSafeInteger(range.offset)&&range.offset>=0);assert.ok(range.length>0);assert.ok(range.offset+range.length<=size);}
 }
});
test('expired catalogue revalidation reuses parsed data when R2 reports unchanged',async t=>{
 let now=1000;t.mock.method(Date,'now',()=>now);
 const {fetch,env,calls}=fixture();await fetch();const previous=env.BUCKET.get;
 env.BUCKET.get=async(key,options)=>{if(key==='atlas-public/catalogue.json'&&options?.onlyIf){calls.push(['conditional',options]);return {httpEtag:'"cat"'};}return previous(key,options);};
 now+=61000;const response=await fetch();assert.equal(response.status,200);assert.equal((await response.json()).items[0].id,id);assert.deepEqual(calls.at(-1),['conditional',{onlyIf:{etagDoesNotMatch:'cat'}}]);
});
test('new PDFs appear without reports; archive refresh changes ETag even with unchanged results',async t=>{
 let now=1000;t.mock.method(Date,'now',()=>now);
 const {fetch,env}=fixture();const before=await fetch();
 const hash='e'.repeat(64),key=`papers/${hash}.pdf`;
 env.BUCKET.list=async()=>({objects:[{key,size:123,customMetadata:{title:'New submission',collection:'iclr/2027'}}],truncated:false});
 now+=61000;
 const after=await fetch('',{headers:{'If-None-Match':before.headers.get('etag')}});
 assert.equal(after.status,200);assert.notEqual(after.headers.get('etag'),before.headers.get('etag'));
 const rows=(await after.json()).items;assert.equal(rows.length,2);assert.equal(rows[1].collection,'iclr/2027');assert.equal(rows[1].classified,false);
 const detail=await (await fetch('/'+hash.slice(0,24))).json();assert.deepEqual(detail,{version:hash,reports:[]});
 assert.equal((await fetch('/'+hash.slice(0,24)+'/file')).status,200);
 assert.equal((await (await fetch('?collection=iclr/2027')).json()).items.length,1);
 assert.equal((await (await fetch('?model=v8')).json()).items.length,1);
});
test('archive discovery follows pagination, uses index metadata and excludes non-PDF keys',async()=>{
 const {fetch,env}=fixture();const original=env.BUCKET.get;const hash='f'.repeat(64),key=`papers/${hash}.pdf`;const cursors=[];
 env.BUCKET.list=async options=>{if(options.prefix!=='papers/f')return {objects:[],truncated:false};cursors.push(options.cursor);return options.cursor?{objects:[{key,size:25}],truncated:false}:{objects:[{key:'private/test.pdf',size:20}],truncated:true,cursor:'next'};};
 env.BUCKET.get=async(k,o)=>k==='indexes/downloaded-papers.json'?{json:async()=>({papers:[{key,filename:'forum.pdf',conference:'iclr',year:2027,title:'Indexed title'}]})}:original(k,o);
 const items=(await (await fetch()).json()).items;assert.deepEqual(cursors,[undefined,'next']);assert.equal(items.length,2);assert.equal(items[1].title,'Indexed title');assert.equal(items[1].collection,'iclr/2027');assert.equal(items[1].pdf_key,undefined);
});
test('paper metadata is scoped to discovered public paper IDs and keeps abstract out of catalogue',async()=>{
 const {fetch,env}=fixture();const original=env.BUCKET.get;
 env.BUCKET.get=async(k,o)=>k==='indexes/iclr2027-pdfs.json'?{json:async()=>({papers:[{key:pdfKey,forum_id:'forum',primary_area:'Robotics',keywords:['Control']}]})}:k.startsWith('indexes/iclr2027-details/')?{json:async()=>({['b'.repeat(24)]:{abstract:'Full abstract',bibtex:'Citation'}})}:original(k,o);
 const paper=(await (await fetch()).json()).items[0];assert.equal(paper.primary_area,'Robotics');assert.equal(paper.abstract,undefined);assert.equal(paper.classified,true);
 assert.equal((await (await fetch('/'+id+'/metadata')).json()).abstract,'Full abstract');
 assert.equal((await fetch('/'+'f'.repeat(24)+'/metadata')).status,404);
});
test('cold reader resolves one public hash prefix without loading the catalogue inventory',async()=>{
 const {fetch,env}=fixture();const original=env.BUCKET.get,hash='e'.repeat(64),key=`papers/${hash}.pdf`,prefixes=[];
 env.BUCKET.list=async options=>{prefixes.push(options.prefix);assert.equal(options.prefix,`papers/${hash.slice(0,24)}`);return {objects:[{key,size:123}],truncated:false};};
 env.BUCKET.get=async(k,o)=>{if(k==='indexes/paper-details/ee.json')return null;if(k==='indexes/iclr2027-details/ee.json')return {json:async()=>({[hash.slice(0,24)]:{abstract:'Saved abstract'}})};assert.ok(!k.startsWith('indexes/'));return original(k,o);};
 assert.deepEqual(await (await fetch('/'+hash.slice(0,24))).json(),{version:hash,reports:[]});
 assert.equal((await (await fetch('/'+hash.slice(0,24)+'/metadata')).json()).abstract,'Saved abstract');
 assert.equal((await fetch('/'+hash.slice(0,24)+'/file')).status,200);assert.equal(prefixes.length,3);
});

test('general metadata supports older collections and takes precedence over legacy shards',async()=>{
 const {fetch,env}=fixture();const original=env.BUCKET.get;
 env.BUCKET.get=async(k,o)=>k.startsWith('indexes/paper-details/')?{json:async()=>({['b'.repeat(24)]:{abstract:'Older collection abstract',venue:'COLM'}})}:original(k,o);
 const detail=await (await fetch('/'+id+'/metadata')).json();assert.equal(detail.abstract,'Older collection abstract');assert.equal(detail.venue,'COLM');
});
test('fresh isolates serve the catalogue from the edge copy and refresh it in the background when stale',async t=>{
 const store=new Map(),pending=[];
 Object.defineProperty(globalThis,'caches',{configurable:true,writable:true,value:undefined});t.after(()=>delete globalThis.caches);
 t.mock.property(globalThis,'caches',{default:{match:async req=>store.get(req.url)?.clone(),put:async(req,res)=>{store.set(req.url,res);}}});
 const ctx={waitUntil:p=>pending.push(p)};
 const first=fixture();const built=await createAtlasWorker().fetch(new Request('https://example.test/backend/v1/pdf-reader'),first.env,ctx);
 await Promise.all(pending);pending.length=0;assert.equal(store.size,1);
 const etag=built.headers.get('etag'),body=await built.text();
 const second=fixture(),worker=createAtlasWorker(),get=(init={})=>worker.fetch(new Request('https://example.test/backend/v1/pdf-reader',init),second.env,ctx);
 const fast=await get();assert.equal(await fast.text(),body);assert.equal(fast.headers.get('etag'),etag);assert.equal(second.calls.length,0);assert.equal(pending.length,0);
 assert.equal((await get({headers:{'If-None-Match':etag}})).status,304);
 const [key,res]=[...store][0];const old=new Headers(res.headers);old.set('X-Built-At','1');store.set(key,new Response(body,{headers:old}));
 assert.equal((await get()).status,200);assert.equal(pending.length,1);await Promise.all(pending);assert.equal(second.calls.filter(c=>c[1]==='atlas-public/catalogue.json').length,1);
 assert.notEqual(store.get(key).headers.get('X-Built-At'),'1');
});
