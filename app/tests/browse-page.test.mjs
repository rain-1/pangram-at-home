import {test} from 'node:test';import assert from 'node:assert/strict';
import {browsePage} from '../lib/browse-page.ts';import {browseParams} from '../lib/browse-api.ts';import {createBrowseHandler} from '../atlas-cloud/browse-worker.ts';
const papers=Array.from({length:47},(_,i)=>({id:i.toString(16).padStart(24,'0'),title:`Paper ${String(i).padStart(2,'0')}`,filename:`${i}.pdf`,collection:i===46?'iclr/2026':'iclr/2027',bytes:100,classified:false,keywords:i%2?['Robotics']:['Control'],primary_area:i%2?'Robotics':'Theory',updated:i}));
test('server pages preserve counts, filters, facets and navigation without transferring all records',()=>{
 let p=browsePage(papers,new URLSearchParams());assert.equal(p.items.length,20);assert.equal(p.total,46);assert.equal(p.pages,3);assert.equal(p.scopeTotal,46);
 p=browsePage(papers,new URLSearchParams({page:'2',paper:papers[43].id}));assert.equal(p.items.length,6);assert.equal(p.previous,papers[42].id);assert.equal(p.next,papers[44].id);assert.equal(p.position,43);
 p=browsePage(papers,new URLSearchParams({q:'paper',tag:'robotics',area:'Robotics',page:'900'}));assert.equal(p.total,23);assert.equal(p.page,1);assert.equal(p.items.length,3);
 p=browsePage(papers,new URLSearchParams({saved:'1',ids:papers[30].id}));assert.deepEqual(p.items.map(p=>p.id),[papers[30].id]);
 assert.equal(browsePage(papers,new URLSearchParams({collection:''})).total,47);
 assert.equal(browsePage(papers,new URLSearchParams({page:'Infinity'})).page,0);
});
function fixture(){
 const calls=[];let revision='a'.repeat(64);
 const bucket={get:async key=>{calls.push(key);if(key==='indexes/browse/current.json')return {json:async()=>({revision,index_key:`indexes/browse/${revision}/papers.json`,home_key:`indexes/browse/${revision}/home.json`})};if(key.endsWith('/home.json'))return {body:JSON.stringify({...browsePage(papers,new URLSearchParams()),revision})};if(key.endsWith('/papers.json'))return {json:async()=>({items:papers})};throw new Error('Unexpected bucket access');},list:()=>{throw new Error('Page load must never scan bucket');}};
 const handler=createBrowseHandler();return {calls,bucket,revision,next:()=>{revision='b'.repeat(64)},fetch:(q='',ctx)=>handler(new Request('https://example.test/backend/v1/browse'+q),bucket,ctx)};
}
test('cold homepage needs two small reads and never loads the full search index',async()=>{
 const f=fixture();const r=await f.fetch('?collection=iclr%2F2027&sort=title&page=0');assert.equal(r.status,200);assert.equal((await r.json()).items.length,20);assert.equal(f.calls.length,2);assert.ok(f.calls.every(k=>!k.endsWith('/papers.json')));
});
test('concurrent searches share index loading; saved filters are private and never cached publicly',async t=>{
 const f=fixture(),puts=[],store=new Map(),pending=[];
 Object.defineProperty(globalThis,'caches',{configurable:true,writable:true,value:undefined});t.after(()=>delete globalThis.caches);
 t.mock.property(globalThis,'caches',{default:{match:async req=>store.get(req.url)?.clone(),put:async(req,res)=>{puts.push(req.url);store.set(req.url,res);}}});
 const ctx={waitUntil:p=>pending.push(p)};
 await Promise.all([f.fetch('?q=paper',ctx),f.fetch('?tag=Control',ctx)]);await Promise.all(pending);
 assert.equal(f.calls.filter(k=>k.endsWith('/papers.json')).length,1);assert.equal(puts.length,2);
 const before=f.calls.length;const hit=await f.fetch('?q=paper',ctx);assert.equal(f.calls.length,before);assert.equal(hit.headers.get('Cache-Control'),'public, max-age=15');
 const saved=await f.fetch('?saved=1&ids='+papers[0].id,ctx);assert.equal(saved.headers.get('Cache-Control'),'private, no-store');assert.equal(puts.length,2);
});
test('new publication uses a different edge cache key and cannot reuse old homepage',async t=>{
 let now=100;t.mock.method(Date,'now',()=>now);const f=fixture();const before=await f.fetch();f.next();now+=16000;const after=await f.fetch();assert.notEqual(before.headers.get('X-Browse-Revision'),after.headers.get('X-Browse-Revision'));
});
test('browser request contains only active saved IDs when viewing the reading list',()=>{
 const filters={q:'',collection:'iclr/2027',area:'',tag:'',sort:'title',saved:false};assert.equal(browseParams(filters,0,'',new Set(['a'])).has('ids'),false);assert.equal(browseParams({...filters,saved:true},0,'',new Set([papers[0].id])).get('ids'),papers[0].id);
});
test('last successful preview is delivered before refresh and survives a failed refresh',async t=>{
 const {loadBrowse}=await import('../lib/browse-api.ts');const values=new Map(),params='collection=iclr%2F2027&sort=title&page=0';
 Object.defineProperty(globalThis,'localStorage',{configurable:true,value:{getItem:k=>values.get(k)||null,setItem:(k,v)=>values.set(k,v)}});t.after(()=>delete globalThis.localStorage);
 let step=0;t.mock.method(globalThis,'fetch',async()=>{if(++step===2)return new Response('Unavailable',{status:503});return Response.json(browsePage(papers,new URLSearchParams()));});
 const data=await loadBrowse(params,new AbortController().signal);let preview;
 await assert.rejects(loadBrowse(params,new AbortController().signal,p=>{preview=p;}));assert.deepEqual(preview,data);assert.equal(values.size,1);
});
