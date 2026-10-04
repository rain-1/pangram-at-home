"use client";
import {lazy,Suspense,useDeferredValue,useEffect,useMemo,useRef,useState,type CSSProperties,type ReactNode} from 'react';
import {Bookmark,ArrowLeft,ArrowRight,Search,ExternalLink,BookOpen,X,FileText,Link2,Quote,TriangleAlert} from 'lucide-react';
import {loadBrowse,browseParams} from '../lib/browse-api';
import type {BrowsePage} from '../lib/browse-page';
import type {AtlasPaper} from '../lib/atlas-collection';
import {type BrowseFilters} from '../lib/paper-discovery';
import ReaderErrorBoundary from './reader-error-boundary';
import {SiteHeader,SiteFooter} from './site-chrome';
import './paper-browser.css';
function MissingTldr(){return <span className="pb-no-tldr" tabIndex={0} aria-label="no tldr found"><TriangleAlert size={14} aria-hidden/><span className="pb-no-tldr-tip" aria-hidden>no tldr found</span></span>;}
const PDFReader=lazy(()=>import('./pdf-reader'));
const defaults:BrowseFilters={q:'',collection:'iclr/2027',area:'',tag:'',saved:false,sort:'title'};
const collectionName=(s:string)=>s.replace('/', ' ').toUpperCase();
async function api(path:string,init?:RequestInit){const r=await fetch('/backend'+path,init);if(!r.ok)throw new Error('Unable to load this paper. Please try again.');return r.json();}
type Detail={abstract?:string;bibtex?:string;license?:string;venue?:string};
// Shared across views: hovering a row starts these loads, so opening a paper rarely waits.
const details=new Map<string,Promise<Detail>>();
function loadDetail(id:string){let p=details.get(id);if(!p){p=api(`/v1/pdf-reader/${id}/metadata`) as Promise<Detail>;details.set(id,p);p.catch(()=>details.delete(id));}return p;}
let readerChunk:Promise<unknown>|undefined;
const warmReader=()=>readerChunk??=import('./pdf-reader').catch(()=>{readerChunk=undefined;});
export default function PaperBrowser(){
 const [data,setData]=useState<BrowsePage|null>(null),[busy,setBusy]=useState(false),[initialized,setInitialized]=useState(false);
 const [loaded,setLoaded]=useState(false),[error,setError]=useState(''),[attempt,setAttempt]=useState(0);
 const [filters,setFilters]=useState<BrowseFilters>(defaults),[selected,setSelected]=useState(''),[page,setPage]=useState(0);
 const [saved,setSaved]=useState<Set<string>>(new Set()),[storageNotice,setStorageNotice]=useState('');
 const [detail,setDetail]=useState<Detail|null>(null),[detailError,setDetailError]=useState(''),[detailAttempt,setDetailAttempt]=useState(0),[copied,setCopied]=useState('');
 const deferred=useDeferredValue(filters),typing=useRef(false),[searchQ,setSearchQ]=useState('');
 // Typing waits for a short pause; any other change to the query applies immediately.
 useEffect(()=>{if(!typing.current){setSearchQ(deferred.q);return;}typing.current=false;const t=setTimeout(()=>setSearchQ(deferred.q),220);return()=>clearTimeout(t);},[deferred.q]);
 const query=useMemo(()=>({...deferred,q:searchQ}),[deferred,searchQ]);
 useEffect(()=>{const sync=()=>{const p=new URLSearchParams(location.search);setFilters({q:p.get('q')||'',collection:p.get('collection')??'iclr/2027',area:p.get('area')||'',tag:p.get('tag')||'',saved:p.get('saved')==='1',sort:p.get('sort')||'title'});setSelected(p.get('paper')||'');setPage(Math.max(0,Number(p.get('page'))||0));};sync();window.addEventListener('popstate',sync);try{const s=JSON.parse(localStorage.getItem('atlas-reading-list')||'[]');if(Array.isArray(s))setSaved(new Set(s.filter(v=>typeof v==='string')));}catch{setStorageNotice('Reading list storage is unavailable in this browser. Saves will last for this visit.');}setInitialized(true);return()=>window.removeEventListener('popstate',sync);},[]);
 const queryParams=useMemo(()=>browseParams(query,page,selected,saved).toString(),[query,page,selected,saved]);
 const prefetched=useRef(new Set<string>());
 function prefetch(id:string){warmReader();loadDetail(id).catch(()=>{});const url=browseParams(query,data?.page||0,id,saved).toString();if(query.saved||prefetched.current.has(url))return;prefetched.current.add(url);fetch('/backend/v1/browse?'+url,{priority:'low'} as RequestInit).catch(()=>{});}
 useEffect(()=>{if(!initialized)return;const abort=new AbortController();let active=true;setBusy(true);const timer=setTimeout(()=>abort.abort(),30000);loadBrowse(queryParams,abort.signal,preview=>{if(active){setData(preview);setLoaded(true);}}).then(next=>{if(active){setData(next);setLoaded(true);setError('');}}).catch(()=>{if(active)setError('Could not load the library. Please retry.');}).finally(()=>{clearTimeout(timer);if(active)setBusy(false);});return()=>{active=false;abort.abort();clearTimeout(timer);};},[queryParams,attempt,initialized]);
 useEffect(()=>{const refresh=setInterval(()=>{if(document.visibilityState==='visible')setAttempt(a=>a+1);},60000);return()=>clearInterval(refresh);},[]);
 useEffect(()=>{setDetail(null);setDetailError('');setCopied('');if(!selected)return;let live=true;if(detailAttempt)details.delete(selected);loadDetail(selected).then(data=>{if(live)setDetail(data);}).catch(()=>{if(live)setDetailError('The full abstract could not be loaded.');});return()=>{live=false;};},[selected,detailAttempt]);
 function navigate(next:BrowseFilters,id='',n=0,replace=false){setFilters(next);setSelected(id);setPage(n);const p=new URLSearchParams();p.set('collection',next.collection);for(const k of ['q','area','tag','sort'] as const)if(next[k])p.set(k,next[k]);if(next.saved)p.set('saved','1');if(id)p.set('paper',id);if(n)p.set('page',String(n));history[replace?'replaceState':'pushState']({},'',`/?${p}`);if(!replace)window.scrollTo(0,0);}
 function change(patch:Partial<BrowseFilters>){navigate({...filters,...patch});}
 function toggle(id:string){const next=new Set(saved);if(next.has(id))next.delete(id);else next.add(id);setSaved(next);try{localStorage.setItem('atlas-reading-list',JSON.stringify([...next]));}catch{setStorageNotice('Could not save to this browser. Your reading list will last for this visit.');}}
 const wide=typeof window==='undefined'||window.matchMedia('(min-width: 901px)').matches;
 // While reading, warm the next paper so stepping through results is instant.
 useEffect(()=>{if(selected&&data?.next)prefetch(data.next);},[selected,data?.next]);// eslint-disable-line react-hooks/exhaustive-deps
 const searchInput=useRef<HTMLInputElement>(null);
 useEffect(()=>{const onKey=(e:KeyboardEvent)=>{const t=e.target as HTMLElement|null;if(e.key!=='/'||e.metaKey||e.ctrlKey||e.altKey||t?.closest('input,textarea,select,[contenteditable="true"]'))return;if(!searchInput.current)return;e.preventDefault();searchInput.current.focus();searchInput.current.select();};window.addEventListener('keydown',onKey);return()=>window.removeEventListener('keydown',onKey);},[]);
 const results=data?.items||[],areas=data?.areas||[],topics=data?.topics||[];
 const active=data?.selected?.id===selected?data.selected:undefined,position=data?.position??-1,lastPage=(data?.pages||1)-1,currentPage=data?.page||0;
 const related=data?.related||[],papers=active?[...results.filter(p=>p.id!==active.id),active]:results;
 const total=data?.total||0,scopeTotal=data?.scopeTotal||0,largestArea=areas[0]?.[1]||1;
 const terms=useMemo(()=>[...new Set(searchQ.toLowerCase().split(/\s+/).filter(t=>t.length>1))].sort((a,b)=>b.length-a.length),[searchQ]);
 const mark=(text:string)=>highlight(text,terms);
 async function copy(value:string,label:string){try{await navigator.clipboard.writeText(value);setCopied(label);}catch{setCopied('Copy unavailable. Select and copy the text below.');}}
 function saveButton(p:AtlasPaper,label=true){const on=saved.has(p.id);return <button className={`pb-save${on?' is-saved':''}`} aria-label={`${on?'Remove from':'Add to'} reading list: ${p.title}`} aria-pressed={on} onClick={()=>toggle(p.id)}><Bookmark size={16}/>{label&&<span>{on?'Saved':'Save'}</span>}</button>;}
 const paperHref=(id:string)=>`/?${new URLSearchParams({...Object.fromEntries(new URLSearchParams(location.search)),paper:id})}`;
 const openreview=(p:AtlasPaper)=>p.forum_id?`https://openreview.net/forum?id=${encodeURIComponent(p.forum_id)}`:'';
 const heading=filters.saved?'Your reading list':filters.collection?`${collectionName(filters.collection)} ${filters.collection.startsWith('iclr/')?'submissions':'papers'}`:'All papers';
 return <div className="site pb">
 <SiteHeader page={filters.saved?'saved':'discover'} savedCount={saved.size} discover={{href:'/',onClick:()=>navigate({...defaults})}} saved={{href:'/?saved=1',onClick:()=>navigate({...filters,saved:true},'')}}/>
 {storageNotice&&<p role="status" className="pb-notice">{storageNotice}</p>}{error&&<p role="alert" className="pb-notice">{error} <button onClick={()=>setAttempt(a=>a+1)}>Retry</button></p>}
 {selected?<main className="pb-reader">
  <aside className="pb-dossier" aria-label="Paper details">
   <div className="pb-dossier-nav"><button onClick={()=>navigate(filters,'',page)}><ArrowLeft size={15}/> All results</button><div className="pb-stepper"><button aria-label="Previous paper" disabled={busy||!data?.previous} onClick={()=>navigate(filters,data!.previous!,page)}><ArrowLeft size={15}/></button><span>{position>=0?<><b>{(position+1).toLocaleString()}</b> / {total.toLocaleString()}</>:'Paper'}</span><button aria-label="Next paper" disabled={busy||!data?.next} onClick={()=>navigate(filters,data!.next!,page)}><ArrowRight size={15}/></button></div></div>
   {(!loaded||busy&&!active)?<DossierSkeleton/>:!active?<div className="pb-empty"><h3>This paper isn’t in the uploaded collection.</h3><p>It may have been withdrawn or not uploaded yet.</p><button className="site-primary" onClick={()=>navigate(filters)}>Browse papers</button></div>:<>
    <p className="site-kicker">{collectionName(active.collection)}{active.number?<> · <span className="site-num">#{active.number}</span></>:null}</p>
    <h1>{tidy(active.title)}</h1>
    <p className="pb-status">{detail?.venue||'Research paper'}{active.collection==='iclr/2027'?' · Anonymous · Under review':''}</p>
    <div className="pb-actions">{saveButton(active)}<a className="site-primary" href={`/backend/v1/pdf-reader/${active.id}/file`} target="_blank" rel="noreferrer"><FileText size={15}/> Open PDF</a>{active.forum_id&&<a href={openreview(active)} target="_blank" rel="noreferrer">OpenReview <ExternalLink size={13}/></a>}</div>
    {active.tldr?.trim()?<p className="pb-tldr"><span>TL;DR</span>{active.tldr}</p>:<p className="pb-tldr"><span>TL;DR</span><MissingTldr/></p>}
    <details className="pb-fold" open={wide}><summary>Abstract</summary><p className="pb-abstract">{detail?.abstract||active.abstract_preview||'No abstract available.'}{!detail&&active.abstract_preview?'…':''}</p>{detailError&&<p role="status" className="pb-muted">{detailError} <button className="site-link" onClick={()=>setDetailAttempt(a=>a+1)}>Retry</button></p>}</details>
    {(active.primary_area||!!active.keywords?.length)&&<div className="pb-fold-static"><h2>Filed under</h2><div className="pb-chips">{active.primary_area&&<button className="pb-chip is-area" onClick={()=>navigate({...filters,area:active.primary_area!,tag:''})}>{active.primary_area}</button>}{active.keywords?.map(k=><button className="pb-chip" key={k} onClick={()=>navigate({...filters,tag:k})}>{k}</button>)}</div></div>}
    <div className="pb-tools"><button onClick={()=>copy(location.href,'Link copied')}><Link2 size={14}/> Copy link</button><a href={`/?view=classifications&paper=${active.id}`}>View classifications <ArrowRight size={13}/></a></div>
    {detail?.bibtex&&<details className="pb-fold"><summary><Quote size={13}/> Cite this submission</summary><pre>{detail.bibtex}</pre><button className="site-link" onClick={()=>copy(detail.bibtex!,'Citation copied')}>Copy BibTeX</button></details>}
    <p role="status" className="pb-copied">{copied}</p>
    {related.length>0&&<section className="pb-related"><h2>Keep exploring</h2><p className="pb-muted">Shares keywords or a research area with this paper.</p><ol>{related.map(p=><li key={p.id}><a href={paperHref(p.id)} onClick={e=>{e.preventDefault();navigate(filters,p.id,page);}}>{p.number&&<span className="site-num">#{p.number}</span>}<span>{tidy(p.title)}</span></a></li>)}</ol></section>}
   </>}
  </aside>
  <section className="pb-sheet" aria-label="PDF">{active&&<ReaderErrorBoundary key={selected} onBack={()=>navigate(filters)}><Suspense fallback={<p className="pb-sheet-wait">Opening PDF…</p>}><PDFReader availablePapers={papers} api={api} initialPaperId={selected} embedded hosted readingOnly/></Suspense></ReaderErrorBoundary>}</section>
 </main>:<main className="pb-main">
  <section className="pb-hero">
   <p className="site-kicker">{filters.saved?`${saved.size} saved in this browser`:loaded?`${scopeTotal.toLocaleString()} papers · public OpenReview metadata`:'Opening the library…'}</p>
   <h1>{heading}</h1>
   <p className="pb-lede">{filters.saved?'Papers you saved in this browser. No account needed.':'Search, filter by research area, and read the PDF.'}</p>
   <div className="pb-search"><Search size={20} aria-hidden/><input ref={searchInput} type="search" aria-label="Search papers" placeholder="Search titles, keywords, TL;DRs or submission numbers" value={filters.q} onChange={e=>{typing.current=true;navigate({...filters,q:e.target.value},'',0,true);}}/>{filters.q?<button aria-label="Clear search" onClick={()=>change({q:''})}><X size={18}/></button>:<kbd aria-hidden>/</kbd>}</div>
   {topics.length>0&&<div className="pb-topics"><span>{filters.area?'Common in this area':'Popular keywords'}</span>{topics.map(([tag,count])=><button aria-pressed={filters.tag===tag} className={filters.tag===tag?'is-on':''} key={tag} onClick={()=>change({tag:filters.tag===tag?'':tag})}>{tag}<small>{count.toLocaleString()}</small></button>)}</div>}
  </section>
  <div className="pb-layout">
   <aside className="pb-sidebar" aria-label="Filters">
    <label className="pb-field">Collection<select aria-label="Collection" value={filters.collection} onChange={e=>change({collection:e.target.value,area:'',tag:''})}><option value="">All collections</option>{(data?.collections||[]).map(c=><option key={c} value={c}>{collectionName(c)} ({(data?.collectionCounts?.find(([name])=>name===c)?.[1]||0).toLocaleString()})</option>)}</select></label>
    <label className="pb-field pb-mobile-area">Research area<select aria-label="Research area" value={filters.area} onChange={e=>change({area:e.target.value,tag:''})}><option value="">All research areas</option>{areas.map(([area,count])=><option key={area} value={area}>{area} ({count.toLocaleString()})</option>)}</select></label>
    <h2>Research areas</h2>
    <div className="pb-areas">
     <button aria-pressed={!filters.area} className={!filters.area?'is-on':''} onClick={()=>change({area:'',tag:''})}><span>All areas</span><b>{scopeTotal.toLocaleString()}</b></button>
     {areas.map(([area,count])=><button aria-pressed={filters.area===area} className={filters.area===area?'is-on':''} key={area} onClick={()=>change({area,tag:''})} style={{'--share':`${Math.max(2,count/largestArea*100)}%`} as CSSProperties}><span>{area}</span><b>{count.toLocaleString()}</b><i aria-hidden/></button>)}
    </div>
    {!areas.length&&loaded&&<p className="pb-muted">No research areas are available for this collection.</p>}
    <p className="pb-sidebar-note"><BookOpen size={15}/> Your reading list stays in this browser.</p>
   </aside>
   <section className="pb-results" aria-busy={busy} aria-label="Results">
    <div className="pb-results-bar"><h2>{filters.saved?'Saved':'Papers'} <span className="site-num">{loaded?total.toLocaleString():'—'}</span></h2>
     {(filters.area||filters.tag||filters.saved||filters.q)&&<div className="pb-active">{filters.q&&<button onClick={()=>change({q:''})}>“{filters.q}” <X size={12}/></button>}{filters.area&&<button onClick={()=>change({area:''})}>{filters.area} <X size={12}/></button>}{filters.tag&&<button onClick={()=>change({tag:''})}>{filters.tag} <X size={12}/></button>}{filters.saved&&<button onClick={()=>change({saved:false})}>Saved only <X size={12}/></button>}<button className="site-link" onClick={()=>navigate({...defaults,collection:filters.collection})}>Clear all</button></div>}
     <label className="pb-sort">Sort<select aria-label="Sort papers" value={filters.sort} onChange={e=>change({sort:e.target.value})}><option value="title">Title A–Z</option><option value="updated">Recently updated</option><option value="number">Submission number</option></select></label>
     {busy&&loaded&&<span className="pb-progress" role="status" aria-label="Updating results"/>}
    </div>
    {!loaded?<ol className="pb-list" aria-hidden>{Array.from({length:6},(_,i)=><li key={i} className="pb-row is-ghost"><span className="pb-gutter"/><div><i/><i/><i/></div></li>)}</ol>:!total?<div className="pb-empty"><BookOpen size={28}/><h3>{filters.saved?(saved.size?'No saved papers match these filters':'Nothing saved yet'):'No papers match'}</h3><p>{filters.saved?'Use Save on any paper to keep it here.':'Try fewer words, or another research area.'}</p><button className="site-primary" onClick={()=>navigate({...defaults,collection:filters.saved?filters.collection:''})}>{filters.saved?'Browse papers':'Search all collections'}</button></div>:
    <ol className="pb-list" start={currentPage*20+1}>{results.map(p=><li key={p.id} className="pb-row" onPointerEnter={()=>prefetch(p.id)} onTouchStart={()=>prefetch(p.id)} onFocus={()=>prefetch(p.id)}>
     <span className="pb-gutter" aria-label={p.number?`Submission ${p.number}`:undefined}>{p.number?String(p.number).padStart(5,'0'):'—'}</span>
     <div className="pb-row-body">
      <h3><a href={paperHref(p.id)} onClick={e=>{e.preventDefault();navigate(filters,p.id,currentPage);}}>{mark(tidy(p.title))}</a>{!p.tldr?.trim()&&<> <MissingTldr/></>}</h3>
      <p className="pb-summary">{mark(p.tldr?.trim()||p.abstract_preview||'Open the PDF to read this paper.')}{!p.tldr?.trim()&&p.abstract_preview?'…':''}</p>
      <div className="pb-row-meta">{p.primary_area&&<button className="pb-area" onClick={()=>change({area:p.primary_area!,tag:''})}>{p.primary_area}</button>}{p.keywords?.slice(0,4).map(k=><button key={k} className="pb-kw" onClick={()=>change({tag:k})}>{mark(k)}</button>)}</div>
     </div>
     <div className="pb-row-side">{saveButton(p,false)}{p.forum_id&&<a href={openreview(p)} target="_blank" rel="noreferrer" aria-label={`OpenReview forum: ${p.title}`}>OpenReview <ExternalLink size={12}/></a>}<span>{(p.bytes/1e6).toFixed(1)} MB{p.classified?' · scored':''}</span></div>
    </li>)}</ol>}
    {total>0&&lastPage>0&&<nav className="pb-pagination" aria-label="Pages"><button disabled={!currentPage} onClick={()=>navigate(filters,'',currentPage-1)}><ArrowLeft size={15}/> Previous</button><span>Page <b>{(currentPage+1).toLocaleString()}</b> of {(lastPage+1).toLocaleString()}</span><button disabled={currentPage>=lastPage} onClick={()=>navigate(filters,'',currentPage+1)}>Next <ArrowRight size={15}/></button></nav>}
   </section>
  </div>
 </main>}
 {!selected&&<SiteFooter/>}
 </div>;
}
function DossierSkeleton(){return <div className="pb-ghost" aria-hidden><i/><i/><i/><i/><i/></div>;}
/** OpenReview titles often carry LaTeX or Markdown markup; show the words, not the markup. */
function tidy(title:string){return title.replace(/\\(?:textsc|texttt|textit|textbf|emph|mathcal|mathbf|mathrm|text)\s*\{([^{}]*)\}/g,'$1').replace(/\*\*([^*]+)\*\*/g,'$1').replace(/\$([^$]*)\$/g,'$1').replace(/\\\(|\\\)/g,'').replace(/\\([a-zA-Z]+)/g,(m,name)=>GREEK[name]??m).replace(/\{([^{}]*)\}/g,'$1').replace(/\^(\d+)/g,(_,n:string)=>[...n].map(d=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]).join('')).replace(/\s+/g,' ').trim();}
const GREEK:Record<string,string>={alpha:'α',beta:'β',gamma:'γ',delta:'δ',epsilon:'ε',lambda:'λ',mu:'μ',pi:'π',sigma:'σ',tau:'τ',phi:'φ',psi:'ψ',omega:'ω',infty:'∞'};
function escapeRe(s:string){return s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');}
/** Marks search terms like a highlighter on a printed page. */
function highlight(text:string,terms:string[]):ReactNode{
 if(!terms.length)return text;
 const parts=text.split(new RegExp(`(${terms.map(escapeRe).join('|')})`,'gi'));
 return parts.map((part,i)=>i%2?<mark key={i}>{part}</mark>:part);
}
