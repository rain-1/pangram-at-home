"use client";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Check,
  ChevronLeft,
  ChevronRight,
  Download,
  Eye,
  EyeOff,
  FileText,
  Loader2,
  Minus,
  Plus,
  Search,
} from "lucide-react";
import { hasTextContent, scoreBand, summarizePassages, paperRate, scoreThresholds } from "@/lib/paper-scores";
import { validateReaderDetail, type ReaderDetail as Detail } from "@/lib/pdf-reader-data";
import { acquirePdfRender, rasterSize, abortable } from "@/lib/pdf-render-budget";
import { mergeHighlightBoxes } from "@/lib/pdf-highlight-boxes";
import legacyAliases from "@/lib/legacy-paper-aliases.json";
import type { PDFDocumentProxy } from "pdfjs-dist";
import {
  alignPassages,
  buildIndex,
  
  type Passage,
} from "@/lib/pdf-alignment";
import { positionPassages, supportsSavedPositions, type PositionedPassage as AlignedPassage } from "@/lib/pdf-positions";
import "./pdf-text-layer.css";
import "./pdf-reader.css";

type Paper = {
  id: string;
  title: string;
  filename: string;
  bytes: number;
  classified: boolean;
  collection: string;
};
type PDFLib = typeof import("pdfjs-dist");
type Index = ReturnType<typeof buildIndex>;
const labels: Record<string,string> = {
  ai:"AI", ai_assisted:"AI assisted", ai_evidence:"Elevated evidence",
  low_evidence:"Lower evidence", human:"Human", uncertain:"Uncertain", below_threshold:"Below threshold",
};
const label = (s:Passage) => Object.hasOwn(labels,s.label)?labels[s.label]:s.label;

const EMPTY_PASSAGES: AlignedPassage[] = [];

function PageCanvas({
  pdf,
  lib,
  number,
  width,
  passages,
  selected,
  highlights,
  onSelect,
  focusId, onFocused,
}: {
  pdf: PDFDocumentProxy;
  lib: PDFLib;
  number: number;
  width: number;
  passages: AlignedPassage[];
  selected: number | null;
  focusId: number | null;
  onFocused: (id:null)=>void;
  highlights: boolean;
  onSelect: (id: number) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [pageImage, setPageImage] = useState("");
  const [boxes, setBoxes] = useState<
    {
      id: number;
      x: number;
      y: number;
      w: number;
      h: number;
      band: string;
    }[]
  >([]);
  const [retry,setRetry] = useState(0);
  const [busy, setBusy] = useState(true),
    [error, setError] = useState("");
  useEffect(() => {
    const target = host.current!;
    let stopped = false;
    const abort = new AbortController();
    let imageUrl: string | undefined;
    let canvas: HTMLCanvasElement | undefined;
    let renderedPage: Awaited<ReturnType<PDFDocumentProxy["getPage"]>> | undefined;
    let release: (()=>void) | undefined;
    let deadline: ReturnType<typeof setTimeout> | undefined;
    let render:
      | ReturnType<Awaited<ReturnType<PDFDocumentProxy["getPage"]>>["render"]>
      | undefined;
    let layer: InstanceType<PDFLib["TextLayer"]> | undefined;
    // Reset transient UI when the external PDF render task changes.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setBoxes([]);
    setPageImage("");
    setBusy(true);
    setError("");
    target.querySelector(".textLayer")?.remove();
    (async () => {
      release = await acquirePdfRender(abort.signal);
      if(stopped)return;
      deadline=setTimeout(()=>{render?.cancel();layer?.cancel();abort.abort();},45000);
      const page = await abortable(pdf.getPage(number),abort.signal);
      renderedPage=page;
      if (stopped) return;
      const viewport = page.getViewport({
        scale: width / page.getViewport({ scale: 1 }).width,
      });
      target.style.width = `${viewport.width}px`;
      target.style.height = `${viewport.height}px`;
      target.style.setProperty("--scale-factor", String(viewport.scale));
      target.style.setProperty("--total-scale-factor", String(viewport.scale));
      canvas = document.createElement("canvas");
      const budget=rasterSize(viewport.width,viewport.height,window.devicePixelRatio);
      const ratio=budget.ratio;
      canvas.width = budget.width;
      canvas.height = budget.height;
      canvas.style.width = `${viewport.width}px`;
      canvas.style.height = `${viewport.height}px`;
      render = page.render({
        canvas,
        viewport,
        transform: [ratio, 0, 0, ratio, 0, 0],
      });
      await render.promise;
      if (stopped) return;
      const blob=await abortable(new Promise<Blob>((resolve,reject)=>canvas!.toBlob(value=>value?resolve(value):reject(new Error("Unable to render PDF image")),"image/png")),abort.signal);
      if(stopped)return;
      imageUrl=URL.createObjectURL(blob);
      setPageImage(imageUrl);
      canvas.width = 0; canvas.height = 0;
      const content = await abortable(page.getTextContent(),abort.signal);
      if (stopped) return;
      const text = document.createElement("div");
      text.className = "textLayer";
      target.appendChild(text);
      // Dense plots may encode thousands of dot glyphs. Avoid a huge invisible
      // text layer when saved positions already provide the highlight geometry.
      if(content.items.length<=3000) {
      layer = new lib.TextLayer({
        textContentSource: content,
        container: text,
        viewport,
      });
      await abortable(layer.render(),abort.signal);
      }
      if (stopped) return;
      const bounds = target.getBoundingClientRect();
      const rects: typeof boxes = [];
      for (const passage of passages) {
        for (const rect of passage.rectangles || []) {
          if(rect.page===number)rects.push({id:passage.id,x:rect.x*viewport.width,y:rect.y*viewport.height,w:rect.w*viewport.width,h:rect.h*viewport.height,band:scoreBand(passage.score)});
        }
        for (const piece of passage.pieces) {
          if (piece.page !== number) continue;
          const span = layer?.textDivs[piece.item];
          if (!span?.firstChild || piece.end > (span.textContent?.length || 0))
            continue;
          const range = document.createRange();
          range.setStart(span.firstChild, piece.start);
          range.setEnd(span.firstChild, piece.end);
          for (const rect of Array.from(range.getClientRects()))
            if (rect.width > 0 && rect.height > 0) {
              rects.push({
                id: passage.id,
                x: rect.left - bounds.left,
                y: rect.top - bounds.top,
                w: rect.width,
                h: rect.height,
                band: scoreBand(passage.score),
              });
            }
        }
      }
      setBoxes(mergeHighlightBoxes(rects));
      setBusy(false);
    })().catch((e) => {
      if (!stopped) {
        setError(abort.signal.aborted ? "This page took too long to render. Try again." : e.message || "This page could not be rendered.");
        setBusy(false);
      }
    }).finally(()=>{
      clearTimeout(deadline);
      if(canvas){canvas.width=0;canvas.height=0;}
      try { renderedPage?.cleanup(); } catch { /* Document may already be closed. */ }
      release?.();
    });
    return () => {
      stopped = true;
      abort.abort();
      render?.cancel();
      layer?.cancel();
      target.querySelector(".textLayer")?.remove();
      if(imageUrl)URL.revokeObjectURL(imageUrl);
    };
  }, [pdf, lib, number, width, passages,retry]);
  useEffect(() => {
    if (focusId === null || focusId !== selected) return;
    const box = boxes.find((b) => b.id === focusId);
    const scroller = host.current?.closest(".pr-viewport");
    if (box && scroller) {
      onFocused(null);
      scroller.scrollTo({
        top: Math.max(0, scroller.scrollTop + host.current!.getBoundingClientRect().top - scroller.getBoundingClientRect().top + box.y - scroller.clientHeight / 3),
        behavior: "instant",
      });
    }
  }, [boxes, selected, focusId, onFocused]);
  return (
    <div className="pr-page-wrap" style={{ width }}>
      {busy && (
        <div className="pr-page-status">
          <Loader2 className="pr-spin" size={18} /> Rendering page {number}
        </div>
      )}
      {error && (
        <div role="alert" className="pr-error">
          {error} <button onClick={()=>setRetry(n=>n+1)}>Retry page</button>
        </div>
      )}
      <div className="pr-page" ref={host}>
        {/* A single rendered page; PDF text and annotations remain separate layers. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        {pageImage && <img src={pageImage} alt={`PDF page ${number}`} style={{display:"block",width:"100%"}} />}
      </div>
      <div className="pr-overlays">
        {boxes
          .filter(() => highlights)
          .map((b, i) => (
            <button
              key={i}
              tabIndex={-1}
              aria-label={`Inspect passage ${b.id + 1}`}
              title={`Passage ${b.id + 1}`}
              className={`pr-highlight ${b.band} ${selected === b.id ? "selected" : ""}`}
              style={{ left: b.x, top: b.y, width: b.w, height: b.h }}
              onClick={() => onSelect(b.id)}
            />
          ))}
      </div>
    </div>
  );
}

// Keep page-sized placeholders so scrolling stays continuous while only nearby pages render.
const BAND_COLORS: Record<string,string> = {low:"#8ebd95", medium:"#ddbc52", high:"#d97669", unknown:"#cfdacc"};
/** One vertical strip per page: each passage gets an equal slice, in reading order, colored by score band. */
function pageStrip(local: {score:number}[]) {
  if (!local.length) return "transparent";
  const step = 100 / local.length;
  return `linear-gradient(to bottom, ${local.map((p, k) => `${BAND_COLORS[scoreBand(p.score)]} ${(k*step).toFixed(2)}% ${((k+1)*step).toFixed(2)}%`).join(", ")})`;
}

function ScrollingPage({ratios,...props}: React.ComponentProps<typeof PageCanvas> & {ratios:Map<number,number>}) {
  const slot = useRef<HTMLDivElement>(null);
  const [nearby, setNearby] = useState(false);
  const [ratio, setRatio] = useState(()=>ratios.get(props.number)||792 / 612);
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => setNearby(entry.isIntersecting), {
      root: slot.current!.closest(".pr-viewport"), rootMargin: "1000px 0px",
    });
    observer.observe(slot.current!);
    return () => observer.disconnect();
  }, [props.pdf, props.number]);
  useEffect(()=>{
    if(!nearby)return;
    let active=true;
    props.pdf.getPage(props.number).then(page=>{
      const size=page.getViewport({scale:1});
      if(active && Number.isFinite(size.height/size.width)){ratios.set(props.number,size.height/size.width);setRatio(size.height/size.width);}
    }).catch(()=>{});
    return ()=>{active=false;};
  },[nearby,props.pdf,props.number,ratios]);
  return <div ref={slot} className="pr-page-slot" data-page={props.number}
    aria-label={`PDF page ${props.number}`} style={{width: props.width, minHeight: props.width * ratio}}>
    {nearby ? <PageCanvas {...props} /> : <div className="pr-page-placeholder">Page {props.number}</div>}
  </div>;
}

export default function PDFReader({
  api,
  availablePapers,
  initialPaperId = "",
  embedded = false,
  hosted = false,
  readingOnly = false,
  preferredModel,
  scoreThreshold = .8,
  fileBasePath = "/backend/v1/pdf-reader",
}: {
  availablePapers?: Paper[];
  initialPaperId?: string;
  embedded?: boolean;
  hosted?: boolean;
  readingOnly?: boolean;
  preferredModel?: string;
  scoreThreshold?: number;
  fileBasePath?: string;
  api: (path: string, init?:RequestInit) => Promise<unknown>;
}) {
  const [papers, setPapers] = useState<Paper[]>([]),
    [paperId, setPaperId] = useState("");
  const [detail, setDetail] = useState<Detail | null>(null),
    [reportId, setReportId] = useState("");
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null),
    [lib, setLib] = useState<PDFLib | null>(null);
  const [index, setIndex] = useState<Index | null>(null),
    [page, setPage] = useState(1);
  const [selected, setSelected] = useState<number | null>(null),
    [highlights, setHighlights] = useState(true);
  const [mode, setMode] = useState<"pdf" | "text">("pdf"),
    [filter, setFilter] = useState("all");
  const [passageSort,setPassageSort] = useState("order");
  const [threshold,setThreshold] = useState(scoreThreshold);
  const [focusId,setFocusId] = useState<number|null>(null);
  const [pageInput,setPageInput] = useState("");
  const [loadAttempt,setLoadAttempt] = useState(0);
  const [textSection,setTextSection] = useState(0);
  const [sidebarPage,setSidebarPage] = useState(0);
  const [search, setSearch] = useState(""),
    [error, setError] = useState(""),
    [progress, setProgress] = useState("");
  const [width, setWidth] = useState(720),
    [zoom, setZoom] = useState(1);
  const viewport = useRef<HTMLDivElement>(null);
  const viewScroll=useRef({pdf:0,text:0});
  // Keep measurements across view switches, but discard them for a new PDF.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const pageRatios=useMemo(()=>new Map<number,number>(),[pdf]);
  function switchMode(next:"pdf"|"text"){
    if(next===mode)return;
    viewScroll.current[mode]=viewport.current?.scrollTop||0;
    setFocusId(null);
    if(next==="text" && selected!==null)setTextSection(Math.floor(selected/100));
    setMode(next);
  }
  useLayoutEffect(()=>{
    viewport.current?.scrollTo({top:viewScroll.current[mode],behavior:"instant"});
  },[mode]);
  useEffect(() => {
    let active = true;
    const abort=new AbortController();
    (availablePapers ? Promise.resolve({items:availablePapers}) : api("/v1/pdf-reader",{signal:abort.signal}))
      .then((payload) => {
        const data = payload as { items: Paper[] };
        if (active) {
          setPapers(data.items);
          const resolvedId=data.items.some(p=>p.id===initialPaperId) ? initialPaperId : (legacyAliases as Record<string,string>)[initialPaperId] || initialPaperId;
          if (resolvedId && !data.items.some(p => p.id === resolvedId)) {
            setError("This paper is no longer in the local collection.");
            return;
          }
          setPaperId(resolvedId || data.items[0]?.id || "");
          if (!data.items.length)
            setProgress("No local PDFs found in the research data folder.");
        }
      })
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
      abort.abort();
    };
  }, [api, initialPaperId,loadAttempt,availablePapers]);
  useEffect(() => {
    if (!paperId) return;
    let active = true;
    const abort=new AbortController();
    let task: ReturnType<PDFLib["getDocument"]> | undefined;
    let timedOut=false;
    const deadline=setTimeout(()=>{
      timedOut=true;abort.abort();void task?.destroy().catch(()=>{});
      if(active){setError("Loading this paper timed out. Please retry.");setProgress("");}
    },45000);
    // A new external document invalidates all page and alignment state.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setPdf(null);
    setIndex(null);
    setDetail(null);
    viewScroll.current={pdf:0,text:0};
    setMode("pdf");
    setTextSection(0);
    setSidebarPage(0);
    setFilter("all");
    setPage(1);
    setSelected(null);
    setError("");
    setProgress(hosted ? "Opening archived PDF…" : "Opening local PDF…");
    (async () => {
      const [payload, library] = await Promise.all([
        api(`/v1/pdf-reader/${paperId}`,{signal:abort.signal}),
        import("pdfjs-dist"),
      ]);
      const data = validateReaderDetail(payload);
      if (!active || timedOut) return;
      library.GlobalWorkerOptions.workerSrc = "/pdfjs/pdf.worker.min.mjs";
      setDetail(data);
      setReportId(data.reports.find(r=>preferredModel && r.model.name.toLowerCase().includes(preferredModel))?.id || data.reports[0]?.id || "");
      setLib(library);
      task = library.getDocument({
        url: `${fileBasePath}/${paperId}/file?v=${encodeURIComponent(data.version)}`,
        cMapUrl: "/pdfjs/cmaps/",
        cMapPacked: true,
        standardFontDataUrl: "/pdfjs/standard_fonts/",
        wasmUrl: "/pdfjs/wasm/",
        isEvalSupported: false,
        disableAutoFetch: true,
        disableStream: true,
      });
      const document = await task.promise;
      if (!active || timedOut) return;
      setPdf(document);
      if (readingOnly || !data.reports.length || data.reports.every(r=>supportsSavedPositions(data.position_maps?.[r.text_sha256 || ""]))) {
        setProgress("");
        return;
      }
      const pages: string[][] = [];
      for (let n = 1; n <= document.numPages; n++) {
        if (!active || timedOut) return;
        setProgress(
          `Locating saved passages · page ${n} of ${document.numPages}`,
        );
        const sourcePage = await abortable(document.getPage(n),abort.signal);
        const content = await abortable(sourcePage.getTextContent(),abort.signal);
        sourcePage.cleanup();
        pages.push(
          content.items
            .filter((item) => "str" in item)
            .map((item) => ("str" in item ? item.str : "")),
        );
      }
      if (active && !timedOut) {
        setIndex(buildIndex(pages));
        setProgress("");
      }
    })().catch((e) => {
      if (active) {
        setError(timedOut ? "Loading this paper timed out. Please retry." : e.message || "Could not open PDF");
        setProgress("");
      }
    }).finally(()=>clearTimeout(deadline));
    return () => {
      active = false;
      clearTimeout(deadline);
      abort.abort();
      void task?.destroy().catch(()=>{});
    };
  }, [api, paperId, hosted, preferredModel,loadAttempt,fileBasePath,readingOnly]);
  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const resize = new ResizeObserver((entries) =>
      setWidth(Math.max(120, Math.min(900, entries[0].contentRect.width))),
    );
    resize.observe(element);
    return () => resize.disconnect();
  }, []);
  const paper = papers.find((p) => p.id === paperId);
  const report = detail?.reports.find((r) => r.id === reportId);
  const passages = useMemo<AlignedPassage[]>(
    () =>
      report && supportsSavedPositions(detail?.position_maps?.[report.text_sha256 || ""])
        ? positionPassages(report.text, report.result.segments, detail!.position_maps![report.text_sha256!])
        : report && index ? alignPassages(report.text, report.result.segments, index) : [],
    [report, index, detail],
  );
  // First passage located on each new PDF page, for soft page boundaries in text mode.
  const pageStarts=useMemo(()=>{const starts=new Map<number,number>();let last=0;
    for(const p of passages)if(p.page!==undefined&&p.page>last){starts.set(p.id,p.page);last=p.page;}
    return starts;},[passages]);
  const section=Math.min(textSection,Math.max(0,Math.ceil(passages.length/100)-1));
  const textStart=section*100, textEnd=Math.min(passages.length,textStart+100);
  useEffect(()=>{
    if(mode==="text" && selected!==null)viewport.current?.querySelector(`[data-passage="${selected}"]`)?.scrollIntoView({block:"center",behavior:"instant"});
  },[mode,selected,section]);
  // Page-map jumps in text mode: scroll the target passage to the top once its section has rendered.
  const [textTarget,setTextTarget]=useState<number|null>(null);
  useEffect(()=>{
    if(mode!=="text"||textTarget===null)return;
    const el=viewport.current?.querySelector<HTMLElement>(`[data-passage="${textTarget}"]`);
    // Land on the page boundary line when the passage starts a page.
    const target=el?.previousElementSibling?.classList.contains("pr-page-break")?el.previousElementSibling as HTMLElement:el;
    const scroller=viewport.current;
    if(target&&scroller){
      // Scroll only the viewer when it scrolls; otherwise let the page scroll.
      if(scroller.scrollHeight>scroller.clientHeight+1)scroller.scrollTo({top:scroller.scrollTop+target.getBoundingClientRect().top-scroller.getBoundingClientRect().top-16,behavior:"instant"});
      else target.scrollIntoView({block:"start",behavior:"instant"});
      setTextTarget(null);
    }
  },[mode,textTarget,section]);
  const current = selected === null ? undefined : passages[selected];
  const sourceChars = useMemo(()=>Array.from(report?.text || ""),[report?.text]);
  const reportSummary = useMemo(()=>summarizePassages(report?.result.segments || [],report?.text),[report]);
  const highlightPassages=useMemo(()=>passages.filter(p=>hasTextContent(p.text)),[passages]);
  const pagePassages=useMemo(()=>{
    const pages=new Map<number,AlignedPassage[]>();
    for(const passage of highlightPassages){
      const numbers=new Set([...passage.pieces.map(p=>p.page),...(passage.rectangles||[]).map(r=>r.page)]);
      for(const n of numbers){const group=pages.get(n)||[];group.push(passage);pages.set(n,group);}
    }
    return pages;
  },[highlightPassages]);
  const mapped = useMemo(()=>passages.filter((p) => p.mapped).length,[passages]);
  const visible = useMemo(()=>passages.filter(
    (p) =>
      filter === "artifacts" ? !hasTextContent(p.text) : hasTextContent(p.text) && (filter === "all" || (filter === "elevated" ? p.score > threshold : !p.mapped)),
  ).sort((a,b)=>passageSort==="high"?b.score-a.score||a.id-b.id:passageSort==="low"?a.score-b.score||a.id-b.id:a.id-b.id),[passages,filter,threshold,passageSort]);
  // Passage list follows the reading position unless the reader scrolls the list themselves.
  const inspector=useRef<HTMLElement>(null);
  const [follow,setFollow]=useState(true);
  const [readingId,setReadingId]=useState<number|null>(null);
  const followActive=follow&&passageSort==="order"&&selected===null;
  const followIndex=followActive&&readingId!==null?visible.findIndex(p=>p.id>=readingId):-1;
  const sidebarSection=followIndex>=0?Math.floor(followIndex/100):Math.min(sidebarPage,Math.max(0,Math.ceil(visible.length/100)-1));
  const sidebarStart=sidebarSection*100, sidebarEnd=Math.min(visible.length,sidebarStart+100);
  // First rectangle of each passage per page, in document order, for locating the reading line in PDF mode.
  const pageAnchors=useMemo(()=>{const anchors=new Map<number,{y:number;id:number}[]>();
    for(const p of passages){const seen=new Set<number>();for(const r of p.rectangles||[]){if(seen.has(r.page))continue;seen.add(r.page);
      if(!anchors.has(r.page))anchors.set(r.page,[]);anchors.get(r.page)!.push({y:r.y,id:p.id});}}
    return anchors;},[passages]);
  const readingFrame=useRef(0);
  // The panel only starts following once the reader has moved through the paper, so its summary stays visible on open.
  const moved=useRef(false);
  function trackReading(){
    cancelAnimationFrame(readingFrame.current);
    readingFrame.current=requestAnimationFrame(()=>{
      const scroller=viewport.current;if(!scroller)return;
      // Reading line: 30% down whichever box is actually scrolling (the viewer, or the window below the site nav).
      const own=scroller.scrollHeight>scroller.clientHeight+1, rect=scroller.getBoundingClientRect();
      const nav=parseFloat(getComputedStyle(scroller).getPropertyValue("--nav"))||0;
      const top=own?rect.top:Math.max(rect.top,nav), bottom=own?rect.bottom:Math.min(rect.bottom,window.innerHeight);
      const line=top+(bottom-top)*.3;
      let id:number|undefined;
      if(mode==="text"){
        const el=Array.from(scroller.querySelectorAll<HTMLElement>("[data-passage]")).find(e=>e.getBoundingClientRect().bottom>line);
        if(el)id=Number(el.dataset.passage);
      }else{
        const slot=Array.from(scroller.querySelectorAll<HTMLElement>("[data-page]")).find(e=>e.getBoundingClientRect().bottom>line);
        if(slot){const n=Number(slot.dataset.page),r=slot.getBoundingClientRect(),rel=Math.min(1,Math.max(0,(line-r.top)/Math.max(1,r.height)));
          for(let pg=n;pg>=1&&id===undefined;pg--){const list=pageAnchors.get(pg);if(!list?.length)continue;
            if(pg<n){id=list[list.length-1].id;break;}
            const before=list.filter(a=>a.y<=rel);id=(before.length?before[before.length-1]:list[0]).id;}}
      }
      if(id!==undefined)setReadingId(id);
    });
  }
  useEffect(()=>{const onScroll=()=>{moved.current=true;trackReading();};window.addEventListener("scroll",onScroll,{passive:true});return()=>window.removeEventListener("scroll",onScroll);});
  // Locate the reading position on open and after switching views, before any scroll happens.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(()=>{const t=setTimeout(trackReading,400);return()=>clearTimeout(t);},[mode,passages,pdf,section]);
  useEffect(()=>{
    if(followIndex<0||!moved.current)return;
    const box=inspector.current,el=box?.querySelector<HTMLElement>(`[data-list-passage="${visible[followIndex].id}"]`);
    if(!box||!el||box.scrollHeight<=box.clientHeight+1)return;
    box.scrollTo({top:box.scrollTop+el.getBoundingClientRect().top-box.getBoundingClientRect().top-56,behavior:"smooth"});
  },[followIndex,visible,sidebarSection]);
  const readingListId=followIndex>=0?visible[followIndex].id:undefined;
  function select(id: number) {
    const p = passages[id];
    if(!p)return;
    setSelected(id);
    setFocusId(mode==="pdf"?id:null);
    if(!hasTextContent(p.text)) {setFocusId(null);if(mode==="text")setTextSection(Math.floor(id/100));return;}
    if (p.page && mode === "pdf") jumpToPage(p.page);
    if (mode === "text") setTextSection(Math.floor(id/100));
  }
  function jumpToPage(n: number) {
    if(!pdf || !Number.isInteger(n) || n<1 || n>pdf.numPages)return;
    const scroller = viewport.current;
    const target = scroller?.querySelector<HTMLElement>(`[data-page="${n}"]`);
    // Narrow layouts let the page itself scroll; then scroll whichever ancestor can.
    if (scroller && target && scroller.scrollHeight <= scroller.clientHeight + 1) target.scrollIntoView({block: "start", behavior: "instant"});
    else if (scroller && target) scroller.scrollTo({
      top: scroller.scrollTop + target.getBoundingClientRect().top - scroller.getBoundingClientRect().top - 24,
      behavior: "instant",
    });
    setPage(n);
  }
  function changePage(n: number) {
    setSelected(null);
    setFocusId(null);
    if (mode === "text") {
      // Text mode has no page elements: jump to the first passage located on that page.
      const first = passages.find(p => p.page === n && hasTextContent(p.text)) ?? passages.find(p => (p.page ?? 0) >= n);
      setPage(n);
      if (first) { setTextSection(Math.floor(first.id/100)); setTextTarget(first.id); }
      return;
    }
    jumpToPage(n);
  }
  function trackPage() {
    const scroller = viewport.current;
    if (!scroller || mode !== "pdf") return;
    const threshold = scroller.getBoundingClientRect().top + scroller.clientHeight * .3;
    const slots = Array.from(scroller.querySelectorAll<HTMLElement>("[data-page]"));
    const active = slots.find(slot => slot.getBoundingClientRect().bottom > threshold);
    if (active) setPage(Number(active.dataset.page));
  }
  const options = embedded ? [] : papers.filter(
    (p) =>
      `${p.title} ${p.filename} ${p.collection}`
        .toLowerCase()
        .includes(search.toLowerCase()) || p.id === paperId,
  );
  return (
    <section className={`pr-shell ${embedded ? "pr-embedded" : ""}`}>
      {!embedded&&<header className="pr-heading">
        <div>
          <div className="pr-eyebrow">
            PANGRAM / READING ROOM <span>LOCAL PROTOTYPE</span>
          </div>
          <h1>Read the paper. See the evidence.</h1>
          <p>
            Original pages, saved classifications, and the context between them.
          </p>
        </div>
        <div className="pr-local">
          <span /> {papers.length} local PDFs
        </div>
      </header>}
      {!embedded && <div className="pr-picker">
        <label className="pr-search">
          <Search size={16} />
          <input
            aria-label="Find a local PDF"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Find a local paper…"
          />
        </label>
        <select
          aria-label="Local paper"
          value={paperId}
          onChange={(e) => setPaperId(e.target.value)}
        >
          {options.map((p) => (
            <option value={p.id} key={p.id}>
              {p.classified ? "● " : ""}
              {p.title} · {(p.bytes / 1e6).toFixed(1)} MB
            </option>
          ))}
        </select>
      </div>}
      {error && (
        <div role="alert" className="pr-error">
          {error} <button onClick={()=>setLoadAttempt(n=>n+1)}>Retry paper</button>
        </div>
      )}
      <div className="pr-document-heading">
        <div>
          <h2>{paper?.title || "Choose a paper"}</h2>
          <p>
            {paper?.collection.replace(/^(\w+)\/(\d{4})$/,(_,v:string,y:string)=>`${v.toUpperCase()} ${y}`).replaceAll("_", " ")} <span>·</span>{" "}
            {paper ? `${(paper.bytes / 1e6).toFixed(1)} MB` : ""}{" "}
            {pdf && (
              <>
                {" "}
                <span>·</span> {pdf.numPages} pages
              </>
            )}
          </p>
        </div>
        {paper && (
          <a
            href={`${fileBasePath}/${paper.id}/file`}
            download={paper.filename}
          >
            <Download size={15} /> Original PDF
          </a>
        )}
      </div>
      <div className={`pr-workspace ${readingOnly?"pr-reading-only":""}`}>
        <nav className="pr-map" aria-label="Page overview">
          <span>PAGES</span>
          {pdf &&
            Array.from({ length: pdf.numPages }, (_, i) => {
              const local = pagePassages.get(i+1)||EMPTY_PASSAGES;
              return (
                <button
                  key={i}
                  aria-label={`Page ${i + 1}, ${local.length} mapped passages`}
                  aria-current={page === i + 1 ? "page" : undefined}
                  className={page === i + 1 ? "active" : ""}
                  onClick={() => changePage(i + 1)}
                >
                  <b>{pdf.numPages <= 30 || i === 0 || (i + 1) % 5 === 0 ? i + 1 : ""}</b>
                  <i style={{background: pageStrip(local)}} />
                </button>
              );
            })}
        </nav>
        <div className="pr-reading">
          <div className="pr-toolbar">
            <div className="pr-toggle">
              <button
                className={mode === "pdf" ? "active" : ""}
                onClick={() => switchMode("pdf")}
              >
                <FileText size={14} /> PDF
              </button>
              <button
                className={mode === "text" ? "active" : ""}
                onClick={() => switchMode("text")}
                disabled={!report}
              >
                <BookOpen size={14} /> Text
              </button>
            </div>
            {!readingOnly&&<button
              aria-pressed={highlights}
              onClick={() => setHighlights(!highlights)}
            >
              {highlights ? <Eye size={15} /> : <EyeOff size={15} />} Highlights
            </button>}
            <div className="pr-zoom">
              <button
                aria-label="Zoom out"
                disabled={zoom <= 0.7}
                onClick={() => setZoom((z) => Math.max(0.7, z - 0.15))}
              >
                <Minus size={14} />
              </button>
              <span>{Math.round(zoom * 100)}%</span>
              <button
                aria-label="Zoom in"
                disabled={zoom >= 1.6}
                onClick={() => setZoom((z) => Math.min(1.6, z + 0.15))}
              >
                <Plus size={14} />
              </button>
            </div>
          </div>
          <div className="pr-viewport" ref={viewport} onScroll={()=>{moved.current=true;trackPage();trackReading();}}>
            {progress && (
              <div className="pr-progress">
                <Loader2 size={14} className="pr-spin" />
                {progress}
              </div>
            )}
            {mode === "pdf" && pdf && lib && (
              <div className="pr-page-stack">
                {Array.from({length: pdf.numPages}, (_, i) => (
                  <ScrollingPage ratios={pageRatios} key={`${paperId}-${i}`} pdf={pdf} lib={lib} number={i + 1}
                    width={Math.round(width * zoom)} passages={pagePassages.get(i+1)||EMPTY_PASSAGES} focusId={focusId} onFocused={setFocusId}
                    selected={current?.page === i + 1 ? selected : null}
                    highlights={!readingOnly&&highlights} onSelect={select} />
                ))}
              </div>
            )}
            {mode === "text" && report && (
              <article className="pr-text">
                <small>CLASSIFIED SOURCE TEXT</small>
                <p className="pr-text-note">
                  This is the text the model received. Formatting can differ
                  from the PDF. Long reports are shown in sections of 100 passages.
                </p>
                {passages.length ? (
                  passages.slice(textStart,textEnd).map((p, i) => (
                    <span key={p.id}>
                      {sourceChars
                        .slice(i+textStart ? passages[i+textStart - 1].end : 0, p.start)
                        .join("")}
                      {pageStarts.has(p.id) &&
                        <span className="pr-page-break" data-page-break={pageStarts.get(p.id)} aria-label={`Page ${pageStarts.get(p.id)}`}><span>Page {pageStarts.get(p.id)}</span></span>}
                      <button
                        data-passage={p.id}
                        title={`Passage ${p.id+1} · score ${p.score.toFixed(3)}`}
                        className={`${highlights && hasTextContent(p.text) ? scoreBand(p.score) : ""} ${selected === p.id ? "selected" : ""}`}
                        onClick={() => select(p.id)}
                      >
                        {p.text}
                      </button>
                    </span>
                  ))
                ) : (
                  <p>{report.text}</p>
                )}
                {passages.length > 0 && textEnd===passages.length &&
                  sourceChars
                    .slice(passages[passages.length - 1].end)
                    .join("")}
              </article>
            )}
          </div>
          {mode==="text" ? <footer className="pr-pagination"><button disabled={section===0} aria-label="Previous text section" onClick={()=>{setSelected(null);setTextSection(section-1);viewport.current?.scrollTo({top:0});}}><ChevronLeft size={16}/>Previous</button><span>Passages {passages.length?textStart+1:0}–{textEnd} of {passages.length}</span><button disabled={textEnd===passages.length} aria-label="Next text section" onClick={()=>{setSelected(null);setTextSection(section+1);viewport.current?.scrollTo({top:0});}}>Next<ChevronRight size={16}/></button></footer> : <footer className="pr-pagination">
            <button
              disabled={page <= 1 || !pdf}
              onClick={() => changePage(page - 1)}
            >
              <ChevronLeft size={16} /> Previous
            </button>
            <div>
              <form className="pr-page-jump" onSubmit={e=>{e.preventDefault();const n=Number(pageInput);if(pdf&&Number.isInteger(n)&&n>=1&&n<=pdf.numPages){changePage(n);setPageInput("");}}}><label>Page <input aria-label="PDF page number" type="number" min={1} max={pdf?.numPages||1} required disabled={!pdf} placeholder={String(page)} value={pageInput} onChange={e=>setPageInput(e.target.value)}/></label> of {pdf?.numPages || "—"}<button type="submit" disabled={!pdf}>Go</button></form>
            </div>
            <button
              disabled={!pdf || page >= pdf.numPages}
              onClick={() => changePage(page + 1)}
            >
              Next <ChevronRight size={16} />
            </button>
          </footer>}
        </div>
        {!readingOnly&&<aside className="pr-inspector" ref={inspector}
          onWheel={()=>setFollow(false)} onTouchMove={()=>setFollow(false)}
          onKeyDown={e=>{if(["ArrowUp","ArrowDown","PageUp","PageDown","Home","End"," "].includes(e.key))setFollow(false);}}>
          <div className="pr-inspector-top">
            <div className="pr-eyebrow">MODEL OBSERVATIONS</div>
            <h3>Passage evidence</h3>
            <p>Experimental results, in context.</p>
            {detail && detail.reports.length > 0 && (
              <select
                aria-label="Saved classification"
                value={reportId}
                onChange={(e) => {
                  setReportId(e.target.value);
                  setTextSection(0);
                  setSidebarPage(0);
                  setSelected(null);
                  setFocusId(null);
                }}
              >
                {detail.reports.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.model.name}
                  </option>
                ))}
              </select>
            )}
          </div>
          {!report && detail && (
            <div className="pr-empty">
              <BookOpen size={28} />
              <h4>A clean reading copy</h4>
              <p>
                This PDF has no saved classification. Browse the original pages,
                or choose a paper marked ● to explore real model results.
              </p>
            </div>
          )}
          {report && (
            <>
              {report.result.notice && <p className="pr-mapping-note">{report.result.notice}</p>}
              <div className="pr-coverage">
                <Check size={15} />
                <span>
                  <b>
                    {mapped} / {report.result.segments.length}
                  </b>{" "}
                  classified passages matched to PDF
                </span>
              </div>
              <p className="pr-mapping-note">
                {supportsSavedPositions(detail?.position_maps?.[report.text_sha256 || ""]) ? "Highlights use saved word positions from the verified extraction." : "Only unique text matches are highlighted."} All classified passages remain available below. {detail?.version_verified ? "PDF and classification input hashes verified." : "PDF version correspondence is unverified."}
              </p>
              <div className="pr-paper-score"><strong>{paperRate(reportSummary,threshold)===null?'—':`${(paperRate(reportSummary,threshold)!*100).toFixed(1)}%`}</strong><span>of passages above threshold</span><label>Threshold <select aria-label="Reader score threshold" value={threshold} onChange={e=>setThreshold(Number(e.target.value))}>{scoreThresholds.map(t=><option key={t} value={t}>{t.toFixed(2)}</option>)}</select></label></div>
              {reportSummary.excluded!>0 && <p className="pr-mapping-note">{reportSummary.excluded!.toLocaleString()} punctuation-only fragments excluded from scores and highlights. View them under “Punctuation / artifacts”; selecting them will not navigate the PDF.</p>}
              <div className="pr-legend"><span><i className="low"/> Low ≤ 0.20</span><span><i className="medium"/> Mid ≤ 0.80</span><span><i className="high"/> High &gt; 0.80</span></div><p className="pr-mapping-note">Highlight colors use fixed score bands; the chosen threshold controls counts and filtering.</p>
              {current ? (
                <div className="pr-detail">
                  <div>
                    <span className="pr-pill">{hasTextContent(current.text)?label(current):"Excluded fragment"}</span>
                    <button
                      aria-label="Clear selected passage"
                      onClick={() => setSelected(null)}
                    >
                      ×
                    </button>
                  </div>
                  <h4>
                    Passage {current.id + 1}{" "}
                    <small>
                      {current.mapped
                        ? `· Page ${current.page}`
                        : "· Not located"}
                    </small>
                  </h4>
                  <blockquote>{current.text}</blockquote>
                  <div className="pr-score">
                    <span>Model score</span>
                    <b>{current.score.toFixed(3)}</b>
                  </div>
                  <p>
                    Scores are experimental model outputs, not probabilities of
                    authorship.
                  </p>
                  <div className="pr-detail-nav">
                    <button
                      disabled={current.id === 0}
                      onClick={() => select(current.id - 1)}
                    >
                      <ArrowLeft size={14} /> Previous
                    </button>
                    <button
                      disabled={current.id === passages.length - 1}
                      onClick={() => select(current.id + 1)}
                    >
                      Next <ArrowRight size={14} />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="pr-hint">
                  Select a highlight on the page or a passage below to inspect
                  it.
                </div>
              )}
              <div className="pr-list-head">
                <h4>
                  Passages <span>{visible.length}</span>
                </h4>
                <select
                  aria-label="Filter passages"
                  value={filter}
                  onChange={(e) => {setFilter(e.target.value);setSidebarPage(0);}}
                >
                  <option value="all">Text passages</option><option value="artifacts">Punctuation / artifacts</option>
                  <option value="elevated">Above threshold</option>
                  <option value="unmapped">Not located</option>
                </select>
              </div>
              <div className="pr-follow-row">
                <label className="pr-follow"><input type="checkbox" checked={follow} onChange={e=>setFollow(e.target.checked)}/> Follow reading position</label>
                {follow&&!followActive&&<span>{passageSort!=="order"?"Paused: sorted by score":"Paused while a passage is selected"}</span>}
              </div>
              {!follow&&<button className="pr-follow-resume" onClick={()=>setFollow(true)}>Follow reading</button>}
              <label className="pr-sort">Sort passages <select aria-label="Sort passages" value={passageSort} onChange={e=>{setPassageSort(e.target.value);setSidebarPage(0);}}><option value="order">Document order</option><option value="high">Highest AI score first</option><option value="low">Lowest AI score first</option></select></label>
              <div className="pr-passages">
                {visible.slice(sidebarStart,sidebarEnd).map((p) => (
                  <button
                    key={p.id}
                    data-list-passage={p.id}
                    aria-current={readingListId===p.id?"location":undefined}
                    className={`${hasTextContent(p.text)?scoreBand(p.score):"unknown"} ${selected === p.id ? "active" : ""} ${readingListId===p.id ? "reading" : ""}`}
                    onClick={() => select(p.id)}
                  >
                    <div>
                      <span
                        className={`pr-dot ${hasTextContent(p.text)?scoreBand(p.score):"unknown"}`}
                      />
                      <b>{p.id + 1}</b>
                      <span>{p.mapped ? `p. ${p.page}` : "Not located"}</span>
                      <strong>{p.score.toFixed(2)}</strong>
                    </div>
                    <p>{p.text}</p>
                  </button>
                ))}
                {visible.length>100 && <nav className="pr-sidebar-pages" aria-label="Passage list pages"><button aria-label="Previous passage list page" disabled={sidebarSection===0} onClick={()=>{setFollow(false);setSidebarPage(sidebarSection-1);}}>Previous</button><span>{sidebarStart+1}–{sidebarEnd} of {visible.length}</span><button aria-label="Next passage list page" disabled={sidebarEnd===visible.length} onClick={()=>{setFollow(false);setSidebarPage(sidebarSection+1);}}>Next</button></nav>}
                {!visible.length && !progress && (
                  <p className="pr-hint">No passages match this filter.</p>
                )}
              </div>
            </>
          )}
        </aside>}
      </div>
    </section>
  );
}
