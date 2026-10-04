/** Only the dedicated public PDF prefix is discoverable; other bucket objects stay private. */
export const publicPdfKey = /^papers\/([a-f0-9]{64})\.pdf$/;
export type DiscoveryMetadata = {forum_id?:string;number?:number;keywords?:string[];primary_area?:string;tldr?:string;abstract_preview?:string;published?:number;updated?:number};
export type PublishedPaper = DiscoveryMetadata & {id:string;title:string;filename:string;collection:string;bytes:number;classified:boolean;models:string[];score_summaries?:Record<string,unknown>;pdf_key:string;detail_key?:string};
type ArchiveRow = DiscoveryMetadata & {key?:string;title?:string;filename?:string;conference?:string;year?:number;collection?:string};
type PdfObject = {key:string;size:number;customMetadata?:Record<string,string>};
export function mergeArchive(published:PublishedPaper[], objects:PdfObject[], rows:ArchiveRow[]):PublishedPaper[] {
  const items=published.map(p=>({...p})), keys=new Set(items.map(p=>p.pdf_key)), ids=new Set(items.map(p=>p.id));
  const metadata=new Map(rows.filter(r=>r&&typeof r.key==='string').map(r=>[r.key,r]));
  const discovery=(row:ArchiveRow|undefined):DiscoveryMetadata=>row?{forum_id:row.forum_id,number:row.number,keywords:row.keywords,primary_area:row.primary_area,tldr:row.tldr,abstract_preview:row.abstract_preview,published:row.published,updated:row.updated}:{};
  for(const item of items)Object.assign(item,discovery(metadata.get(item.pdf_key)));
  for(const object of objects){
    const match=publicPdfKey.exec(object.key);
    if(!match||keys.has(object.key))continue;
    const hash=match[1],id=hash.slice(0,24);
    if(ids.has(id))throw new Error('PDF identifier collision');
    const row=metadata.get(object.key),meta=object.customMetadata;
    const filename=row?.filename||meta?.filename||`${hash}.pdf`;
    const collection=row?.collection||(row?.conference&&row.year?`${row.conference}/${row.year}`:null)||meta?.collection||'Other uploads';
    items.push({...discovery(row),id,title:row?.title||meta?.title||filename.replace(/\.pdf$/i,''),filename,collection,bytes:object.size,classified:false,models:[],pdf_key:object.key});
    keys.add(object.key);ids.add(id);
  }
  return items;
}
