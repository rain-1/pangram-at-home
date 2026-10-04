"use client";
/* eslint-disable @next/next/no-html-link-for-pages */
import {useEffect,useState,type ReactNode} from 'react';
import {ArrowRight} from 'lucide-react';
import './site.css';

export type SitePage='discover'|'saved'|'classifications';
type NavTarget={href:string;onClick?:()=>void};

/** Number of papers in the browser-local reading list (shared with the discover page). */
export function useSavedCount(){
 const [count,setCount]=useState(0);
 useEffect(()=>{const read=()=>{try{const s=JSON.parse(localStorage.getItem('atlas-reading-list')||'[]');setCount(Array.isArray(s)?s.length:0);}catch{setCount(0);}};read();window.addEventListener('storage',read);return()=>window.removeEventListener('storage',read);},[]);
 return count;
}

function NavItem({target,current,children}:{target:NavTarget;current:boolean;children:ReactNode}){
 return <a href={target.href} aria-current={current?'page':undefined} onClick={target.onClick?e=>{e.preventDefault();target.onClick!();}:undefined}>{children}</a>;
}

/** One header for every public page, so Discover and Classifications read as one site. */
export function SiteHeader({page,savedCount,discover={href:'/'},saved={href:'/?saved=1'},classifications={href:'/?view=classifications'},browse=true,extra}:{page:SitePage;savedCount:number;discover?:NavTarget;saved?:NavTarget;classifications?:NavTarget;browse?:boolean;extra?:ReactNode}){
 return <header className="site-nav"><a href="/" className="site-brand" onClick={discover.onClick&&browse?e=>{e.preventDefault();discover.onClick!();}:undefined}><span className="site-mark" aria-hidden>p.</span>Paper Atlas</a>
  <nav aria-label="Primary">
   {browse&&<NavItem target={discover} current={page==='discover'}>Discover</NavItem>}
   {browse&&<NavItem target={saved} current={page==='saved'}><span className="site-long">Reading list</span><span className="site-short">Saved</span> <span className="site-count">{savedCount}</span></NavItem>}
   <NavItem target={classifications} current={page==='classifications'}>Classifications{page!=='classifications'&&<ArrowRight size={14}/>}</NavItem>
   {extra}
  </nav></header>;
}

export function SiteFooter({children}:{children?:ReactNode}){
 return <footer className="site-footer"><span>Paper Atlas</span><span>Metadata comes from OpenReview; inclusion does not imply acceptance. Classifications are model outputs, not conclusions about authorship.{children}</span></footer>;
}
