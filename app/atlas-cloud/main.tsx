import React, {lazy,Suspense} from "react";
import { createRoot } from "react-dom/client";
import PaperBrowser from "../components/paper-browser";
import "./base.css";
const PaperAtlas=lazy(()=>import('../components/paper-atlas'));
createRoot(document.getElementById("root")!).render(new URLSearchParams(location.search).get("view")==="classifications"?<Suspense fallback={<p>Opening classifications…</p>}><PaperAtlas hosted browse/></Suspense>:<PaperBrowser/>);
