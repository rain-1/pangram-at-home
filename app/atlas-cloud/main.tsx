import React, {lazy,Suspense} from "react";
import { createRoot } from "react-dom/client";
import PaperBrowser from "../components/paper-browser";
import "./base.css";
const PaperAtlas=lazy(()=>import('../components/paper-atlas'));
const ModelCalibration=lazy(()=>import('../components/model-calibration'));
const view=new URLSearchParams(location.search).get("view");
createRoot(document.getElementById("root")!).render(view==="calibration"?<Suspense fallback={<p>Opening calibration…</p>}><ModelCalibration/></Suspense>:view==="classifications"?<Suspense fallback={<p>Opening classifications…</p>}><PaperAtlas hosted browse/></Suspense>:<PaperBrowser/>);
