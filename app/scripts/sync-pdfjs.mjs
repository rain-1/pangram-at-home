// Serve a matching local worker and fonts; no third-party CDN is needed.
import { cpSync, mkdirSync } from "node:fs";
const source = new URL("../node_modules/pdfjs-dist/", import.meta.url);
const target = new URL("../public/pdfjs/", import.meta.url);
mkdirSync(target, { recursive: true });
cpSync(
  new URL("build/pdf.worker.min.mjs", source),
  new URL("pdf.worker.min.mjs", target),
);
for (const folder of ["cmaps", "standard_fonts", "wasm"])
  cpSync(new URL(folder, source), new URL(folder, target), { recursive: true });
