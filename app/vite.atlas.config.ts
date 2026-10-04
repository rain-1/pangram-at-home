import { defineConfig } from "vite";
import {readFileSync} from "node:fs";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
export default defineConfig({
  root: fileURLToPath(new URL("./atlas-cloud", import.meta.url)),
  publicDir: fileURLToPath(new URL("./.sites-runtime/atlas/assets", import.meta.url)),
  plugins: [react(),{name:"atlas-security-headers",generateBundle(){this.emitFile({type:"asset",fileName:"_headers",source:readFileSync(new URL("./atlas-cloud/_headers",import.meta.url),"utf8")});}}],
  resolve: { alias: { "@": fileURLToPath(new URL("./", import.meta.url)) } },
  build: {outDir: "../dist-atlas", emptyOutDir: true, minify: true},
});
